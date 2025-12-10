import os
import logging
from dataclasses import dataclass
from typing import Any, Dict

import httpx
from jose import jwt
from jose.exceptions import JWTError

logger = logging.getLogger(__name__)

KEYCLOAK_ISSUER = os.getenv(
    "KEYCLOAK_ISSUER",
    "https://auth.nanosamur.ai/realms/your-realm-name",
)
KEYCLOAK_AUDIENCE = os.getenv("KEYCLOAK_AUDIENCE", "bff-api")  # your Keycloak client-id

# We cache JWKS in memory; in prod you’d refresh periodically.
_JWKS: Dict[str, Any] | None = None
_JWKS_URL: str | None = None


@dataclass
class OIDCUser:
    sub: str
    preferred_username: str | None
    email: str | None
    raw: Dict[str, Any]


class OIDCError(Exception):
    pass


def _load_jwks() -> Dict[str, Any]:
    global _JWKS, _JWKS_URL

    if _JWKS is not None:
        return _JWKS

    # 1) Discover Keycloak’s OIDC config
    well_known = f"{KEYCLOAK_ISSUER.rstrip('/')}/.well-known/openid-configuration"
    logger.info("Fetching OIDC configuration from %s", well_known)

    try:
        resp = httpx.get(well_known, timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to fetch OIDC config: %s", e)
        raise OIDCError("OIDC configuration fetch failed") from e

    conf = resp.json()
    _JWKS_URL = conf["jwks_uri"]

    # 2) Fetch JWKS
    logger.info("Fetching JWKS from %s", _JWKS_URL)
    try:
        resp = httpx.get(_JWKS_URL, timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to fetch JWKS: %s", e)
        raise OIDCError("JWKS fetch failed") from e

    _JWKS = resp.json()
    return _JWKS


def _get_public_key(token: str) -> Dict[str, Any]:
    """Find the JWK that matches the token's kid."""
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        raise OIDCError("Token missing 'kid' header")

    jwks = _load_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    raise OIDCError("No matching JWK for token kid")


def verify_token(token: str) -> OIDCUser:
    """
    Verify a Keycloak-issued JWT access token.

    Returns an OIDCUser on success, raises OIDCError on failure.
    """
    try:
        jwk = _get_public_key(token)
        # jose can take the JWK directly
        claims = jwt.decode(
            token,
            jwk,
            algorithms=[jwk.get("alg", "RS256")],
            audience=KEYCLOAK_AUDIENCE,
            issuer=KEYCLOAK_ISSUER,
        )
    except JWTError as e:
        logger.warning("JWT verification failed: %s", e)
        raise OIDCError("Invalid token") from e

    sub = claims.get("sub")
    if not sub:
        raise OIDCError("Token missing 'sub' claim")

    user = OIDCUser(
        sub=sub,
        preferred_username=claims.get("preferred_username"),
        email=claims.get("email"),
        raw=claims,
    )
    return user
