#tests/test_bff_session_persistence.py
import pytest
import uuid
from bff.sessions import start_session_db
from drsynth_common.db import fetch_one, exec1

# Use the same tenant_id that is created in conftest.py
TEST_TENANT_ID = "00000000-0000-0000-0000-000000000000"

@pytest.mark.integration
def test_start_session_db_persists_session(postgres_container):
    """Test that start_session_db correctly persists a session to the database."""
    # Test with tenant_id and user_id (as UUID4 strings)
    tenant_id = TEST_TENANT_ID
    user_id = str(uuid.uuid4())

    # Create a test user first
    exec1(
        "INSERT INTO app_users (id, tenant_id, email, name, roles) VALUES (%s, %s, %s, %s, %s)",
        (user_id, tenant_id, "test@example.com", "Test User", "user")
    )

    session_data = start_session_db(x_tenant_id=tenant_id, x_user_id=user_id)

    # Verify the session was created
    session_id = session_data["session_id"]
    session_key = session_data["session_key"]

    # Query the database to verify the session exists
    result = fetch_one(
        "SELECT id, tenant_id, user_id, session_key, status FROM sessions WHERE id = %s",
        (session_id,)
    )

    assert result is not None
    assert str(result[0]) == str(session_id)  # id (convert UUID to string)
    assert str(result[1]) == tenant_id        # tenant_id (convert UUID to string)
    assert str(result[2]) == user_id          # user_id (convert UUID to string)
    assert result[3] == session_key           # session_key
    assert result[4] == "active"              # status

@pytest.mark.integration
def test_start_session_db_persists_session_without_user_id(postgres_container):
    """Test that start_session_db correctly persists a session to the database without user_id."""
    # Test with tenant_id only (as UUID4 string)
    tenant_id = TEST_TENANT_ID

    session_data = start_session_db(x_tenant_id=tenant_id, x_user_id=None)

    # Verify the session was created
    session_id = session_data["session_id"]
    session_key = session_data["session_key"]

    # Query the database to verify the session exists
    result = fetch_one(
        "SELECT id, tenant_id, user_id, session_key, status FROM sessions WHERE id = %s",
        (session_id,)
    )

    assert result is not None
    assert str(result[0]) == str(session_id)  # id (convert UUID to string)
    assert str(result[1]) == tenant_id        # tenant_id (convert UUID to string)
    assert result[2] is None                  # user_id (should be NULL)
    assert result[3] == session_key           # session_key
    assert result[4] == "active"              # status
