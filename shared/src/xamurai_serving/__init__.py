"""Small serving primitives shared by realtime Xamurai providers."""

from .admission import SessionSlots, max_sessions_from_env, serving_instance_id

__all__ = ["SessionSlots", "max_sessions_from_env", "serving_instance_id"]
