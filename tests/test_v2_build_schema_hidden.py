"""Regression guard: the V2 build-schema surface is intentionally hidden.

The public `client.v2.build_schema(...)` method and `client.v2.build_schema_jobs`
resource were soft-hidden; the implementation and types are retained internally.
These assertions ensure a future spec-sync wiring change cannot silently
re-expose the surface. V1 `client.extract_build_schema` is intentionally kept.
"""

from __future__ import annotations

from landingai_ade import LandingAIADE, AsyncLandingAIADE

base_url = "http://127.0.0.1:4010"
apikey = "My Apikey"


def test_sync_v2_build_schema_surface_absent() -> None:
    with LandingAIADE(base_url=base_url, apikey=apikey) as client:
        assert not hasattr(client.v2, "build_schema")
        assert not hasattr(client.v2, "build_schema_jobs")
        # V1 build-schema is intentionally retained.
        assert hasattr(client, "extract_build_schema")


async def test_async_v2_build_schema_surface_absent() -> None:
    async with AsyncLandingAIADE(base_url=base_url, apikey=apikey) as client:
        assert not hasattr(client.v2, "build_schema")
        assert not hasattr(client.v2, "build_schema_jobs")
        assert hasattr(client, "extract_build_schema")
