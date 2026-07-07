from __future__ import annotations

import os

import pytest

from landingai_ade import LandingAIADE

pytestmark = pytest.mark.contract

STAGING_KEY = os.environ.get("LANDINGAI_ADE_STAGING_APIKEY")
# V1 staging host. (Once Problem 2 adds the 4-environment map, this can become
# environment="staging"; until then we target staging via base_url.)
STAGING_V1_BASE_URL = "https://api.va.staging.landing.ai"


@pytest.fixture()
def staging_client() -> LandingAIADE:
    if not STAGING_KEY:
        pytest.skip("LANDINGAI_ADE_STAGING_APIKEY not set")
    return LandingAIADE(apikey=STAGING_KEY, base_url=STAGING_V1_BASE_URL)


def test_parse_jobs_list_reachable(staging_client: LandingAIADE) -> None:
    """The implemented V1 surface answers on staging (auth + routing sanity)."""
    result = staging_client.parse_jobs.list(page=0, page_size=1)
    assert hasattr(result, "jobs")
