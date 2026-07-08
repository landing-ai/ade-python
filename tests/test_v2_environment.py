from __future__ import annotations

from typing_extensions import Literal

import pytest

from landingai_ade import LandingAIADE

APIKEY = "My Apikey"


def test_default_production_pair() -> None:
    c = LandingAIADE(apikey=APIKEY)
    assert str(c.base_url).rstrip("/") == "https://api.va.landing.ai"
    assert c._v2_base_url == "https://aide.landing.ai"


@pytest.mark.parametrize(
    "env,v1,v2",
    [
        ("production", "https://api.va.landing.ai", "https://aide.landing.ai"),
        ("eu", "https://api.va.eu-west-1.landing.ai", "https://aide.eu-west-1.landing.ai"),
        ("staging", "https://api.va.staging.landing.ai", "https://aide.staging.landing.ai"),
        ("dev", "https://api.va.dev.landing.ai", "https://aide.dev.landing.ai"),
    ],
)
def test_environment_pairs(env: Literal["production", "eu", "staging", "dev"], v1: str, v2: str) -> None:
    c = LandingAIADE(apikey=APIKEY, environment=env)
    assert str(c.base_url).rstrip("/") == v1
    assert c._v2_base_url == v2


def test_environment_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANDINGAI_ADE_ENVIRONMENT", "staging")
    c = LandingAIADE(apikey=APIKEY)
    assert c._v2_base_url == "https://aide.staging.landing.ai"


def test_explicit_v2_base_url_wins() -> None:
    c = LandingAIADE(apikey=APIKEY, v2_base_url="https://mock.local")
    assert c._v2_base_url == "https://mock.local"


def test_v2_follows_base_url_when_only_base_url_set() -> None:
    c = LandingAIADE(apikey=APIKEY, base_url="http://127.0.0.1:4010")
    assert str(c.base_url).rstrip("/") == "http://127.0.0.1:4010"
    assert c._v2_base_url == "http://127.0.0.1:4010"


def test_v2_base_url_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANDINGAI_ADE_V2_BASE_URL", "https://v2.mock.local")
    c = LandingAIADE(apikey=APIKEY)
    assert c._v2_base_url == "https://v2.mock.local"
