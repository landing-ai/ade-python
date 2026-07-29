"""Client identity headers — User-Agent + X-Source (landing-ai/ade-python#131)."""

from __future__ import annotations

import re

import httpx
import pytest

from landingai_ade import LandingAIADE, AsyncLandingAIADE
from landingai_ade._models import FinalRequestOptions
from landingai_ade._client_identity import SOURCE, build_user_agent

base_url = "http://127.0.0.1:4010"
apikey = "My Apikey"

# Endpoints spanning the operations called out in the issue: parse, extract,
# and job submit (POST .../jobs) + poll (GET .../jobs/{id}).
ENDPOINTS = [
    ("post", "/v1/ade/parse"),
    ("post", "/v1/ade/extract"),
    ("post", "/v1/ade/parse/jobs"),  # submit
    ("get", "/v1/ade/parse/jobs/job_123"),  # poll
    ("post", "/v1/ade/extract/jobs"),  # submit
    ("get", "/v1/ade/extract/jobs/job_123"),  # poll
]


def _parse_user_agent(raw: str) -> dict[str, str]:
    """Faithful port of vision-agent-ui's ``parseUserAgent.ts``.

    Kept in lockstep with the platform parser so these tests fail if our
    User-Agent ever drifts from the shape the platform actually reads.
    """
    analysis: dict[str, str] = {"raw": raw}

    comment = re.search(r"\(([^)]*)\)", raw)
    if comment:
        parts = comment.group(1).strip().split()
        if len(parts) == 2 and not re.search(r"[;,]", comment.group(1)):
            analysis["os"] = parts[0]
            analysis["arch"] = parts[1]
        tokens = raw.replace(comment.group(0), " ").strip().split()
    else:
        tokens = raw.strip().split()

    product, *rest = tokens
    product_match = re.match(r"^([^/]+)/(.+)$", product)
    if product_match:
        analysis["product"] = product_match.group(1)
        analysis["productVersion"] = product_match.group(2)

    reserved = {"raw", "product", "productVersion", "os", "arch"}
    for token in rest:
        slash = token.find("/")
        if slash <= 0 or slash == len(token) - 1:
            continue
        key = token[:slash]
        if key in reserved or key in analysis:
            continue
        analysis[key] = token[slash + 1 :]

    return analysis


class TestUserAgentGrammar:
    def test_parses_per_platform_contract(self) -> None:
        parsed = _parse_user_agent(build_user_agent())
        assert parsed["product"] == "ade-python"
        assert parsed.get("productVersion")
        # The `(<os> <arch>)` comment fills both dimensions.
        assert parsed.get("os") and parsed.get("arch")
        # Runtime + HTTP-lib key/value tokens.
        assert parsed.get("python")
        assert parsed.get("httpx")

    def test_platform_comment_shape(self) -> None:
        # Exactly two space-separated words, no `;`/`,` — the only shape the
        # platform parser reads into os/arch.
        comment = re.search(r"\(([^)]*)\)", build_user_agent())
        assert comment is not None
        inner = comment.group(1)
        assert ";" not in inner and "," not in inner
        assert len(inner.split()) == 2

    def test_python_token_is_major_minor(self) -> None:
        assert re.search(r"\bpython/\d+\.\d+\b", build_user_agent())


class TestSource:
    def test_source_constant(self) -> None:
        assert SOURCE == "sdk"


class TestNeverFailsARequest:
    @pytest.mark.parametrize("target", ["system", "machine"])
    def test_platform_detection_failure_degrades(self, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
        import platform

        def boom(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("boom")

        monkeypatch.setattr(platform, target, boom)
        ua = build_user_agent()
        assert ua.startswith("ade-python/")
        assert "unknown" in ua[ua.index("(") : ua.index(")") + 1]

    def test_non_ascii_platform_value_stays_ascii(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An exotic platform.machine() must not smuggle non-latin-1 bytes into the
        # header (httpx would raise while encoding and fail the request).
        import platform

        monkeypatch.setattr(platform, "machine", lambda: "arm\U0001f525")  # arm🔥
        ua = build_user_agent()
        ua.encode("ascii")  # must not raise
        comment = ua[ua.index("(") + 1 : ua.index(")")]
        assert len(comment.split()) == 2
        assert "arm" in comment

    def test_missing_package_metadata_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.metadata as metadata

        def boom(_name: str) -> str:
            raise metadata.PackageNotFoundError(_name)

        monkeypatch.setattr(metadata, "version", boom)
        # Falls back to the packaged __version__ rather than raising.
        assert build_user_agent().startswith("ade-python/")


def _request_headers(client: LandingAIADE | AsyncLandingAIADE, method: str, url: str) -> httpx.Headers:
    return client._build_request(FinalRequestOptions(method=method, url=url)).headers


class TestEveryRequestCarriesIdentity:
    @pytest.mark.parametrize("method,url", ENDPOINTS)
    def test_sync(self, client: LandingAIADE, method: str, url: str) -> None:
        headers = _request_headers(client, method, url)
        assert headers["x-source"] == "sdk"
        assert headers["user-agent"].startswith("ade-python/")

    @pytest.mark.parametrize("method,url", ENDPOINTS)
    async def test_async(self, async_client: AsyncLandingAIADE, method: str, url: str) -> None:
        headers = _request_headers(async_client, method, url)
        assert headers["x-source"] == "sdk"
        assert headers["user-agent"].startswith("ade-python/")


class TestCallerOverride:
    def test_default_is_the_sdk_identity(self) -> None:
        with LandingAIADE(base_url=base_url, apikey=apikey) as client:
            assert client.default_headers["X-Source"] == "sdk"
            assert str(client.default_headers["User-Agent"]).startswith("ade-python/")

    def test_caller_supplied_header_overrides(self) -> None:
        # A caller who explicitly supplies the header wins (documented override);
        # the identity is never *silently* dropped for callers who don't.
        with LandingAIADE(base_url=base_url, apikey=apikey, default_headers={"X-Source": "myapp"}) as client:
            assert client.default_headers["X-Source"] == "myapp"
