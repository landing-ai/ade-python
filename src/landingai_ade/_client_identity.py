"""Client identity attached to every API request (see landing-ai/ade-python#131).

The AIDE gateway relays these two headers into the recorded ``inference_history``
row so SDK traffic is distinguishable from raw API calls:

* ``X-Source: sdk`` — names the row's ``source`` column (the coarse API/CLI/SDK
  split; ``sdk`` is already live in the platform's ``InferenceHistorySource`` enum).
* a structured ``User-Agent`` — parsed platform-side into os/arch/runtime
  dimensions.

The User-Agent grammar is shared with ade-cli (its ``docs/user-agent.md``) and
parsed by vision-agent-ui's ``parseUserAgent.ts``::

    ade-python/<version> (<os> <arch>) python/<major.minor> httpx/<version>

The parser owns the grammar, not the vocabulary: the leading token identifies
the product, the parenthesized comment fills os/arch *only* when it is exactly
two space-separated words with no ``;``/``,``, and every remaining ``key/value``
token is captured generically — so appending further tokens later needs no
platform change.

This is the single construction seam for the identity string. Nothing here may
raise: identity must never fail a request, so every lookup degrades to a
placeholder rather than propagating an error.
"""

from __future__ import annotations

import re
import sys
import platform

import httpx

# ``X-Source`` value: the coarse API/CLI/SDK split.
SOURCE = "sdk"

# Leading product token. ``ade-python`` (not the distribution name
# ``landingai-ade``) keeps the two SDKs separable — the TypeScript SDK uses
# ``ade-typescript``.
PRODUCT = "ade-python"

_PLACEHOLDER = "unknown"

# Keep only visible-ASCII token characters; collapse everything else into a
# single ``-``. This covers what would break the parser's ``(<os> <arch>)`` shape
# (whitespace, ``;``/``,``, parentheses) AND non-ASCII values (which httpx cannot
# latin-1 encode — an exotic ``platform.machine()`` like ``arm<emoji>`` would
# otherwise raise while building the header and fail the request).
_UNSAFE_COMMENT_CHARS = re.compile(r"[^A-Za-z0-9._:+-]+")


def _product_version() -> str:
    # The installed distribution version is the true runtime version; fall back
    # to the packaged ``__version__``, then to a placeholder — never raise.
    try:
        from importlib.metadata import version

        return version("landingai-ade")
    except Exception:
        pass
    try:
        from ._version import __version__

        return __version__
    except Exception:
        return _PLACEHOLDER


def _clean_comment_word(value: str) -> str:
    return _UNSAFE_COMMENT_CHARS.sub("-", value).strip("-") or _PLACEHOLDER


def _platform_comment() -> str:
    # `(<os> <arch>)`: exactly two space-separated words, no `;`/`,`, so the
    # platform parser fills the os/arch dimensions. Uses the raw
    # platform.system()/machine() values (e.g. "Darwin", "arm64") to match
    # ade-cli, so a given machine reports identical os/arch across the CLI and
    # this SDK.
    try:
        os_name = _clean_comment_word(platform.system())
    except Exception:
        os_name = _PLACEHOLDER
    try:
        arch = _clean_comment_word(platform.machine())
    except Exception:
        arch = _PLACEHOLDER
    return f"({os_name} {arch})"


def _python_token() -> str:
    # sys.version_info is always present and its fields are ints, so this can't
    # raise — no guard needed (unlike the platform/httpx lookups below).
    return f"python/{sys.version_info.major}.{sys.version_info.minor}"


def _httpx_token() -> str:
    try:
        return f"httpx/{httpx.__version__}"
    except Exception:
        return f"httpx/{_PLACEHOLDER}"


def build_user_agent(version: str | None = None) -> str:
    """Build the structured ``User-Agent``. Never raises.

    ``version`` lets the caller pass the SDK version it already holds — the
    client threads its ``self._version`` here (mirroring ``platform_headers``)
    so the process-constant identity isn't rebuilt from an
    ``importlib.metadata`` distribution scan on every request. When omitted, the
    installed distribution version is resolved as a fallback.
    """
    return " ".join(
        [
            f"{PRODUCT}/{version or _product_version()}",
            _platform_comment(),
            _python_token(),
            _httpx_token(),
        ]
    )
