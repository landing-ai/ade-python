# src/landingai_ade/resources/v2/v2.py
from __future__ import annotations

from ._base import V2ResourceMixin
from ..._resource import SyncAPIResource, AsyncAPIResource

__all__ = ["V2Resource", "AsyncV2Resource"]


class V2Resource(SyncAPIResource, V2ResourceMixin):
    """Container for the V2 (ADE) surface: ``client.v2.<resource>``.

    Sub-resources (``files``, ``parse``, ``parse_jobs``, ``extract``, ``extract_jobs``)
    are attached as cached properties by later tasks (7-11), each doing its own lazy
    import inside the property body -- mirroring ``LandingAIADE.parse_jobs`` -- so that
    this module keeps importing standalone regardless of which sub-resources exist yet.
    """


class AsyncV2Resource(AsyncAPIResource, V2ResourceMixin):
    """Async mirror of :class:`V2Resource`."""
