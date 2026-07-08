# src/landingai_ade/resources/v2/v2.py
from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import V2ResourceMixin
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from .files import FilesResource, AsyncFilesResource

__all__ = ["V2Resource", "AsyncV2Resource"]


class V2Resource(SyncAPIResource, V2ResourceMixin):
    """Container for the V2 (ADE) surface: ``client.v2.<resource>``.

    Sub-resources (``parse``, ``parse_jobs``, ``extract``, ``extract_jobs``) are
    attached as cached properties by later tasks (8-11), each doing its own lazy
    import inside the property body -- mirroring ``LandingAIADE.parse_jobs`` -- so that
    this module keeps importing standalone regardless of which sub-resources exist yet.
    """

    @cached_property
    def files(self) -> FilesResource:
        from .files import FilesResource

        return FilesResource(self._client)


class AsyncV2Resource(AsyncAPIResource, V2ResourceMixin):
    """Async mirror of :class:`V2Resource`."""

    @cached_property
    def files(self) -> AsyncFilesResource:
        from .files import AsyncFilesResource

        return AsyncFilesResource(self._client)
