from __future__ import annotations

from typing import Any
from typing_extensions import override

from ._proxy import LazyProxy


class ResourcesProxy(LazyProxy[Any]):
    """A proxy for the `landingai_ade.resources` module.

    This is used so that we can lazily import `landingai_ade.resources` only when
    needed *and* so that users can just import `landingai_ade` and reference `landingai_ade.resources`
    """

    @override
    def __load__(self) -> Any:
        import importlib

        mod = importlib.import_module("landingai_ade.resources")
        return mod


resources = ResourcesProxy().__as_proxied__()
