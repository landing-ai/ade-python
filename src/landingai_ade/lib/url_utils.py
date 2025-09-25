"""Utility functions for handling URL and file path conversions."""

from typing import Tuple, Union, Optional, cast
from pathlib import Path
from urllib.parse import urlparse

from .._types import Omit, FileTypes, omit


def convert_url_to_file_if_local(
    file: Union[Optional[FileTypes], Omit],
    file_url: Union[Optional[str], Omit],
) -> Tuple[Union[Optional[FileTypes], Omit], Union[Optional[str], Omit]]:
    """
    Convert a URL parameter to a file parameter if it's a local file path.

    If the file_url is a local file path that exists, it will be converted to a Path object
    and returned as the file parameter, with the file_url parameter set to omit.

    If the file_url is a remote URL (http/https) or doesn't exist as a local file,
    it will be returned unchanged.

    Args:
        file: The existing file parameter
        file_url: The URL parameter that might be a local file path

    Returns:
        A tuple of (file, file_url) where one will be set and the other omit
    """
    # If file_url is omit or None, return unchanged
    if file_url is omit or file_url is None:
        return file, file_url

    # At this point, file_url is guaranteed to be a string, use cast for type narrowing
    url_str = cast(str, file_url)

    # Check if it's a remote URL (http/https)
    parsed = urlparse(url_str)
    if parsed.scheme in ("http", "https", "ftp", "ftps"):
        # It's a remote URL, keep it as is
        return file, file_url

    # Check if it's a local file path
    path = Path(url_str)
    if path.exists() and path.is_file():
        # It's a local file, convert to file parameter
        return path, omit

    # Path doesn't exist or is not a file, treat as URL
    # (could be a URL with a different scheme or a typo)
    return file, file_url
