"""Utility functions for handling URL and file path conversions."""

from typing import Tuple, Union, Optional
from pathlib import Path
from urllib.parse import urlparse

from .._types import Omit, FileTypes, omit


def convert_url_to_file_if_local(
    url: Union[Optional[str], Omit],
    file_param: Union[Optional[FileTypes], Omit],
) -> Tuple[Union[Optional[FileTypes], Omit], Union[Optional[str], Omit]]:
    """
    Convert a URL parameter to a file parameter if it's a local file path.

    If the URL is a local file path that exists, it will be converted to a Path object
    and returned as the file parameter, with the URL parameter set to omit.

    If the URL is a remote URL (http/https) or doesn't exist as a local file,
    it will be returned unchanged.

    Args:
        url: The URL parameter that might be a local file path
        file_param: The existing file parameter

    Returns:
        A tuple of (file_parameter, url_parameter) where one will be set and the other omit
    """
    # If url is omit or None, return unchanged
    if url is omit or url is None:
        return file_param, url

    # Check if it's a remote URL (http/https)
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https", "ftp", "ftps"):
        # It's a remote URL, keep it as is
        return file_param, url

    # Check if it's a local file path
    path = Path(url)
    if path.exists() and path.is_file():
        # It's a local file, convert to file parameter
        return path, omit

    # Path doesn't exist or is not a file, treat as URL
    # (could be a URL with a different scheme or a typo)
    return file_param, url
