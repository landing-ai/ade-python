from pathlib import Path

import anyio
import pytest
from dirty_equals import IsDict, IsList, IsBytes, IsTuple

from landingai_ade._files import to_httpx_files, async_to_httpx_files

readme_path = Path(__file__).parent.parent.joinpath("README.md")


def test_pathlib_includes_file_name() -> None:
    result = to_httpx_files({"file": readme_path})
    print(result)
    assert result == IsDict({"file": IsTuple("README.md", IsBytes())})


def test_tuple_input() -> None:
    result = to_httpx_files([("file", readme_path)])
    print(result)
    assert result == IsList(IsTuple("file", IsTuple("README.md", IsBytes())))


@pytest.mark.asyncio
async def test_async_pathlib_includes_file_name() -> None:
    result = await async_to_httpx_files({"file": readme_path})
    print(result)
    assert result == IsDict({"file": IsTuple("README.md", IsBytes())})


@pytest.mark.asyncio
async def test_async_supports_anyio_path() -> None:
    result = await async_to_httpx_files({"file": anyio.Path(readme_path)})
    print(result)
    assert result == IsDict({"file": IsTuple("README.md", IsBytes())})


@pytest.mark.asyncio
async def test_async_tuple_input() -> None:
    result = await async_to_httpx_files([("file", readme_path)])
    print(result)
    assert result == IsList(IsTuple("file", IsTuple("README.md", IsBytes())))


def test_string_allowed() -> None:
    # Strings are now allowed and should be converted to bytes
    result = to_httpx_files(
        {
            "file": "foo",
        }
    )
    assert result == {"file": b"foo"}
