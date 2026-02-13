# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
import importlib.metadata
from typing import TYPE_CHECKING, Any, Dict, Union, Mapping, Iterable, Optional, cast
from pathlib import Path
from urllib.parse import urlparse
from typing_extensions import Self, Literal, override

import httpx

from . import _exceptions
from ._qs import Querystring
from .types import client_parse_params, client_split_params, client_extract_params
from ._types import (
    Body,
    Omit,
    Query,
    Headers,
    Timeout,
    NotGiven,
    FileTypes,
    Transport,
    ProxiesTypes,
    RequestOptions,
    omit,
    not_given,
)
from ._utils import (
    is_given,
    extract_files,
    maybe_transform,
    deepcopy_minimal,
    get_async_library,
    async_maybe_transform,
)
from ._compat import cached_property
from ._version import __version__
from ._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ._streaming import Stream as Stream, AsyncStream as AsyncStream
from ._exceptions import APIStatusError, LandingAiadeError
from ._base_client import (
    DEFAULT_MAX_RETRIES,
    SyncAPIClient,
    AsyncAPIClient,
    make_request_options,
)
from .lib.url_utils import convert_url_to_file_if_local
from .types.parse_response import ParseResponse
from .types.split_response import SplitResponse
from .types.extract_response import ExtractResponse

if TYPE_CHECKING:
    from .resources import parse_jobs
    from .resources.parse_jobs import ParseJobsResource, AsyncParseJobsResource
_LIB_VERSION = importlib.metadata.version("landingai-ade")

__all__ = [
    "ENVIRONMENTS",
    "Timeout",
    "Transport",
    "ProxiesTypes",
    "RequestOptions",
    "LandingAIADE",
    "AsyncLandingAIADE",
    "Client",
    "AsyncClient",
]

ENVIRONMENTS: Dict[str, str] = {
    "production": "https://api.va.landing.ai",
    "eu": "https://api.va.eu-west-1.landing.ai",
}


def _get_input_filename(
    file_input: Union[FileTypes, Omit, None], url_input: Union[str, Omit, None]
) -> str:
    """Extract base filename (without extension) from file or URL input."""
    if file_input is not None and not isinstance(file_input, Omit):
        if isinstance(file_input, (Path, str)):
            return Path(file_input).stem
        elif isinstance(file_input, tuple) and len(file_input) > 0:
            # Tuple format: (filename, content, mime_type)
            return Path(str(file_input[0])).stem
        elif isinstance(file_input, (bytes, bytearray, memoryview)):
            # Raw bytes input does not carry a filename
            pass
        elif hasattr(file_input, "name"):
            # IO objects may have a name attribute
            name = getattr(file_input, "name", None)
            if name and isinstance(name, str):
                return Path(name).stem
    if url_input is not None and not isinstance(url_input, Omit):
        path = urlparse(url_input).path
        stem = Path(path).stem if path else ""
        return stem if stem else "url_input"
    return "output"


def _save_response(
    save_to: Union[str, Path],
    filename: str,
    method_name: str,
    result: Any,
) -> None:
    """Save API response to a JSON file in the specified folder."""
    try:
        folder = Path(save_to)
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / f"{filename}_{method_name}_output.json"
        output_path.write_text(result.to_json())
    except OSError as exc:
        raise LandingAiadeError(f"Failed to save {method_name} response to {save_to}: {exc}") from exc


class LandingAIADE(SyncAPIClient):
    # client options
    apikey: str

    _environment: Literal["production", "eu"] | NotGiven

    def __init__(
        self,
        *,
        apikey: str | None = None,
        environment: Literal["production", "eu"] | NotGiven = not_given,
        base_url: str | httpx.URL | None | NotGiven = not_given,
        timeout: float | Timeout | None | NotGiven = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        # Configure a custom httpx client.
        # We provide a `DefaultHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
        # See the [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
        http_client: httpx.Client | None = None,
        # Enable or disable schema validation for data returned by the API.
        # When enabled an error APIResponseValidationError is raised
        # if the API responds with invalid data for the expected schema.
        #
        # This parameter may be removed or changed in the future.
        # If you rely on this feature, please open a GitHub issue
        # outlining your use-case to help us decide if it should be
        # part of our public interface in the future.
        _strict_response_validation: bool = False,
    ) -> None:
        """Construct a new synchronous LandingAIADE client instance.

        This automatically infers the `apikey` argument from the `VISION_AGENT_API_KEY` environment variable if it is not provided.
        """
        if apikey is None:
            apikey = os.environ.get("VISION_AGENT_API_KEY")
        if apikey is None:
            raise LandingAiadeError(
                "The apikey client option must be set either by passing apikey to the client or by setting the VISION_AGENT_API_KEY environment variable"
            )
        self.apikey = apikey

        self._environment = environment

        base_url_env = os.environ.get("LANDINGAI_ADE_BASE_URL")
        if is_given(base_url) and base_url is not None:
            # cast required because mypy doesn't understand the type narrowing
            base_url = cast("str | httpx.URL", base_url)  # pyright: ignore[reportUnnecessaryCast]
        elif is_given(environment):
            if base_url_env and base_url is not None:
                raise ValueError(
                    "Ambiguous URL; The `LANDINGAI_ADE_BASE_URL` env var and the `environment` argument are given. If you want to use the environment, you must pass base_url=None",
                )

            try:
                base_url = ENVIRONMENTS[environment]
            except KeyError as exc:
                raise ValueError(f"Unknown environment: {environment}") from exc
        elif base_url_env is not None:
            base_url = base_url_env
        else:
            self._environment = environment = "production"

            try:
                base_url = ENVIRONMENTS[environment]
            except KeyError as exc:
                raise ValueError(f"Unknown environment: {environment}") from exc

        super().__init__(
            version=__version__,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

    @cached_property
    def parse_jobs(self) -> ParseJobsResource:
        from .resources.parse_jobs import ParseJobsResource

        return ParseJobsResource(self)

    @cached_property
    def with_raw_response(self) -> LandingAIADEWithRawResponse:
        return LandingAIADEWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> LandingAIADEWithStreamedResponse:
        return LandingAIADEWithStreamedResponse(self)

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="comma")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        apikey = self.apikey
        return {"Authorization": f"Bearer {apikey}"}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": "false",
            **self._custom_headers,
        }

    def copy(
        self,
        *,
        apikey: str | None = None,
        environment: Literal["production", "eu"] | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        http_client: httpx.Client | None = None,
        max_retries: int | NotGiven = not_given,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(
            apikey=apikey or self.apikey,
            base_url=base_url or self.base_url,
            environment=environment or self._environment,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    def extract(
        self,
        *,
        schema: str,
        markdown: Union[FileTypes, str, None] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExtractResponse:
        """
        Extract structured data from Markdown using a JSON schema.

        This endpoint processes Markdown content and extracts structured data according
        to the provided JSON schema.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/extract`.

        Args:
          schema: JSON schema for field extraction. This schema determines what key-values pairs
              are extracted from the Markdown. The schema must be a valid JSON object and will
              be validated before processing the document.

          markdown: The Markdown file or Markdown content to extract data from.

          markdown_url: The URL to the Markdown file to extract data from.

          model: The version of the model to use for extraction. Use `extract-latest` to use the
              latest version.

          save_to: Optional output folder path. If provided, the response will be saved as
              JSON to this folder with the filename format: {input_file}_extract_output.json.
              The folder will be created if it doesn't exist.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        # Store original inputs for filename extraction before conversion
        original_markdown = markdown
        original_markdown_url = markdown_url
        # Convert local file paths to file parameters
        markdown, markdown_url = convert_url_to_file_if_local(markdown, markdown_url)

        body = deepcopy_minimal(
            {
                "schema": schema,
                "markdown": markdown,
                "markdown_url": markdown_url,
                "model": model,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {
            "Content-Type": "multipart/form-data",
            "runtime_tag": f"ade-python-v{_LIB_VERSION}",
            **(extra_headers or {}),
        }
        result = self.post(
            "/v1/ade/extract",
            body=maybe_transform(body, client_extract_params.ClientExtractParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=ExtractResponse,
        )
        if save_to:
            filename = _get_input_filename(original_markdown, original_markdown_url)
            _save_response(save_to, filename, "extract", result)
        return result

    def parse(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        split: Optional[Literal["page"]] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ParseResponse:
        """
        Parse a document or spreadsheet.

        This endpoint parses documents (PDF, images) and spreadsheets (XLSX, CSV) into
        structured Markdown, chunks, and metadata.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/parse`.

        Args:
          document: A file to be parsed. The file can be a PDF or an image. See the list of
              supported file types here: https://docs.landing.ai/ade/ade-file-types. Either
              this parameter or the `document_url` parameter must be provided.

          document_url: The URL to the file to be parsed. The file can be a PDF or an image. See the
              list of supported file types here: https://docs.landing.ai/ade/ade-file-types.
              Either this parameter or the `document` parameter must be provided.

          model: The version of the model to use for parsing.

          split: If you want to split documents into smaller sections, include the split
              parameter. Set the parameter to page to split documents at the page level. The
              splits object in the API output will contain a set of data for each page.

          save_to: Optional output folder path. If provided, the response will be saved as
              JSON to this folder with the filename format: {input_file}_parse_output.json.
              The folder will be created if it doesn't exist.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        # Store original inputs for filename extraction before conversion
        original_document = document
        original_document_url = document_url
        # Convert local file paths to file parameters
        document, document_url = convert_url_to_file_if_local(document, document_url)

        body = deepcopy_minimal(
            {
                "document": document,
                "document_url": document_url,
                "model": model,
                "split": split,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {
            "Content-Type": "multipart/form-data",
            "runtime_tag": f"ade-python-v{_LIB_VERSION}",
            **(extra_headers or {}),
        }
        result = self.post(
            "/v1/ade/parse",
            body=maybe_transform(body, client_parse_params.ClientParseParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=ParseResponse,
        )
        if save_to:
            filename = _get_input_filename(original_document, original_document_url)
            _save_response(save_to, filename, "parse", result)
        return result

    def split(
        self,
        *,
        split_class: Iterable[client_split_params.SplitClass],
        markdown: Union[FileTypes, str, None] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SplitResponse:
        """
        Split classification for documents.

        This endpoint classifies document sections based on markdown content and split
        options.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/split`.

        Args:
          split_class: List of split classification options/configuration. Can be provided as JSON
              string in form data.

          markdown: The Markdown file or Markdown content to split.

          markdown_url: The URL to the Markdown file to split.

          model: Model version to use for split classification. Defaults to the latest version.

          save_to: Optional output folder path. If provided, the response will be saved as
              JSON to this folder with the filename format: {input_file}_split_output.json.
              The folder will be created if it doesn't exist.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        # Store original inputs for filename extraction
        original_markdown = markdown
        original_markdown_url = markdown_url
        body = deepcopy_minimal(
            {
                "split_class": split_class,
                "markdown": markdown,
                "markdown_url": markdown_url,
                "model": model,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        result = self.post(
            "/v1/ade/split",
            body=maybe_transform(body, client_split_params.ClientSplitParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=SplitResponse,
        )
        if save_to:
            filename = _get_input_filename(original_markdown, original_markdown_url)
            _save_response(save_to, filename, "split", result)
        return result

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=body)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=body)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=body)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=body)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=body)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=body)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=body)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=body)
        return APIStatusError(err_msg, response=response, body=body)


class AsyncLandingAIADE(AsyncAPIClient):
    # client options
    apikey: str

    _environment: Literal["production", "eu"] | NotGiven

    def __init__(
        self,
        *,
        apikey: str | None = None,
        environment: Literal["production", "eu"] | NotGiven = not_given,
        base_url: str | httpx.URL | None | NotGiven = not_given,
        timeout: float | Timeout | None | NotGiven = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        # Configure a custom httpx client.
        # We provide a `DefaultAsyncHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
        # See the [httpx documentation](https://www.python-httpx.org/api/#asyncclient) for more details.
        http_client: httpx.AsyncClient | None = None,
        # Enable or disable schema validation for data returned by the API.
        # When enabled an error APIResponseValidationError is raised
        # if the API responds with invalid data for the expected schema.
        #
        # This parameter may be removed or changed in the future.
        # If you rely on this feature, please open a GitHub issue
        # outlining your use-case to help us decide if it should be
        # part of our public interface in the future.
        _strict_response_validation: bool = False,
    ) -> None:
        """Construct a new async AsyncLandingAIADE client instance.

        This automatically infers the `apikey` argument from the `VISION_AGENT_API_KEY` environment variable if it is not provided.
        """
        if apikey is None:
            apikey = os.environ.get("VISION_AGENT_API_KEY")
        if apikey is None:
            raise LandingAiadeError(
                "The apikey client option must be set either by passing apikey to the client or by setting the VISION_AGENT_API_KEY environment variable"
            )
        self.apikey = apikey

        self._environment = environment

        base_url_env = os.environ.get("LANDINGAI_ADE_BASE_URL")
        if is_given(base_url) and base_url is not None:
            # cast required because mypy doesn't understand the type narrowing
            base_url = cast("str | httpx.URL", base_url)  # pyright: ignore[reportUnnecessaryCast]
        elif is_given(environment):
            if base_url_env and base_url is not None:
                raise ValueError(
                    "Ambiguous URL; The `LANDINGAI_ADE_BASE_URL` env var and the `environment` argument are given. If you want to use the environment, you must pass base_url=None",
                )

            try:
                base_url = ENVIRONMENTS[environment]
            except KeyError as exc:
                raise ValueError(f"Unknown environment: {environment}") from exc
        elif base_url_env is not None:
            base_url = base_url_env
        else:
            self._environment = environment = "production"

            try:
                base_url = ENVIRONMENTS[environment]
            except KeyError as exc:
                raise ValueError(f"Unknown environment: {environment}") from exc

        super().__init__(
            version=__version__,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

    @cached_property
    def parse_jobs(self) -> AsyncParseJobsResource:
        from .resources.parse_jobs import AsyncParseJobsResource

        return AsyncParseJobsResource(self)

    @cached_property
    def with_raw_response(self) -> AsyncLandingAIADEWithRawResponse:
        return AsyncLandingAIADEWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncLandingAIADEWithStreamedResponse:
        return AsyncLandingAIADEWithStreamedResponse(self)

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="comma")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        apikey = self.apikey
        return {"Authorization": f"Bearer {apikey}"}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": f"async:{get_async_library()}",
            **self._custom_headers,
        }

    def copy(
        self,
        *,
        apikey: str | None = None,
        environment: Literal["production", "eu"] | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | NotGiven = not_given,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(
            apikey=apikey or self.apikey,
            base_url=base_url or self.base_url,
            environment=environment or self._environment,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    async def extract(
        self,
        *,
        schema: str,
        markdown: Union[FileTypes, str, None] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExtractResponse:
        """
        Extract structured data from Markdown using a JSON schema.

        This endpoint processes Markdown content and extracts structured data according
        to the provided JSON schema.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/extract`.

        Args:
          schema: JSON schema for field extraction. This schema determines what key-values pairs
              are extracted from the Markdown. The schema must be a valid JSON object and will
              be validated before processing the document.

          markdown: The Markdown file or Markdown content to extract data from.

          markdown_url: The URL to the Markdown file to extract data from.

          model: The version of the model to use for extraction. Use `extract-latest` to use the
              latest version.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        # Convert local file paths to file parameters
        markdown, markdown_url = convert_url_to_file_if_local(markdown, markdown_url)

        body = deepcopy_minimal(
            {
                "schema": schema,
                "markdown": markdown,
                "markdown_url": markdown_url,
                "model": model,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {
            "Content-Type": "multipart/form-data",
            "runtime_tag": f"ade-python-v{_LIB_VERSION}",
            **(extra_headers or {}),
        }
        return await self.post(
            "/v1/ade/extract",
            body=await async_maybe_transform(body, client_extract_params.ClientExtractParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=ExtractResponse,
        )

    async def parse(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        split: Optional[Literal["page"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ParseResponse:
        """
        Parse a document or spreadsheet.

        This endpoint parses documents (PDF, images) and spreadsheets (XLSX, CSV) into
        structured Markdown, chunks, and metadata.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/parse`.

        Args:
          document: A file to be parsed. The file can be a PDF or an image. See the list of
              supported file types here: https://docs.landing.ai/ade/ade-file-types. Either
              this parameter or the `document_url` parameter must be provided.

          document_url: The URL to the file to be parsed. The file can be a PDF or an image. See the
              list of supported file types here: https://docs.landing.ai/ade/ade-file-types.
              Either this parameter or the `document` parameter must be provided.

          model: The version of the model to use for parsing.

          split: If you want to split documents into smaller sections, include the split
              parameter. Set the parameter to page to split documents at the page level. The
              splits object in the API output will contain a set of data for each page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        # Convert local file paths to file parameters
        document, document_url = convert_url_to_file_if_local(document, document_url)

        body = deepcopy_minimal(
            {
                "document": document,
                "document_url": document_url,
                "model": model,
                "split": split,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {
            "Content-Type": "multipart/form-data",
            "runtime_tag": f"ade-python-v{_LIB_VERSION}",
            **(extra_headers or {}),
        }
        return await self.post(
            "/v1/ade/parse",
            body=await async_maybe_transform(body, client_parse_params.ClientParseParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=ParseResponse,
        )

    async def split(
        self,
        *,
        split_class: Iterable[client_split_params.SplitClass],
        markdown: Union[FileTypes, str, None] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SplitResponse:
        """
        Split classification for documents.

        This endpoint classifies document sections based on markdown content and split
        options.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/split`.

        Args:
          split_class: List of split classification options/configuration. Can be provided as JSON
              string in form data.

          markdown: The Markdown file or Markdown content to split.

          markdown_url: The URL to the Markdown file to split.

          model: Model version to use for split classification. Defaults to the latest version.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_minimal(
            {
                "split_class": split_class,
                "markdown": markdown,
                "markdown_url": markdown_url,
                "model": model,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return await self.post(
            "/v1/ade/split",
            body=await async_maybe_transform(body, client_split_params.ClientSplitParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=SplitResponse,
        )

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=body)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=body)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=body)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=body)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=body)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=body)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=body)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=body)
        return APIStatusError(err_msg, response=response, body=body)


class LandingAIADEWithRawResponse:
    _client: LandingAIADE

    def __init__(self, client: LandingAIADE) -> None:
        self._client = client

        self.extract = to_raw_response_wrapper(
            client.extract,
        )
        self.parse = to_raw_response_wrapper(
            client.parse,
        )
        self.split = to_raw_response_wrapper(
            client.split,
        )

    @cached_property
    def parse_jobs(self) -> parse_jobs.ParseJobsResourceWithRawResponse:
        from .resources.parse_jobs import ParseJobsResourceWithRawResponse

        return ParseJobsResourceWithRawResponse(self._client.parse_jobs)


class AsyncLandingAIADEWithRawResponse:
    _client: AsyncLandingAIADE

    def __init__(self, client: AsyncLandingAIADE) -> None:
        self._client = client

        self.extract = async_to_raw_response_wrapper(
            client.extract,
        )
        self.parse = async_to_raw_response_wrapper(
            client.parse,
        )
        self.split = async_to_raw_response_wrapper(
            client.split,
        )

    @cached_property
    def parse_jobs(self) -> parse_jobs.AsyncParseJobsResourceWithRawResponse:
        from .resources.parse_jobs import AsyncParseJobsResourceWithRawResponse

        return AsyncParseJobsResourceWithRawResponse(self._client.parse_jobs)


class LandingAIADEWithStreamedResponse:
    _client: LandingAIADE

    def __init__(self, client: LandingAIADE) -> None:
        self._client = client

        self.extract = to_streamed_response_wrapper(
            client.extract,
        )
        self.parse = to_streamed_response_wrapper(
            client.parse,
        )
        self.split = to_streamed_response_wrapper(
            client.split,
        )

    @cached_property
    def parse_jobs(self) -> parse_jobs.ParseJobsResourceWithStreamingResponse:
        from .resources.parse_jobs import ParseJobsResourceWithStreamingResponse

        return ParseJobsResourceWithStreamingResponse(self._client.parse_jobs)


class AsyncLandingAIADEWithStreamedResponse:
    _client: AsyncLandingAIADE

    def __init__(self, client: AsyncLandingAIADE) -> None:
        self._client = client

        self.extract = async_to_streamed_response_wrapper(
            client.extract,
        )
        self.parse = async_to_streamed_response_wrapper(
            client.parse,
        )
        self.split = async_to_streamed_response_wrapper(
            client.split,
        )

    @cached_property
    def parse_jobs(self) -> parse_jobs.AsyncParseJobsResourceWithStreamingResponse:
        from .resources.parse_jobs import AsyncParseJobsResourceWithStreamingResponse

        return AsyncParseJobsResourceWithStreamingResponse(self._client.parse_jobs)


Client = LandingAIADE

AsyncClient = AsyncLandingAIADE
