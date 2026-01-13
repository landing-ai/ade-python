# Tests for save_to helper functions

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from landingai_ade._client import _save_response, _get_input_filename
from landingai_ade._exceptions import LandingAiadeError


class TestGetInputFilename:
    """Tests for _get_input_filename helper function."""

    def test_path_input(self) -> None:
        """Test with Path object input."""
        result = _get_input_filename(Path("/path/to/document.pdf"), None)
        assert result == "document"

    def test_string_path_input(self) -> None:
        """Test with string path input."""
        result = _get_input_filename("/path/to/document.pdf", None)
        assert result == "document"

    def test_tuple_input(self) -> None:
        """Test with tuple (filename, content, mime_type) input."""
        result = _get_input_filename(("myfile.pdf", b"content", "application/pdf"), None)
        assert result == "myfile"

    def test_bytes_input_falls_through(self) -> None:
        """Test that bytes input falls through to default."""
        result = _get_input_filename(b"raw bytes content", None)
        assert result == "output"

    def test_io_object_with_name(self) -> None:
        """Test with IO object that has a name attribute."""
        file_obj = io.BytesIO(b"content")
        file_obj.name = "/path/to/uploaded.pdf"
        result = _get_input_filename(file_obj, None)
        assert result == "uploaded"

    def test_io_object_without_name(self) -> None:
        """Test with IO object that has no name attribute."""
        file_obj = io.BytesIO(b"content")
        result = _get_input_filename(file_obj, None)
        assert result == "output"

    def test_url_input(self) -> None:
        """Test with URL input."""
        result = _get_input_filename(None, "https://example.com/path/to/document.pdf")
        assert result == "document"

    def test_url_with_query_params(self) -> None:
        """Test URL with query parameters."""
        result = _get_input_filename(None, "https://example.com/file.pdf?token=abc123")
        assert result == "file"

    def test_url_no_path(self) -> None:
        """Test URL with no meaningful path."""
        result = _get_input_filename(None, "https://example.com/")
        assert result == "url_input"

    def test_both_none(self) -> None:
        """Test when both inputs are None."""
        result = _get_input_filename(None, None)
        assert result == "output"

    def test_file_takes_precedence_over_url(self) -> None:
        """Test that file input takes precedence over URL."""
        result = _get_input_filename(Path("local.pdf"), "https://example.com/remote.pdf")
        assert result == "local"


class TestSaveResponse:
    """Tests for _save_response helper function."""

    def test_creates_folder_and_saves_file(self, tmp_path: Path) -> None:
        """Test that folder is created and file is saved."""
        output_folder = tmp_path / "output"
        mock_result = MagicMock()
        mock_result.to_json.return_value = '{"key": "value"}'

        _save_response(output_folder, "testfile", "parse", mock_result)

        expected_file = output_folder / "testfile_parse_output.json"
        assert expected_file.exists()
        assert expected_file.read_text() == '{"key": "value"}'

    def test_saves_to_existing_folder(self, tmp_path: Path) -> None:
        """Test saving to an already existing folder."""
        output_folder = tmp_path / "existing"
        output_folder.mkdir()
        mock_result = MagicMock()
        mock_result.to_json.return_value = '{"data": "test"}'

        _save_response(output_folder, "doc", "extract", mock_result)

        expected_file = output_folder / "doc_extract_output.json"
        assert expected_file.exists()

    def test_correct_filename_format(self, tmp_path: Path) -> None:
        """Test that filename follows the correct format."""
        mock_result = MagicMock()
        mock_result.to_json.return_value = "{}"

        for method in ["parse", "extract", "split"]:
            _save_response(tmp_path, "myinput", method, mock_result)
            expected_file = tmp_path / f"myinput_{method}_output.json"
            assert expected_file.exists(), f"Expected {expected_file} to exist"

    def test_raises_error_on_permission_denied(self, tmp_path: Path) -> None:
        """Test that OSError is wrapped in LandingAiadeError."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        mock_result = MagicMock()
        mock_result.to_json.return_value = "{}"

        try:
            with pytest.raises(LandingAiadeError, match="Failed to save"):
                _save_response(readonly_dir / "subdir", "file", "parse", mock_result)
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Test that string paths work as well as Path objects."""
        mock_result = MagicMock()
        mock_result.to_json.return_value = '{"test": true}'

        _save_response(str(tmp_path), "strpath", "split", mock_result)

        expected_file = tmp_path / "strpath_split_output.json"
        assert expected_file.exists()
