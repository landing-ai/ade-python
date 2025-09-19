"""Visualization utilities for ADE parsing results."""

import math
from pathlib import Path
from typing import List, Union, Optional, Literal, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from PIL import Image
    import numpy as np
    from ..types import AdeParseResponse

from .config import VisualizationConfig

# Import visualization dependencies with graceful fallback
try:
    import cv2
    import numpy as np
    import pymupdf
    from PIL import Image
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    _import_error = e


def _check_visualization_dependencies() -> None:
    """Check if visualization dependencies are available."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError(
            "Visualization dependencies are not installed. "
            "Please install them with: pip install 'ade-python[visualization]'"
        ) from _import_error


def _get_file_type(file_path: Path) -> Literal["pdf", "image"]:
    """Detect if file is PDF or image by checking magic number.

    Args:
        file_path: Path to the file.

    Returns:
        "pdf" if file is a PDF, "image" otherwise.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
            if header == b"%PDF-":
                return "pdf"
            return "image"
    except Exception:
        # Fallback to extension check
        return "pdf" if file_path.suffix.lower() == ".pdf" else "image"


def _pdf_page_to_image(pdf_doc: "pymupdf.Document", page_idx: int, dpi: int = 150) -> "np.ndarray":
    """Convert a PDF page to an image.

    Args:
        pdf_doc: PyMuPDF document object.
        page_idx: Page index to convert.
        dpi: DPI for rendering.

    Returns:
        RGB image as numpy array.
    """
    page = pdf_doc[page_idx]
    # Scale image and use RGB colorspace
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    img: np.ndarray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, -1
    )
    # Ensure the image has 3 channels
    if img.shape[-1] == 4:  # If RGBA, drop the alpha channel
        img = img[..., :3]
    return img


def _read_image(image_path: Union[str, Path]) -> "np.ndarray":
    """Read an image file and return as RGB numpy array.

    Args:
        image_path: Path to image file.

    Returns:
        RGB image as numpy array.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Handle grayscale or RGBA
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = img[..., :3]

    return img


def _draw_bounding_box(
    img: "np.ndarray",
    box: dict,
    text: str,
    color_bgr: tuple,
    config: VisualizationConfig
) -> None:
    """Draw a bounding box with label on the image.

    Args:
        img: Image to draw on (modified in place).
        box: Bounding box with left, top, right, bottom coordinates (normalized).
        text: Label text to display.
        color_bgr: Box color in BGR format.
        config: Visualization configuration.
    """
    height, width = img.shape[:2]

    # Convert normalized coordinates to absolute
    xmin = max(0, math.floor(box["left"] * width))
    ymin = max(0, math.floor(box["top"] * height))
    xmax = min(width, math.ceil(box["right"] * width))
    ymax = min(height, math.ceil(box["bottom"] * height))

    # Draw bounding box
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bgr, config.thickness)

    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, config.font, config.font_scale, config.thickness
    )

    # Position text at top-left of box with background
    text_x = xmin + config.padding
    text_y = ymin - config.padding

    # Ensure text stays within image bounds
    if text_y - text_height < 0:
        text_y = ymin + text_height + config.padding

    # Draw text background
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (text_x - config.padding, text_y - text_height - config.padding),
        (text_x + text_width + config.padding, text_y + config.padding),
        config.text_bg_color,
        -1
    )
    cv2.addWeighted(
        overlay, config.text_bg_opacity, img, 1 - config.text_bg_opacity, 0, img
    )

    # Draw text
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        config.font,
        config.font_scale,
        color_bgr,
        config.thickness,
        cv2.LINE_AA
    )


def visualize_parse_response(
    document_path: Union[str, Path],
    response: "AdeParseResponse",
    *,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[VisualizationConfig] = None,
    dpi: int = 150
) -> List["Image.Image"]:
    """Visualize parsing results by drawing bounding boxes on the document.

    Args:
        document_path: Path to the original document (PDF or image).
        response: Parse response from ADE API containing chunks with grounding info.
        output_dir: Optional directory to save visualization images.
        config: Optional visualization configuration.
        dpi: DPI for PDF rendering.

    Returns:
        List of PIL Images with visualizations (one per page).

    Raises:
        ImportError: If visualization dependencies are not installed.
        ValueError: If document cannot be read.
    """
    _check_visualization_dependencies()

    document_path = Path(document_path)
    if not document_path.exists():
        raise ValueError(f"Document not found: {document_path}")

    if config is None:
        config = VisualizationConfig()

    file_type = _get_file_type(document_path)
    visualized_images: List[Image.Image] = []

    if file_type == "image":
        # Process single image
        img = _read_image(document_path)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw all chunks
        for i, chunk in enumerate(response.chunks):
            if chunk.grounding and chunk.grounding.box:
                box_dict = {
                    "left": chunk.grounding.box.left,
                    "top": chunk.grounding.box.top,
                    "right": chunk.grounding.box.right,
                    "bottom": chunk.grounding.box.bottom,
                }
                color = config.get_color(chunk.type)
                label = f"{i} {chunk.type}"
                _draw_bounding_box(img_bgr, box_dict, label, color, config)

        # Convert back to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        visualized_images.append(Image.fromarray(img_rgb))

        # Save if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{document_path.stem}_viz.png"
            visualized_images[0].save(output_file)

    else:  # PDF
        # Group chunks by page
        chunks_by_page = defaultdict(list)
        for chunk in response.chunks:
            if chunk.grounding:
                page_idx = chunk.grounding.page
                chunks_by_page[page_idx].append(chunk)

        # Process each page
        with pymupdf.open(document_path) as pdf_doc:
            num_pages = len(pdf_doc)

            for page_idx in range(num_pages):
                # Convert page to image
                img_rgb = _pdf_page_to_image(pdf_doc, page_idx, dpi)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # Draw chunks for this page
                if page_idx in chunks_by_page:
                    for i, chunk in enumerate(chunks_by_page[page_idx]):
                        if chunk.grounding and chunk.grounding.box:
                            box_dict = {
                                "left": chunk.grounding.box.left,
                                "top": chunk.grounding.box.top,
                                "right": chunk.grounding.box.right,
                                "bottom": chunk.grounding.box.bottom,
                            }
                            color = config.get_color(chunk.type)
                            label = f"{i} {chunk.type}"
                            _draw_bounding_box(img_bgr, box_dict, label, color, config)

                # Convert back to RGB for PIL
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                visualized_images.append(pil_image)

                # Save if output directory specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    output_file = output_path / f"{document_path.stem}_viz_page_{page_idx}.png"
                    pil_image.save(output_file)

    return visualized_images


def save_chunk_images(
    document_path: Union[str, Path],
    response: "AdeParseResponse",
    output_dir: Union[str, Path],
    dpi: int = 150
) -> dict:
    """Extract and save individual chunk images based on their bounding boxes.

    Args:
        document_path: Path to the original document (PDF or image).
        response: Parse response from ADE API containing chunks with grounding info.
        output_dir: Directory to save extracted chunk images.
        dpi: DPI for PDF rendering.

    Returns:
        Dictionary mapping chunk IDs to list of saved image paths.

    Raises:
        ImportError: If visualization dependencies are not installed.
    """
    _check_visualization_dependencies()

    document_path = Path(document_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_type = _get_file_type(document_path)
    saved_paths = defaultdict(list)

    if file_type == "image":
        img = _read_image(document_path)
        height, width = img.shape[:2]

        for chunk in response.chunks:
            if not chunk.grounding:
                continue

            chunk_id = str(hash(chunk.markdown[:50])) if hasattr(chunk, 'markdown') else chunk.type

            if not chunk.grounding or not chunk.grounding.box:
                continue

            # Extract region
            xmin = max(0, math.floor(chunk.grounding.box.left * width))
            ymin = max(0, math.floor(chunk.grounding.box.top * height))
            xmax = min(width, math.ceil(chunk.grounding.box.right * width))
            ymax = min(height, math.ceil(chunk.grounding.box.bottom * height))

            cropped = img[ymin:ymax, xmin:xmax]

            # Save cropped image
            save_path = output_dir / f"{chunk.type}_{chunk_id}.png"
            Image.fromarray(cropped).save(save_path)
            saved_paths[chunk_id].append(save_path)

    else:  # PDF
        with pymupdf.open(document_path) as pdf_doc:
            # Group chunks by page
            chunks_by_page = defaultdict(list)
            for chunk in response.chunks:
                if chunk.grounding:
                    page_idx = chunk.grounding.page
                    chunks_by_page[page_idx].append(chunk)

            # Process each page
            for page_idx, chunks in chunks_by_page.items():
                if page_idx >= len(pdf_doc):
                    continue

                img_rgb = _pdf_page_to_image(pdf_doc, page_idx, dpi)
                height, width = img_rgb.shape[:2]

                # Create page subdirectory
                page_dir = output_dir / f"page_{page_idx}"
                page_dir.mkdir(exist_ok=True)

                for chunk in chunks:
                    chunk_id = str(hash(chunk.markdown[:50])) if hasattr(chunk, 'markdown') else chunk.type

                    if not chunk.grounding or not chunk.grounding.box:
                        continue

                    # Extract region
                    xmin = max(0, math.floor(chunk.grounding.box.left * width))
                    ymin = max(0, math.floor(chunk.grounding.box.top * height))
                    xmax = min(width, math.ceil(chunk.grounding.box.right * width))
                    ymax = min(height, math.ceil(chunk.grounding.box.bottom * height))

                    cropped = img_rgb[ymin:ymax, xmin:xmax]

                    # Save cropped image
                    save_path = page_dir / f"{chunk.type}_{chunk_id}.png"
                    Image.fromarray(cropped).save(save_path)
                    saved_paths[chunk_id].append(save_path)

    return dict(saved_paths)