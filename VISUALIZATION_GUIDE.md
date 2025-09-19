# ADE Python SDK - Visualization Guide

This guide explains how to use the visualization features added to the ADE Python SDK to visualize document parsing results.

## Installation

To use the visualization features, you need to install the optional visualization dependencies:

```bash
pip install 'ade-python[visualization]'
```

This will install the required packages:

- `opencv-python`: For image processing and drawing
- `Pillow`: For image handling
- `PyMuPDF`: For PDF rendering
- `numpy`: For array operations

## Basic Usage

### Visualizing Parse Results

```python
from ade import Client, visualize_parse_response

# Initialize client
client = Client(apikey="your-api-key")

# Parse a document
with open("path/to/document.pdf", "rb") as f:
    response = client.ade.parse(document=f)

# Visualize the results
images = visualize_parse_response(
    document_path="path/to/document.pdf",
    response=response,
    output_dir="output/"  # Optional: save visualizations to disk
)

# The function returns a list of PIL Images (one per page)
for i, img in enumerate(images):
    img.show()  # Display the image
    # Or save individually
    img.save(f"page_{i}.png")
```

### Custom Visualization Configuration

You can customize the appearance of the visualizations:

```python
from ade import VisualizationConfig, visualize_parse_response

# Create custom configuration
config = VisualizationConfig(
    thickness=3,                # Thicker bounding boxes
    font_scale=0.7,             # Larger text labels
    text_bg_opacity=0.8,        # More opaque label background
    color_map={
        "text": (0, 0, 255),    # Red for text (BGR format)
        "table": (0, 255, 0),   # Green for tables
        "figure": (255, 0, 0),  # Blue for figures
        "formula": (255, 255, 0), # Cyan for formulas
    }
)

# Use custom configuration
images = visualize_parse_response(
    document_path="path/to/document.pdf",
    response=response,
    config=config,
    output_dir="custom_output/"
)
```

### Extracting Individual Chunk Images

You can also extract individual chunks as separate images:

```python
from ade import save_chunk_images

# Extract and save chunk images
saved_chunks = save_chunk_images(
    document_path="path/to/document.pdf",
    response=response,
    output_dir="chunks/"
)

# saved_chunks is a dictionary mapping chunk IDs to file paths
for chunk_id, paths in saved_chunks.items():
    print(f"Chunk {chunk_id}: {paths}")
```

## Configuration Options

The `VisualizationConfig` class supports the following parameters:

- **thickness** (int): Thickness of bounding boxes in pixels (default: 2)
- **text_bg_color** (tuple): Background color for labels in BGR format (default: light gray)
- **text_bg_opacity** (float): Opacity of label background, 0.0-1.0 (default: 0.7)
- **padding** (int): Padding around text labels in pixels (default: 2)
- **font_scale** (float): Scale factor for font size (default: 0.5)
- **font** (int): OpenCV font type (default: cv2.FONT_HERSHEY_SIMPLEX)
- **color_map** (dict): Dictionary mapping chunk types to BGR colors

## Default Color Scheme

The default color map includes:

- **text**: Blue (255, 0, 0)
- **table**: Green (0, 255, 0)
- **figure**: Red (0, 0, 255)
- **marginalia**: Cyan (255, 255, 0)
- **formula**: Magenta (255, 0, 255)
- **code**: Yellow (0, 255, 255)
- **header**: Purple (128, 0, 128)
- **footer**: Teal (128, 128, 0)
- **footnote**: Olive (0, 128, 128)
- **list**: Silver (192, 192, 192)
- **title**: Dark gray (64, 64, 64)
- **subtitle**: Gray (96, 96, 96)
- **caption**: Light gray (160, 160, 160)

## Example Script

A complete example script is provided in `examples/visualization_example.py`:

```bash
# Run with default colors
python examples/visualization_example.py document.pdf

# Run with custom colors
python examples/visualization_example.py document.pdf --custom-colors
```

## Output Structure

When saving visualizations to disk:

- Main visualizations are saved as `{document_name}_viz_page_{n}.png`
- Individual chunks are saved in subdirectories by page: `chunks/page_{n}/{type}_{id}_{i}.png`

## Notes

- The visualization works with both PDF and image files
- PDF pages are rendered at 150 DPI by default (configurable via the `dpi` parameter)
- Colors are specified in BGR format (Blue, Green, Red) as used by OpenCV
- All coordinates are normalized (0.0 to 1.0) in the parse response and converted to absolute pixels for visualization
