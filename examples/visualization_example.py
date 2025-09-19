"""Example demonstrating visualization of ADE parsing results."""

import os
from pathlib import Path
from typing import Optional

# Import the ADE client and visualization utilities
import ade
from ade import Client, VisualizationConfig, visualize_parse_response, save_chunk_images


def visualize_document(
    document_path: str,
    api_key: Optional[str] = None,
    output_dir: str = "visualization_output",
    custom_colors: bool = False
) -> None:
    """Demonstrate document visualization with ADE SDK.

    Args:
        document_path: Path to the document to parse and visualize.
        api_key: ADE API key (uses environment variable if not provided).
        output_dir: Directory to save visualization outputs.
        custom_colors: Whether to use custom color configuration.
    """
    # Initialize the ADE client
    client = Client(apikey=api_key) if api_key else Client()

    print(f"Parsing document: {document_path}")

    # Parse the document
    try:
        # Open the file and pass it to the API
        with open(document_path, 'rb') as f:
            response = client.ade.parse(document=f)
        print(f"Successfully parsed document with {len(response.chunks)} chunks")
    except Exception as e:
        print(f"Error parsing document: {e}")
        return

    # Print summary of chunks found
    chunk_types = {}
    for chunk in response.chunks:
        chunk_types[chunk.type] = chunk_types.get(chunk.type, 0) + 1

    print("\nChunk summary:")
    for chunk_type, count in chunk_types.items():
        print(f"  - {chunk_type}: {count}")

    # Create visualization configuration
    if custom_colors:
        # Custom color scheme (BGR format)
        config = VisualizationConfig(
            thickness=3,
            font_scale=0.6,
            text_bg_opacity=0.8,
            color_map={
                "text": (0, 0, 255),       # Red for text
                "table": (0, 255, 0),       # Green for tables
                "figure": (255, 0, 0),      # Blue for figures
                "formula": (255, 255, 0),   # Cyan for formulas
                "header": (128, 0, 255),    # Purple for headers
                "footer": (255, 128, 0),    # Orange for footers
            }
        )
        print("\nUsing custom visualization configuration")
    else:
        config = None  # Use default configuration
        print("\nUsing default visualization configuration")

    # Visualize the parsing results
    try:
        print(f"\nGenerating visualizations...")
        images = visualize_parse_response(
            document_path=document_path,
            response=response,
            output_dir=output_dir,
            config=config
        )
        print(f"Generated {len(images)} visualization(s)")
        print(f"Visualizations saved to: {output_dir}")
    except ImportError as e:
        print(f"\nVisualization dependencies not installed.")
        print("Install with: pip install 'ade-python[visualization]'")
        return
    except Exception as e:
        print(f"Error during visualization: {e}")
        return

    # Also extract individual chunk images
    chunk_output_dir = Path(output_dir) / "chunks"
    try:
        print(f"\nExtracting individual chunk images...")
        saved_chunks = save_chunk_images(
            document_path=document_path,
            response=response,
            output_dir=chunk_output_dir
        )
        print(f"Extracted images for {len(saved_chunks)} chunks")
        print(f"Chunk images saved to: {chunk_output_dir}")
    except Exception as e:
        print(f"Error extracting chunk images: {e}")


def main():
    """Main function to run the example."""
    import sys

    # Check if a document path was provided
    if len(sys.argv) < 2:
        print("Usage: python visualization_example.py <document_path> [--custom-colors]")
        print("\nExample:")
        print("  python visualization_example.py sample.pdf")
        print("  python visualization_example.py sample.pdf --custom-colors")
        sys.exit(1)

    document_path = sys.argv[1]
    use_custom_colors = "--custom-colors" in sys.argv

    # Check if file exists
    if not Path(document_path).exists():
        print(f"Error: File '{document_path}' not found")
        sys.exit(1)

    # Check for API key
    api_key = os.environ.get("ADE_API_KEY")
    if not api_key:
        print("Warning: ADE_API_KEY environment variable not set")
        print("Please set it with: export ADE_API_KEY='your-api-key'")
        sys.exit(1)

    # Run visualization
    visualize_document(
        document_path=document_path,
        custom_colors=use_custom_colors
    )


if __name__ == "__main__":
    main()