#!/usr/bin/env python3
"""
Invoice extraction example using the ADE SDK with Pydantic schemas.

Usage: python simple_schema_example.py <invoice.pdf>
"""

import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from LandingAIAde import Landingai

from ade.lib import pydantic_to_json_schema


class InvoiceData(BaseModel):
    """Schema for extracting invoice data."""
    invoice_number: str = Field(description="The invoice number")
    invoice_date: str = Field(description="The date of the invoice")
    due_date: Optional[str] = Field(
        default=None, description="The payment due date"
    )
    vendor_name: str = Field(description="The vendor/supplier name")
    vendor_address: Optional[str] = Field(
        default=None, description="The vendor's address"
    )
    customer_name: Optional[str] = Field(
        default=None, description="The customer/buyer name"
    )
    customer_address: Optional[str] = Field(
        default=None, description="The customer's address"
    )
    subtotal: Optional[float] = Field(
        default=None, description="Subtotal before tax"
    )
    tax_amount: Optional[float] = Field(
        default=None, description="Tax amount"
    )
    total_amount: float = Field(description="The total amount due")
    currency: Optional[str] = Field(
        default="USD", description="Currency code (USD, EUR, etc.)"
    )


def main() -> None:
    """Parse and extract data from an invoice."""

    # Check for API key
    api_key = os.getenv("LANDINGAI_API_KEY")
    if not api_key:
        print("Error: Please set LANDINGAI_API_KEY environment variable")
        print("export LANDINGAI_API_KEY='your-api-key'")
        sys.exit(1)

    # Get file path from command line
    if len(sys.argv) != 2:
        print("Usage: python simple_schema_example.py <invoice.pdf>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Processing: {file_path}")

    # Initialize client
    environment = os.getenv("LANDINGAI_ENVIRONMENT", "production")
    client = Landingai(apikey=api_key, environment=environment)  # type: ignore

    # Parse the document
    print("Parsing document...")
    with open(file_path, "rb") as f:
        parse_response = client.parse(document=f)
    print(f"Parsed {len(parse_response.chunks)} chunks, {len(parse_response.markdown)} characters")

    # Convert Pydantic schema to JSON schema
    schema = pydantic_to_json_schema(InvoiceData)

    # Extract structured data from the parsed markdown
    print("Extracting invoice data...")
    extract_response = client.extract(
        schema=schema,
        markdown=BytesIO(parse_response.markdown.encode('utf-8'))
    )

    # Display results
    if not extract_response.extraction:
        print("Error: No data extracted")
        sys.exit(1)

    if not isinstance(extract_response.extraction, dict):
        print(f"Error: Unexpected extraction type: {type(extract_response.extraction)}")
        sys.exit(1)

    try:
        invoice = InvoiceData(**extract_response.extraction)

        print("\n" + "-" * 50)
        print("EXTRACTED DATA")
        print("-" * 50)
        print(f"Invoice #:     {invoice.invoice_number}")
        print(f"Date:          {invoice.invoice_date}")
        print(f"Due Date:      {invoice.due_date or 'Not specified'}")
        print(f"Vendor:        {invoice.vendor_name}")
        print(f"Customer:      {invoice.customer_name or 'Not specified'}")

        if invoice.subtotal:
            print(f"Subtotal:      ${invoice.subtotal:,.2f}")
        if invoice.tax_amount:
            print(f"Tax:           ${invoice.tax_amount:,.2f}")

        print(f"Total Amount:  {invoice.currency} {invoice.total_amount:,.2f}")

    except Exception as e:
        print(f"Error parsing extraction data: {e}")
        print("Raw data:", extract_response.extraction)
        sys.exit(1)


if __name__ == "__main__":
    main()