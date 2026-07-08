#!/usr/bin/env -S rye run python
"""Runnable example: extract structured data with the V2 (client.v2) sub-client,
using a pydantic model directly as the extraction schema.

This targets the ADE gateway host (aide.[env].landing.ai), separate from the
V1 host used by client.extract(). It is purely additive -- V1 usage elsewhere
in this SDK is unaffected.

Set VISION_AGENT_API_KEY (and optionally LANDINGAI_ADE_ENVIRONMENT) before running:

    export VISION_AGENT_API_KEY="My Apikey"
    export LANDINGAI_ADE_ENVIRONMENT="staging"  # optional; defaults to "production"
    ./examples/v2_extract.py
"""

from __future__ import annotations

from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2ExtractResult


class Invoice(BaseModel):
    total: str = Field(description="Invoice grand total")
    vendor: str = Field(description="Name of the vendor issuing the invoice")


# apikey is read from VISION_AGENT_API_KEY; environment from LANDINGAI_ADE_ENVIRONMENT
# (or pass apikey=..., environment=... explicitly).
client = LandingAIADE()

# `schema` accepts a pydantic BaseModel subclass directly -- it's coerced to a
# JSON Schema dict for you. A dict or a JSON-encoded string also work.
result = client.v2.extract(
    schema=Invoice,
    markdown_url="https://example.com/invoice.md",
    strict=False,  # prune unsupported schema fields instead of raising a 422
)
print(result.extraction)

# Long-running extractions can instead go through the async jobs route:
job = client.v2.extract_jobs.create(schema=Invoice, markdown_url="https://example.com/invoice.md")
done = client.v2.extract_jobs.wait(job.job_id, timeout=600)
if isinstance(done.result, V2ExtractResult):
    print(done.result.extraction)
else:
    print(f"No result available (error={done.error})")
