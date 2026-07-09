#!/usr/bin/env -S rye run python
"""Runnable example: parse a document with the V2 (client.v2) sub-client.

This targets the ADE gateway host (api.ade.[env].landing.ai), separate from the
V1 host used by client.parse(). It is purely additive -- V1 usage elsewhere in
this SDK is unaffected.

Set VISION_AGENT_API_KEY (and optionally LANDINGAI_ADE_ENVIRONMENT) before running:

    export VISION_AGENT_API_KEY="My Apikey"
    export LANDINGAI_ADE_ENVIRONMENT="staging"  # optional; defaults to "production"
    ./examples/v2_parse.py
"""

from __future__ import annotations

from pathlib import Path

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2ParseResponse

# apikey is read from VISION_AGENT_API_KEY; environment from LANDINGAI_ADE_ENVIRONMENT
# (or pass apikey=..., environment=... explicitly).
client = LandingAIADE()

document_path = Path(__file__).parent / "sample.pdf"

# Create an async parse job for a (potentially large) document.
job = client.v2.parse_jobs.create(document=document_path, service_tier="priority")
print(f"Created parse job {job.job_id} (status={job.status})")

# Block until the job reaches a terminal state.
done = client.v2.parse_jobs.wait(job.job_id, timeout=600)
print(f"Job finished with status={done.status}")

if isinstance(done.result, V2ParseResponse):
    print(done.result.markdown[:200] if done.result.markdown else None)
else:
    print(f"No result available (error={done.error})")

# For smaller documents you can skip the job/wait dance entirely:
sync_result = client.v2.parse(document=document_path)
print(sync_result.markdown[:200] if sync_result.markdown else None)
