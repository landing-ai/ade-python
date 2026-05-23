#!/usr/bin/env python
"""
Test script for async extract jobs in the ade-python SDK.
"""
import os
import json
import asyncio
import time
from pathlib import Path
from landingai_ade import LandingAIADE, AsyncLandingAIADE

# Configuration
API_KEY = os.getenv("LANDING_LENS_API_KEY")
BASE_URL = "http://localhost:5300"  # Local dev server

# Test schema for extraction
test_schema = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "revenue": {"type": "string"},
        "employees": {"type": "number"}
    }
}

# Test markdown content
test_markdown = """
# Company Report

**Company Name**: Acme Corp

## Financial Overview
- **Annual Revenue**: $5.2 million
- **Employees**: 42

## Summary
Acme Corp is a growing technology company specializing in AI solutions.
"""


def test_sync_extract_jobs():
    """Test synchronous extract jobs API"""
    print("\n=== Testing Synchronous Extract Jobs API ===")

    client = LandingAIADE(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # Create extract job
    print("\n1. Creating extract job...")
    response = client.extract_jobs.create(
        schema=json.dumps(test_schema),
        markdown=test_markdown,
        model="extract-latest"
    )

    job_id = response.job_id
    print(f"   Job created with ID: {job_id}")

    # Poll for job status
    print("\n2. Checking job status...")
    max_attempts = 30
    for attempt in range(max_attempts):
        status_response = client.extract_jobs.get(job_id)
        print(f"   Attempt {attempt + 1}: Status = {status_response.status}")

        if status_response.status in ["completed", "failed"]:
            break

        time.sleep(2)

    # Display results
    if status_response.status == "completed":
        print("\n3. Job completed successfully!")
        print(f"   Extracted data: {json.dumps(status_response.extraction_response, indent=2)}")
        print(f"   Credits used: {status_response.credits_used}")
    else:
        print(f"\n3. Job failed with status: {status_response.status}")
        if hasattr(status_response, 'error_message'):
            print(f"   Error: {status_response.error_message}")

    # List jobs
    print("\n4. Listing extract jobs...")
    jobs_list = client.extract_jobs.list(limit=5)
    print(f"   Found {len(jobs_list.jobs)} jobs")
    for job in jobs_list.jobs[:3]:  # Show first 3
        print(f"   - Job {job.job_id}: {job.status}")


async def test_async_extract_jobs():
    """Test asynchronous extract jobs API"""
    print("\n=== Testing Asynchronous Extract Jobs API ===")

    client = AsyncLandingAIADE(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # Create extract job
    print("\n1. Creating extract job...")
    response = await client.extract_jobs.create(
        schema=json.dumps(test_schema),
        markdown=test_markdown,
        model="extract-latest"
    )

    job_id = response.job_id
    print(f"   Job created with ID: {job_id}")

    # Poll for job status
    print("\n2. Checking job status...")
    max_attempts = 30
    for attempt in range(max_attempts):
        status_response = await client.extract_jobs.get(job_id)
        print(f"   Attempt {attempt + 1}: Status = {status_response.status}")

        if status_response.status in ["completed", "failed"]:
            break

        await asyncio.sleep(2)

    # Display results
    if status_response.status == "completed":
        print("\n3. Job completed successfully!")
        print(f"   Extracted data: {json.dumps(status_response.extraction_response, indent=2)}")
        print(f"   Credits used: {status_response.credits_used}")
    else:
        print(f"\n3. Job failed with status: {status_response.status}")
        if hasattr(status_response, 'error_message'):
            print(f"   Error: {status_response.error_message}")

    # List jobs
    print("\n4. Listing extract jobs...")
    jobs_list = await client.extract_jobs.list(limit=5)
    print(f"   Found {len(jobs_list.jobs)} jobs")
    for job in jobs_list.jobs[:3]:  # Show first 3
        print(f"   - Job {job.job_id}: {job.status}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Async Extract Jobs Support in ade-python SDK")
    print("=" * 60)

    if not API_KEY:
        print("ERROR: Please set LANDING_LENS_API_KEY environment variable")
        return

    # Test sync client
    test_sync_extract_jobs()

    # Test async client
    asyncio.run(test_async_extract_jobs())

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()