#!/usr/bin/env -S rye run python
"""Manual smoke test for the V2 (`client.v2`) endpoints against a live environment.

This drives every V2 surface end-to-end so you can confirm auth, routing, and the
response shapes against a real gateway. It is a manual/QA tool — not part of the
automated test suite (which uses mocked transports).

This is a *live* test that can hit real endpoints (and consume credits), so — unlike
the client itself, which defaults to `production` when no environment is configured —
this script defaults to `staging` when neither `--environment` nor
`LANDINGAI_ADE_ENVIRONMENT` is set. Pass `--environment production` (or set
`LANDINGAI_ADE_ENVIRONMENT=production`) explicitly if you really want production.

Setup
-----
    export VISION_AGENT_API_KEY=<your api key for the target environment>
    # optional: override the target environment (script default: staging).
    # V2 lives on api.ade.<env>.landing.ai
    export LANDINGAI_ADE_ENVIRONMENT=staging

Run
---
    ./examples/v2_smoke_test.py                        # extract + files (no document needed)
    ./examples/v2_smoke_test.py --document ./sample.pdf      # + parse (sync & job)
    ./examples/v2_smoke_test.py --document-url https://.../sample.pdf
    ./examples/v2_smoke_test.py --only extract,extract_jobs  # run a subset
    ./examples/v2_smoke_test.py --environment dev --async     # dev env, async client

Exit code is non-zero if any selected check failed, so it is CI-friendly too.
"""

from __future__ import annotations

import os
import sys
import asyncio
import argparse
import traceback
from typing import Any, List, Callable, Optional
from pathlib import Path

from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE, AsyncLandingAIADE

ALL_CHECKS = ["files", "extract", "extract_jobs", "parse", "parse_jobs"]

# A tiny self-contained markdown document + schema so extract/files can run without any file.
SAMPLE_MARKDOWN = "# Acme Inc. — Q1 Report\n\nTotal revenue for the quarter was **$1,250,000**.\n"


class RevenueSchema(BaseModel):
    """Demonstrates passing a pydantic model as the extract schema."""

    revenue: str = Field(description="The total revenue figure, verbatim")
    company: str = Field(description="The company name")


def _short(value: Any, limit: int = 200) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live smoke test for client.v2 endpoints.")
    p.add_argument(
        "--environment",
        default=None,
        help="production|eu|staging|dev (else LANDINGAI_ADE_ENVIRONMENT; this script defaults to staging if neither is set)",
    )
    p.add_argument("--document", type=Path, default=None, help="Local document (PDF/image) for the parse checks")
    p.add_argument("--document-url", default=None, help="Public document URL for the parse checks")
    p.add_argument("--only", default=None, help=f"Comma-separated subset of: {','.join(ALL_CHECKS)}")
    p.add_argument("--parse-model", default=None, help="Override parse model (e.g. dpt-3-latest)")
    p.add_argument("--extract-model", default=None, help="Override extract model (e.g. extract-latest)")
    p.add_argument("--timeout", type=float, default=600.0, help="wait() timeout for job checks (seconds)")
    p.add_argument("--use-async", "--async", dest="use_async", action="store_true", help="Exercise the async client")
    return p


def _resolve_environment(args: argparse.Namespace) -> Optional[str]:
    """Pick the target environment, defaulting this *live* smoke test to `staging`.

    The client itself defaults to `production` when no environment is configured,
    but that's the wrong default for a script that can hit real endpoints and
    consume credits. Only fall back to `staging` when the caller hasn't set
    either `--environment` or `LANDINGAI_ADE_ENVIRONMENT`.
    """
    explicit_environment: Optional[str] = args.environment
    if explicit_environment:
        return explicit_environment
    if os.environ.get("LANDINGAI_ADE_ENVIRONMENT"):
        return None  # let the client pick it up from the env var
    return "staging"


def _selected(only: Optional[str]) -> List[str]:
    if not only:
        return ALL_CHECKS
    chosen = [c.strip() for c in only.split(",") if c.strip()]
    bad = [c for c in chosen if c not in ALL_CHECKS]
    if bad:
        raise SystemExit(f"Unknown check(s): {bad}. Valid: {ALL_CHECKS}")
    return chosen


# --------------------------------------------------------------------------- sync

def run_sync(args: argparse.Namespace, checks: List[str]) -> int:
    kwargs: dict[str, Any] = {}
    environment = _resolve_environment(args)
    if environment:
        kwargs["environment"] = environment
    client = LandingAIADE(**kwargs)
    print(f"V1 base: {client.base_url}  |  V2 base: {client._v2_base_url}\n")

    results: dict[str, str] = {}
    file_ref: Optional[str] = None

    def record(name: str, fn: Callable[[], Any]) -> None:
        print(f"── {name} ".ljust(60, "─"))
        try:
            out = fn()
            results[name] = "PASS"
            print(f"   PASS  {_short(out)}\n")
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic harness
            results[name] = "FAIL"
            print(f"   FAIL  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print()

    if "files" in checks:
        def _files() -> Any:
            nonlocal file_ref
            file_ref = client.v2.files.upload(file=("doc.md", SAMPLE_MARKDOWN.encode(), "text/markdown"))
            return f"file_ref={file_ref}"

        record("files.upload", _files)

    if "extract" in checks:
        def _extract() -> Any:
            res = client.v2.extract(
                schema=RevenueSchema,
                markdown=SAMPLE_MARKDOWN,
                model=args.extract_model or None,  # type: ignore[arg-type]
            )
            return f"extraction={res.extraction}  version={res.metadata.version}"

        record("v2.extract (sync)", _extract)

    if "extract_jobs" in checks:
        def _extract_jobs() -> Any:
            job = client.v2.extract_jobs.create(schema=RevenueSchema, markdown=SAMPLE_MARKDOWN)
            done = client.v2.extract_jobs.wait(job.job_id, timeout=args.timeout)
            return f"job={done.job_id} status={done.status} result={'set' if done.result else 'none'}"

        record("v2.extract_jobs (create+wait)", _extract_jobs)

    doc_kwargs = _document_kwargs(args)
    if "parse" in checks:
        if doc_kwargs is None:
            print("── v2.parse (sync) ".ljust(60, "─") + "\n   SKIP  (no --document / --document-url)\n")
            results["v2.parse (sync)"] = "SKIP"
        else:
            record("v2.parse (sync)", lambda: _short(client.v2.parse(model=args.parse_model or None, **doc_kwargs).markdown))  # type: ignore[arg-type]

    if "parse_jobs" in checks:
        if doc_kwargs is None:
            print("── v2.parse_jobs ".ljust(60, "─") + "\n   SKIP  (no --document / --document-url)\n")
            results["v2.parse_jobs"] = "SKIP"
        else:
            def _parse_jobs() -> Any:
                job = client.v2.parse_jobs.create(**doc_kwargs)
                done = client.v2.parse_jobs.wait(job.job_id, timeout=args.timeout)
                return f"job={done.job_id} status={done.status}"

            record("v2.parse_jobs (create+wait)", _parse_jobs)

    return _summarize(results)


def _document_kwargs(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    if args.document is not None:
        return {"document": args.document}
    if args.document_url:
        return {"document_url": args.document_url}
    return None


# -------------------------------------------------------------------------- async

def run_async(args: argparse.Namespace, checks: List[str]) -> int:
    async def _main() -> int:
        kwargs: dict[str, Any] = {}
        environment = _resolve_environment(args)
        if environment:
            kwargs["environment"] = environment
        client = AsyncLandingAIADE(**kwargs)
        print(f"[async] V1 base: {client.base_url}  |  V2 base: {client._v2_base_url}\n")
        results: dict[str, str] = {}

        if "files" in checks:
            print("── [async] files.upload ".ljust(60, "─"))
            try:
                ref = await client.v2.files.upload(file=("doc.md", SAMPLE_MARKDOWN.encode(), "text/markdown"))
                results["files.upload"] = "PASS"
                print(f"   PASS  file_ref={ref}\n")
            except Exception as exc:  # noqa: BLE001
                results["files.upload"] = "FAIL"
                print(f"   FAIL  {type(exc).__name__}: {exc}\n")

        if "extract" in checks:
            print("── [async] v2.extract ".ljust(60, "─"))
            try:
                res = await client.v2.extract(schema=RevenueSchema, markdown=SAMPLE_MARKDOWN)
                results["v2.extract"] = "PASS"
                print(f"   PASS  extraction={res.extraction}\n")
            except Exception as exc:  # noqa: BLE001
                results["v2.extract"] = "FAIL"
                print(f"   FAIL  {type(exc).__name__}: {exc}\n")

        if "extract_jobs" in checks:
            print("── [async] v2.extract_jobs ".ljust(60, "─"))
            try:
                job = await client.v2.extract_jobs.create(schema=RevenueSchema, markdown=SAMPLE_MARKDOWN)
                done = await client.v2.extract_jobs.wait(job.job_id, timeout=args.timeout)
                results["v2.extract_jobs"] = "PASS"
                print(f"   PASS  job={done.job_id} status={done.status}\n")
            except Exception as exc:  # noqa: BLE001
                results["v2.extract_jobs"] = "FAIL"
                print(f"   FAIL  {type(exc).__name__}: {exc}\n")

        await client.close()
        return _summarize(results)

    return asyncio.run(_main())


# ------------------------------------------------------------------------- shared

def _summarize(results: dict[str, str]) -> int:
    print("═" * 60)
    for name, status in results.items():
        print(f"  {status:5}  {name}")
    failed = [n for n, s in results.items() if s == "FAIL"]
    print("═" * 60)
    print(f"{len(failed)} failed / {len(results)} run")
    return 1 if failed else 0


def main() -> int:
    args = _make_parser().parse_args()
    checks = _selected(args.only)
    if args.use_async:
        async_checks = [c for c in checks if c in {"files", "extract", "extract_jobs"}]
        if async_checks != checks:
            print("note: --async runs files/extract/extract_jobs only (parse checks are sync-only here)\n")
        return run_async(args, async_checks)
    return run_sync(args, checks)


if __name__ == "__main__":
    sys.exit(main())
