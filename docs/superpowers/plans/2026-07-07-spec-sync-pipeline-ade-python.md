# Spec-Sync Pipeline (ade-python) Implementation Plan — Problem 3

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up an automated, unattended pipeline in ade-python that mechanically tracks the live OpenAPI spec(s), opens a two-phase PR on drift (mechanical snapshot+models commit, then an AI wiring commit), and gates that PR so the released public surface can never change silently — replacing what Stainless's codegen/release machinery used to do.

**Architecture:** A single scheduled GitHub Actions workflow (`spec-sync.yml`) fetches and normalizes the live spec, diffs it against a committed snapshot (`specs/v1-ade.json`), and on drift creates one branch with two attributed commits: **(1) mechanical** — updated snapshot + regenerated *reference* pydantic models (`datamodel-code-generator`); **(2) AI** — `anthropics/claude-code-action` in automation mode wires the Stainless-conventioned resources/types/tests/docs from the diff. PR gates (`griffe` surface-lock against the last **release tag**, contract tests vs **staging**, plus existing lint/test/typecheck) run as ordinary checks; humans review and merge. The loop is driven by the **staging** spec; releases gate on the **production** spec.

**Tech Stack:** GitHub Actions, bash + `jq`, Rye, `datamodel-code-generator` (reference models), `griffe` (Python surface-lock), `anthropics/claude-code-action@v1`, a GitHub App installation token.

## Global Constraints

- **Scope: ade-python only.** The doc says prove the loop here first on a real change, then port to ade-typescript (openapi-typescript + api-extractor). The TS port is a follow-up (Task 10 records it; not built here).
- **Purely additive to the SDK.** The pipeline may only *add* public surface; the surface-lock gate fails any change to **released** surface. Package stays `1.x`.
- **Reference-models, not type replacement (key adaptation).** This repo's V1 types are Stainless-shaped, hand-maintained pydantic/TypedDict code with backward-compat guarantees. `datamodel-code-generator` output is committed under `specs/_generated/` as a **reference** the AI phase reads and reviewers skim — it is **not** shipped and does **not** replace `src/landingai_ade/types/*`. The AI phase writes the actual shipped code in existing conventions (mirror `resources/parse_jobs.py`). *(Alternative — regenerating shipped types wholesale — is heavier and risks the compat guarantee; deferred.)*
- **Staging in, production out.** Automation targets the **staging** spec; the release gate checks the **production** spec. The surface-lock baseline is the **last release tag**, not the previous commit on `main` — merged-but-unreleased surface stays mutable; only released surface is locked.
- **Token gotcha (do not regress).** Commits pushed with the default `GITHUB_TOKEN` do **not** trigger other workflows (GitHub anti-recursion), so gate jobs would never run on the sync PR. Both the mechanical push and the PR creation **must** use a **GitHub App installation token** (or fine-grained PAT with `contents: write`, `pull-requests: write`). This is why the workflow mints an app token in step 1.
- **Spec URLs (verified reachable, HTTP 200):**
  - V1 production: `https://api.va.landing.ai/v1/ade/openapi.json`
  - V1 staging: `https://api.va.staging.landing.ai/v1/ade/openapi.json`
  - V2 (aide): behind auth today (401) — added to the pipeline only after the aide "expose the curated spec" ask lands (Problem 2). This plan wires V1; V2 is a one-line URL addition later (Task 8 note).
- **Required secrets (prerequisites, set before the workflow can run green):** `ANTHROPIC_API_KEY`; `SPEC_SYNC_APP_ID` + `SPEC_SYNC_APP_PRIVATE_KEY` (GitHub App installed on the repo with contents/PR write); `LANDINGAI_ADE_STAGING_APIKEY` (staging ADE key from the vault, for contract tests).
- **Problem 1 dependency.** The publish tail (`merge → self-hosted release-please → PyPI`) requires Problem 1's self-hosted release-please workflow. This plan builds spec-sync + gates up to **merge**; the release/publish tail and its production-spec release gate (Task 9) assume Problem 1 has landed, and are marked accordingly.

---

## File Structure

**New files**
- `specs/v1-ade.json` — committed, normalized V1 spec snapshot (replaces `.stats.yml` as the drift baseline).
- `specs/_generated/v1_models.py` — generated reference models (committed; input to the AI phase, not shipped).
- `scripts/spec-sync/fetch-normalize.sh` — fetch a spec URL, emit stable normalized JSON (`jq -S`).
- `scripts/spec-sync/check-drift.sh` — diff live vs committed; update file + signal drift via exit code.
- `scripts/spec-sync/gen-models.sh` — run `datamodel-code-generator` into `specs/_generated/`.
- `scripts/spec-sync/surface-lock.sh` — `griffe` breaking-change check vs last release tag.
- `scripts/spec-sync/release-gate.sh` — assert every implemented route exists in the **production** spec.
- `.github/workflows/spec-sync.yml` — the scheduled two-phase pipeline.
- `.github/workflows/pr-gates.yml` — surface-lock + contract-tests on PRs to `main`.
- `tests/contract/__init__.py`, `tests/contract/test_v1_smoke.py` — contract tests (marker `contract`) hitting staging.

**Modified files**
- `pyproject.toml` — add `datamodel-code-generator` and `griffe` to `[tool.rye].dev-dependencies`; register a `contract` pytest marker.
- `README.md` / `CONTRIBUTING.md` — short "how spec-sync works" section (final task).

---

## Task 1: Committed spec snapshot + fetch/normalize

**Files:**
- Create: `scripts/spec-sync/fetch-normalize.sh`, `specs/v1-ade.json`

**Interfaces:**
- Produces: `fetch-normalize.sh <url>` → prints stable, key-sorted JSON to stdout (deterministic across runs). `specs/v1-ade.json` = the current normalized V1 production spec.

- [ ] **Step 1: Write the fetch/normalize script**

```bash
# scripts/spec-sync/fetch-normalize.sh
#!/usr/bin/env bash
# Fetch a live OpenAPI spec and emit normalized (stable, key-sorted) JSON to stdout.
# Normalization makes byte-diffs meaningful: same spec content -> identical bytes.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: fetch-normalize.sh <spec-url>" >&2
  exit 2
fi

url="$1"
curl -fsSL --max-time 30 --retry 3 --retry-delay 2 "$url" | jq -S .
```

- [ ] **Step 2: Make it executable and verify determinism**

Run:
```bash
chmod +x scripts/spec-sync/fetch-normalize.sh
a=$(./scripts/spec-sync/fetch-normalize.sh https://api.va.landing.ai/v1/ade/openapi.json | shasum)
b=$(./scripts/spec-sync/fetch-normalize.sh https://api.va.landing.ai/v1/ade/openapi.json | shasum)
[ "$a" = "$b" ] && echo "deterministic: $a"
```
Expected: two identical shasums printed once as `deterministic: <hash>`.

- [ ] **Step 3: Commit the baseline snapshot**

Run:
```bash
mkdir -p specs
./scripts/spec-sync/fetch-normalize.sh https://api.va.landing.ai/v1/ade/openapi.json > specs/v1-ade.json
jq -e . specs/v1-ade.json >/dev/null && echo "valid json"
```
Expected: `valid json`.

- [ ] **Step 4: Commit**

```bash
git add scripts/spec-sync/fetch-normalize.sh specs/v1-ade.json
git commit -m "chore(spec-sync): add fetch/normalize script and V1 spec snapshot baseline"
```

---

## Task 2: Drift detector

**Files:**
- Create: `scripts/spec-sync/check-drift.sh`

**Interfaces:**
- Consumes: `fetch-normalize.sh` (Task 1).
- Produces: `check-drift.sh <spec-url> <committed-path>` → exit `0` (no drift), exit `10` (drift; overwrites `<committed-path>` with the normalized live spec). Any other non-zero = operational error (fetch failure).

- [ ] **Step 1: Write the script**

```bash
# scripts/spec-sync/check-drift.sh
#!/usr/bin/env bash
# Compare a live spec against its committed snapshot.
#   exit 0  -> no drift
#   exit 10 -> drift detected; <committed-path> updated in place with the live spec
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: check-drift.sh <spec-url> <committed-path>" >&2
  exit 2
fi

url="$1"
committed="$2"
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

"$here/fetch-normalize.sh" "$url" > "$tmp"

if [ -f "$committed" ] && diff -q "$committed" "$tmp" >/dev/null; then
  echo "no drift: $committed"
  exit 0
fi

cp "$tmp" "$committed"
echo "drift detected -> updated $committed"
exit 10
```

- [ ] **Step 2: Verify no-drift path**

Run:
```bash
chmod +x scripts/spec-sync/check-drift.sh
set +e
./scripts/spec-sync/check-drift.sh https://api.va.landing.ai/v1/ade/openapi.json specs/v1-ade.json
echo "exit=$?"
set -e
```
Expected: `no drift: specs/v1-ade.json` and `exit=0` (snapshot from Task 1 matches live).

- [ ] **Step 3: Verify drift path (deliberate mutation)**

Run:
```bash
cp specs/v1-ade.json /tmp/backup.json
jq '.info.version = "MUTATED"' /tmp/backup.json > specs/v1-ade.json
set +e; ./scripts/spec-sync/check-drift.sh https://api.va.landing.ai/v1/ade/openapi.json specs/v1-ade.json; echo "exit=$?"; set -e
git checkout -- specs/v1-ade.json   # restore true snapshot
```
Expected: `drift detected -> updated specs/v1-ade.json` and `exit=10`.

- [ ] **Step 4: Commit**

```bash
git add scripts/spec-sync/check-drift.sh
git commit -m "chore(spec-sync): add drift detector"
```

---

## Task 3: Reference-model codegen

**Files:**
- Modify: `pyproject.toml` (`[tool.rye].dev-dependencies` += `datamodel-code-generator`)
- Create: `scripts/spec-sync/gen-models.sh`, `specs/_generated/v1_models.py`

**Interfaces:**
- Produces: `gen-models.sh <spec-path> <out-path>` → writes reference pydantic models generated from the committed spec. Deterministic given the same spec.

- [ ] **Step 1: Add the dev dependency**

In `pyproject.toml`, add to `[tool.rye].dev-dependencies` (alongside `pyright`, `mypy`, ...):

```toml
    "datamodel-code-generator>=0.25",
    "griffe>=0.49",
```

Run: `rye sync --all-features`
Expected: both install.

- [ ] **Step 2: Write the codegen script**

```bash
# scripts/spec-sync/gen-models.sh
#!/usr/bin/env bash
# Generate REFERENCE pydantic models from a committed spec snapshot.
# Output is committed under specs/_generated/ as an input for the AI wiring phase
# and for human review — it is NOT shipped and does not replace src/landingai_ade/types/*.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: gen-models.sh <spec-path> <out-path>" >&2
  exit 2
fi

spec="$1"
out="$2"
mkdir -p "$(dirname "$out")"
rye run datamodel-codegen \
  --input "$spec" \
  --input-file-type openapi \
  --output "$out" \
  --output-model-type pydantic.BaseModel \
  --use-standard-collections \
  --use-schema-description \
  --field-constraints \
  --disable-timestamp
```

`--disable-timestamp` keeps output byte-stable across runs (no generation date), so the mechanical commit is reproducible.

- [ ] **Step 3: Generate and verify determinism**

Run:
```bash
chmod +x scripts/spec-sync/gen-models.sh
./scripts/spec-sync/gen-models.sh specs/v1-ade.json specs/_generated/v1_models.py
a=$(shasum specs/_generated/v1_models.py)
./scripts/spec-sync/gen-models.sh specs/v1-ade.json specs/_generated/v1_models.py
b=$(shasum specs/_generated/v1_models.py)
[ "${a%% *}" = "${b%% *}" ] && echo "stable" && rye run python -c "import ast; ast.parse(open('specs/_generated/v1_models.py').read()); print('parses')"
```
Expected: `stable` then `parses`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml requirements-dev.lock scripts/spec-sync/gen-models.sh specs/_generated/v1_models.py
git commit -m "chore(spec-sync): add reference-model codegen (datamodel-code-generator)"
```

---

## Task 4: Surface-lock gate (griffe)

**Files:**
- Create: `scripts/spec-sync/surface-lock.sh`

**Interfaces:**
- Produces: `surface-lock.sh` → exit `0` if the public API of `landingai_ade` has no breaking change vs the last release tag; non-zero otherwise. No release tag present → passes with a notice (fresh repo).

- [ ] **Step 1: Write the script**

```bash
# scripts/spec-sync/surface-lock.sh
#!/usr/bin/env bash
# Fail if the RELEASED public API of landingai_ade changed in a breaking way.
# Baseline = last release tag: released surface is the promise to users; merged-but-
# unreleased surface stays mutable.
set -euo pipefail

baseline="$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)"
if [ -z "$baseline" ]; then
  echo "surface-lock: no release tag found; skipping (nothing released yet)."
  exit 0
fi

echo "surface-lock: checking landingai_ade against $baseline"
rye run griffe check landingai_ade --against "$baseline" --search src
```

`griffe check` exits non-zero only on **breaking** changes (removed/renamed/retyped public API); purely additive changes pass — exactly the additive contract.

- [ ] **Step 2: Verify it passes on the current tree**

Run:
```bash
chmod +x scripts/spec-sync/surface-lock.sh
git fetch --tags --quiet
./scripts/spec-sync/surface-lock.sh; echo "exit=$?"
```
Expected: either the skip notice, or a pass against `v1.12.0` with `exit=0`.

- [ ] **Step 3: Verify it FAILS on a deliberate breaking change**

Run:
```bash
# temporarily rename a public method to simulate a break
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("src/landingai_ade/resources/parse_jobs.py")
s = p.read_text()
p.write_text(s.replace("    def list(", "    def list_BROKEN(", 1))
PY
set +e; ./scripts/spec-sync/surface-lock.sh; echo "exit=$?"; set -e
git checkout -- src/landingai_ade/resources/parse_jobs.py
```
Expected: griffe reports a breaking change and `exit` is non-zero.

- [ ] **Step 4: Commit**

```bash
git add scripts/spec-sync/surface-lock.sh
git commit -m "chore(spec-sync): add griffe surface-lock gate (baseline=last release tag)"
```

---

## Task 5: Contract tests vs staging

**Files:**
- Modify: `pyproject.toml` (register `contract` marker)
- Create: `tests/contract/__init__.py`, `tests/contract/test_v1_smoke.py`

**Interfaces:**
- Produces: a `contract`-marked pytest suite that runs against the live staging ADE API when `LANDINGAI_ADE_STAGING_APIKEY` is set; skipped otherwise (so local `./scripts/test` stays offline/green).

- [ ] **Step 1: Register the marker**

In `pyproject.toml`, under `[tool.pytest.ini_options]` add (create the `markers` key if absent):

```toml
markers = [
  "contract: hits the live staging API; requires LANDINGAI_ADE_STAGING_APIKEY",
]
```

- [ ] **Step 2: Write a minimal contract smoke test**

```python
# tests/contract/test_v1_smoke.py
from __future__ import annotations

import os

import pytest

from landingai_ade import LandingAIADE

pytestmark = pytest.mark.contract

STAGING_KEY = os.environ.get("LANDINGAI_ADE_STAGING_APIKEY")


@pytest.fixture()
def staging_client() -> LandingAIADE:
    if not STAGING_KEY:
        pytest.skip("LANDINGAI_ADE_STAGING_APIKEY not set")
    return LandingAIADE(apikey=STAGING_KEY, environment="staging")


def test_parse_jobs_list_reachable(staging_client: LandingAIADE) -> None:
    """The implemented V1 surface answers on staging (auth + routing sanity)."""
    result = staging_client.parse_jobs.list(page=0, page_size=1)
    assert hasattr(result, "jobs")
```

> When the pipeline adds `extract_jobs` (its inaugural change, Task 8), the AI phase adds an analogous `extract_jobs.list` contract test here.

- [ ] **Step 3: Verify it skips cleanly with no key, runs with one**

Run: `rye run pytest tests/contract -m contract -v`
Expected (no key set): 1 skipped. (With a staging key exported, it hits staging and passes.)

- [ ] **Step 4: Confirm the default suite still excludes contract tests by network**

Run: `rye run pytest tests/contract -q`
Expected: skipped (no key), so `./scripts/test` remains offline-safe.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/contract
git commit -m "test(spec-sync): add staging contract-test scaffold (contract marker)"
```

---

## Task 6: The spec-sync workflow (two-phase)

**Files:**
- Create: `.github/workflows/spec-sync.yml`

**Interfaces:**
- Consumes: all scripts above; secrets `SPEC_SYNC_APP_ID`, `SPEC_SYNC_APP_PRIVATE_KEY`, `ANTHROPIC_API_KEY`.
- Produces: on drift, a branch `spec-sync/v1-<utc>` with a mechanical commit then an AI commit, and an open PR to `main` — pushed with the app token so gate workflows fire.

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/spec-sync.yml
name: Spec Sync
on:
  schedule:
    - cron: '0 */6 * * *'   # every ~6h; cron covers correctness, dispatch cuts latency
  workflow_dispatch: {}

permissions:
  contents: read            # writes happen via the app token, not GITHUB_TOKEN

jobs:
  spec-sync:
    if: github.repository == 'landing-ai/ade-python'
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      V1_SPEC_URL: https://api.va.staging.landing.ai/v1/ade/openapi.json  # staging drives the loop
    steps:
      - name: Mint GitHub App token
        id: app-token
        uses: actions/create-github-app-token@v1
        with:
          app-id: ${{ secrets.SPEC_SYNC_APP_ID }}
          private-key: ${{ secrets.SPEC_SYNC_APP_PRIVATE_KEY }}

      - uses: actions/checkout@v6
        with:
          token: ${{ steps.app-token.outputs.token }}
          fetch-depth: 0     # tags needed for surface-lock baseline

      - name: Install Rye
        run: |
          curl -sSf https://rye.astral.sh/get | bash
          echo "$HOME/.rye/shims" >> "$GITHUB_PATH"
        env:
          RYE_VERSION: '0.44.0'
          RYE_INSTALL_OPTION: '--yes'

      - name: Install dependencies
        run: rye sync --all-features

      - name: Detect drift
        id: drift
        run: |
          set +e
          ./scripts/spec-sync/check-drift.sh "$V1_SPEC_URL" specs/v1-ade.json
          echo "code=$?" >> "$GITHUB_OUTPUT"

      - name: No drift
        if: steps.drift.outputs.code == '0'
        run: echo "specs in sync; nothing to do."

      - name: Fail on fetch error
        if: steps.drift.outputs.code != '0' && steps.drift.outputs.code != '10'
        run: |
          echo "spec fetch/normalize failed (exit ${{ steps.drift.outputs.code }})"; exit 1

      # ---- Phase 1: mechanical ----
      - name: Mechanical commit (snapshot + reference models)
        if: steps.drift.outputs.code == '10'
        id: mech
        run: |
          git config user.name "spec-sync[bot]"
          git config user.email "spec-sync@users.noreply.github.com"
          branch="spec-sync/v1-$(date -u +%Y%m%dT%H%M%SZ)"
          git checkout -b "$branch"
          ./scripts/spec-sync/gen-models.sh specs/v1-ade.json specs/_generated/v1_models.py
          git add specs/v1-ade.json specs/_generated/v1_models.py
          git commit -m "chore(spec-sync): update V1 spec snapshot + regenerated reference models"
          git push -u origin "$branch"
          echo "branch=$branch" >> "$GITHUB_OUTPUT"

      # ---- Phase 2: AI wiring (same checkout/branch) ----
      - name: AI wiring commit
        if: steps.drift.outputs.code == '10'
        uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          github_token: ${{ steps.app-token.outputs.token }}
          prompt: |
            The previous commit updated specs/v1-ade.json and regenerated reference
            models in specs/_generated/v1_models.py. Update the SDK to match the spec
            diff: add or adjust resource classes, method signatures, param TypedDicts,
            response models, tests, api.md, and README examples, mirroring the existing
            conventions in src/landingai_ade/resources/parse_jobs.py.
            Rules: PURELY ADDITIVE — never modify or remove an existing public signature
            (the surface-lock CI job will fail the PR); follow existing code style;
            run ./scripts/format; commit to the current branch.
          claude_args: |
            --max-turns 40
            --allowedTools "Edit,Write,Read,Bash(git *),Bash(rye *),Bash(./scripts/*)"

      - name: Open sync PR
        if: steps.drift.outputs.code == '10'
        env:
          GH_TOKEN: ${{ steps.app-token.outputs.token }}
        run: |
          gh pr create \
            --base main \
            --head "${{ steps.mech.outputs.branch }}" \
            --title "spec-sync: track V1 spec drift" \
            --body $'Automated spec-sync PR.\n\n- **Commit 1 (mechanical):** normalized spec snapshot + regenerated reference models.\n- **Commit 2 (AI):** resources/methods/tests/docs wired from the spec diff.\n\nGates (surface-lock, contract tests, lint/test/typecheck) must pass. **Human review required before merge.**'
```

- [ ] **Step 2: Validate YAML + action pin**

Run: `rye run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/spec-sync.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/spec-sync.yml
git commit -m "ci(spec-sync): add two-phase spec-sync workflow (mechanical + AI)"
```

> **Runtime verification** (post-merge, needs secrets): trigger `workflow_dispatch` on a branch where `specs/v1-ade.json` has been deliberately staled (e.g. delete a route) so drift is guaranteed, and confirm a branch + two commits + PR appear, and that PR triggers the gate workflows (proving the app-token path). This is the real proof and is folded into Task 8.

---

## Task 7: PR gate workflow

**Files:**
- Create: `.github/workflows/pr-gates.yml`

**Interfaces:**
- Consumes: `surface-lock.sh` (Task 4), contract tests (Task 5).
- Produces: two required checks on PRs to `main` — `surface-lock` (always) and `contract-tests` (against staging, using `LANDINGAI_ADE_STAGING_APIKEY`).

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/pr-gates.yml
name: PR Gates
on:
  pull_request:
    branches: [main]

jobs:
  surface-lock:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0     # tags for the release-tag baseline
      - name: Install Rye
        run: |
          curl -sSf https://rye.astral.sh/get | bash
          echo "$HOME/.rye/shims" >> "$GITHUB_PATH"
        env:
          RYE_VERSION: '0.44.0'
          RYE_INSTALL_OPTION: '--yes'
      - run: rye sync --all-features
      - name: Surface lock (no breaking change to released API)
        run: ./scripts/spec-sync/surface-lock.sh

  contract-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v6
      - name: Install Rye
        run: |
          curl -sSf https://rye.astral.sh/get | bash
          echo "$HOME/.rye/shims" >> "$GITHUB_PATH"
        env:
          RYE_VERSION: '0.44.0'
          RYE_INSTALL_OPTION: '--yes'
      - run: rye sync --all-features
      - name: Contract tests vs staging
        env:
          LANDINGAI_ADE_STAGING_APIKEY: ${{ secrets.LANDINGAI_ADE_STAGING_APIKEY }}
        run: rye run pytest tests/contract -m contract -v
```

> `contract-tests` skips its assertions if the secret is absent (Task 5's fixture), so forked PRs without secret access don't hard-fail; make `surface-lock` a required status check in branch protection, and `contract-tests` required for internal branches.

- [ ] **Step 2: Validate YAML**

Run: `rye run python -c "import yaml; yaml.safe_load(open('.github/workflows/pr-gates.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/pr-gates.yml
git commit -m "ci(spec-sync): add PR gates (surface-lock + staging contract tests)"
```

---

## Task 8: Inaugural proof run — supersede PR #93

**Files:** none new (validation task; may produce a pipeline-generated PR).

**Interfaces:** Consumes the full pipeline. Validates it end-to-end on the known V1 `extract/jobs` gap, then closes #93.

> **Decision recorded:** supersede #93 with pipeline output — keep #93 open as the reference for expected code, let the pipeline generate the change, compare, then close #93.

- [ ] **Step 1: Confirm the gap is real in the committed snapshot**

Run:
```bash
jq -r '.paths | keys[] | select(test("extract/jobs"))' specs/v1-ade.json
```
Expected: the `/v1/ade/extract/jobs` route(s) present in the spec but absent from `src/landingai_ade/resources/` (only `parse_jobs.py` exists) — confirming real drift the pipeline should close.

- [ ] **Step 2: Provision secrets** (ops; one-time)

Set repo secrets: `SPEC_SYNC_APP_ID`, `SPEC_SYNC_APP_PRIVATE_KEY` (GitHub App with contents/PR write, installed on `landing-ai/ade-python`), `ANTHROPIC_API_KEY`, `LANDINGAI_ADE_STAGING_APIKEY`. Verify: `gh secret list --repo landing-ai/ade-python` shows all four.

- [ ] **Step 3: Force a drift and dispatch the pipeline**

To make the inaugural run produce the extract/jobs wiring, temporarily point the snapshot behind the live spec so drift fires, then dispatch:
```bash
# on a scratch branch pushed to origin, or via the Actions UI:
gh workflow run spec-sync.yml --repo landing-ai/ade-python
```
Expected: a `spec-sync/v1-<utc>` branch with two commits and an open PR; the PR triggers `PR Gates` + existing `CI` (proving the app-token path).

- [ ] **Step 4: Compare against #93 and validate gates**

Diff the pipeline PR's shipped code against PR #93 (`zzhang/async_extract`): resources, param TypedDicts, response models. Confirm equivalence (methods `create/get/list`, sync+async, multipart handling), that `surface-lock` passes (additive), and `contract-tests` green against staging.
Expected: functionally equivalent surface; gates green.

- [ ] **Step 5: Merge the pipeline PR; close #93 as superseded**

Run:
```bash
gh pr comment 93 --repo landing-ai/ade-python --body "Superseded by the spec-sync pipeline's inaugural PR (equivalent surface, generated + gated). Closing in favor of it. Thanks @zzhang — your implementation was the reference we validated against."
gh pr close 93 --repo landing-ai/ade-python
```
Expected: #93 closed; the pipeline-generated extract/jobs support on `main`.

---

## Task 9: Release gate — staging-in, production-out *(depends on Problem 1 release-please)*

**Files:**
- Create: `scripts/spec-sync/release-gate.sh`
- Modify: the self-hosted release-please workflow from Problem 1 (add a gate job) — **only after Problem 1 lands**.

**Interfaces:**
- Produces: `release-gate.sh` → non-zero if any route the SDK implements is absent from the **production** spec (i.e., a staging-only feature is trying to release).

- [ ] **Step 1: Write the release-gate script**

```bash
# scripts/spec-sync/release-gate.sh
#!/usr/bin/env bash
# Block a release while the SDK implements a route that is not yet in PRODUCTION.
# (staging-only features hold the release train until they reach prod or are reverted.)
set -euo pipefail

prod_url="https://api.va.landing.ai/v1/ade/openapi.json"
prod="$(mktemp)"; trap 'rm -f "$prod"' EXIT
./scripts/spec-sync/fetch-normalize.sh "$prod_url" > "$prod"

# Routes the SDK implements are those present in the staging snapshot we shipped code for.
# A conservative check: every path in the committed snapshot must also exist in prod.
missing="$(jq -r --slurpfile p "$prod" '
  .paths | keys[] as $k | select(($p[0].paths | has($k)) | not) | $k
' specs/v1-ade.json)"

if [ -n "$missing" ]; then
  echo "release-gate: these implemented routes are not in production yet:" >&2
  echo "$missing" >&2
  exit 1
fi
echo "release-gate: all implemented routes exist in production."
```

- [ ] **Step 2: Verify against current prod/staging (should pass today)**

Run: `chmod +x scripts/spec-sync/release-gate.sh && ./scripts/spec-sync/release-gate.sh; echo "exit=$?"`
Expected: pass (`exit=0`) — staging and prod V1 specs currently match.

- [ ] **Step 3: Wire into release-please PR checks** *(after Problem 1)*

Add a `release-gate` job to the self-hosted release-please workflow, running on the `release-please--*` PR head, invoking `./scripts/spec-sync/release-gate.sh`. Make it a required check for merging release PRs.

- [ ] **Step 4: Commit**

```bash
git add scripts/spec-sync/release-gate.sh
git commit -m "ci(spec-sync): add production-spec release gate (staging-in, production-out)"
```

---

## Task 10: Docs + TS-port note

**Files:**
- Modify: `README.md` or `CONTRIBUTING.md` (a "How spec-sync works" section)

- [ ] **Step 1: Document the loop**

Add a short section: what `spec-sync.yml` does (cron + dispatch, two-phase PR), what `specs/*.json` and `specs/_generated/` are, the gates (surface-lock baseline = last release tag; contract tests vs staging), staging-in/production-out, and the required secrets. State that PRs are AI-drafted and **require human review**.

- [ ] **Step 2: Record the ade-typescript port as a follow-up**

Note that the same shape ports to `ade-typescript` with `openapi-typescript` (mechanical) + `api-extractor` (surface-lock); tracked separately.

- [ ] **Step 3: Commit**

```bash
git add README.md CONTRIBUTING.md
git commit -m "docs(spec-sync): document the spec-sync pipeline and TS-port follow-up"
```

---

## Dependencies & prerequisites (call out before executing)

- **Problem 1 (get off Stainless)** must land for the *publish tail*: the self-hosted release-please workflow (Task 9's gate hangs off it), removal of the `stainless-sdks/*` CI conditionals + `pkg.stainless.com` upload in `ci.yml`, and retiring `.stats.yml` (superseded by `specs/*.json`). Spec-sync PR *generation + gates* (Tasks 1–8) do **not** depend on Problem 1.
- **Secrets** (Task 8 step 2): GitHub App (`SPEC_SYNC_APP_ID`/`SPEC_SYNC_APP_PRIVATE_KEY`), `ANTHROPIC_API_KEY`, `LANDINGAI_ADE_STAGING_APIKEY`.
- **V2 (aide) spec**: add `specs/v2-aide.json` + its URL to `spec-sync.yml` only after aide serves the curated spec unauthenticated (the Problem 2 aide ask). One env var + one drift-check step; the two-phase machinery is unchanged.

## Self-Review

**Spec coverage (vs the doc's Problem 3):** committed snapshots replacing `.stats.yml` (Task 1) ✓; cron+dispatch workflow (Task 6) ✓; one job / two attributed phases / one branch (Task 6) ✓; mechanical codegen (Task 3, adapted to reference-models with rationale) ✓; AI phase via `anthropics/claude-code-action` automation mode with allowedTools + additive-contract prompt (Task 6) ✓; surface-lock via griffe, baseline=release tag (Task 4) ✓; contract tests vs staging (Task 5) ✓; existing lint/test/typecheck reused (unchanged CI) ✓; token gotcha via app token (Task 6, Global Constraints) ✓; staging-in/production-out with release gate (Task 9) ✓; prove on V1 extract/jobs gap + reconcile PR #93 (Task 8) ✓; per-repo/independent + TS port follow-up (Task 10) ✓.

**Placeholder scan:** every step has a runnable script/YAML/command; no TBD/TODO. The two items that genuinely can't run until infra exists (Task 8 dispatch, Task 9 wiring) are explicitly marked as dependent on secrets / Problem 1, not left vague.

**Consistency:** script names, exit-code contract (`10` = drift), the `specs/v1-ade.json` / `specs/_generated/v1_models.py` paths, and the `contract` marker are used identically across the scripts, the workflow, and the gate jobs.
