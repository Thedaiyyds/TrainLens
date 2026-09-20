# Working on TrainLens

TrainLens is a local CLI for comparing PyTorch training runs: **Diff your PyTorch training runs.** Keep contributions small, reviewable, and aligned with v0.1.

## Before implementation

1. Read [the v0.1 specification](docs/spec-v0.1.md) and the ADRs in [docs/adr](docs/adr/), starting with [ADR 0001](docs/adr/0001-bootstrap-instrumentation.md). An ADR is an architectural decision record: it explains a design choice and its tradeoffs.
2. Check each document's status. The current spec deliberately defers implementation details, and ADR 0001 is proposed. Do not turn its detailed policies into additional product requirements or assume that proposed decisions are approved.
3. Run `git status --short` and inspect relevant staged and unstaged changes. Preserve existing work; do not reset, overwrite, stage, or commit unrelated changes. A dirty working tree means there are uncommitted changes, not that the repository needs cleaning.
4. Inspect the files, dependencies, and test configuration relevant to the issue. Reuse established patterns rather than inventing a new scaffold or toolchain.
5. Share a short plan before coding: the intended behavior, files likely to change, verification steps, and any unresolved decision. Routine work within the authorized scope does not need another approval round.

## Scope and product contracts

- Complete the current issue and its acceptance criteria. Avoid unrelated cleanup, speculative abstractions, new features, or dependency changes that the issue does not need.
- For a documentation-only task, change documentation only. Respect any explicit file limits.
- Preserve the run/list/diff workflow, local versioned JSON records, environment and Git metadata, training outcome, and Markdown report contracts in the spec.
- Keep non-CUDA functionality developable and testable on macOS/Apple Silicon. CUDA absence must produce explicit unavailable metrics, not fabricated zeros or failure of otherwise valid non-CUDA work.
- Keep training outcome separate from metric collection success. A successful collection must not hide failed training, and an optional metric failure must not redefine the training outcome.
- Preserve the bootstrap architecture: the training script and allocator instrumentation share a Python process. Do not substitute external GPU usage measurements for PyTorch peak allocated/reserved memory.
- Stay within single-process v0.1 scope and the spec's non-goals. Official CUDA validation requires Linux with an NVIDIA GPU; local CPU tests or mocks do not establish CUDA correctness.
- If the issue needs broader changes, explain why and propose a separate follow-up or a scope change before expanding the work.

## Architectural decisions

- Surface consequential choices before committing the implementation to them. Examples include changing process boundaries, persistent record format or compatibility, CLI behavior, supported platforms, dependencies that shape the architecture, or the meaning of a metric.
- Explain the problem, practical options, tradeoffs, recommendation, and effect on v0.1 in plain language. Link the relevant spec or ADR section.
- If an unresolved choice changes an agreed contract or requires the owner's preference, request a decision and continue independent work while it is pending. Do not silently treat a deferred policy as settled.
- Make routine, reversible implementation choices within the approved design without asking about every detail. State assumptions when they affect behavior or interpretation.
- When a decision is approved, record it in the appropriate spec or ADR if documentation changes are in scope. Otherwise report the documentation follow-up; do not edit files outside the authorized scope.

## Tests and lint

- Discover the repository's actual test and lint commands from its configuration, documentation, or CI. Do not invent commands or claim tooling exists before it does. Lint means automated checks for code quality and style; CI means checks run automatically for changes.
- For code changes, add or update tests for meaningful changed behavior and run the relevant tests, configured lint, and applicable formatting/type checks. Broaden testing when the change affects shared behavior or a failure reveals additional risk.
- Check non-CUDA behavior when touching execution, collection, or reporting. For CUDA changes, distinguish local checks from real Linux + NVIDIA validation and identify what remains unverified.
- For documentation-only changes, review accuracy, local links, consistency, scope, and whitespace. Do not create application code or a test scaffold merely to validate prose.
- Fix failures introduced by the change. Identify pre-existing failures separately. Report exact commands and outcomes, including checks skipped because tools or hardware are unavailable; never describe an unrun check as passing.

## Inspect the diff and self-review

- Before finishing, inspect `git status --short`, your full diff, and relevant staged changes. Include new files in the review: ordinary `git diff` does not show untracked file contents.
- Run `git diff --check` for tracked changes and check new files for whitespace errors as well. Avoid staging files just to make them visible to review tools.
- Review your changes as a maintainer would: do they solve the issue, preserve the spec, handle relevant failure paths, keep missing data honest, and avoid unnecessary complexity?
- Check for accidental files, sensitive data, unrelated edits, undocumented contract changes, and claims unsupported by tests. Verify that pre-existing user changes remain intact.
- Resolve findings within scope, then rerun affected checks. Self-review is a final correctness pass, not a substitute for tests or owner decisions.

## Communicating with the project owner

- Assume the owner is building their first project. Explain important engineering terms briefly when they first matter, connecting each to the practical consequence for TrainLens.
- For example, a schema is the structure of a saved record; a bootstrap runner sets up measurement before running the script; a regression is previously working behavior broken by a change. Explain terms in context rather than adding a glossary to every update.
- Use plain language and distinguish facts, assumptions, proposals, and verified results. Give concise progress updates during sustained work and surface blockers early.
- Finish with what changed and why, tests/lint or documentation checks performed, self-review results, remaining limitations, and any decision the owner needs to make. Link relevant files and keep routine implementation details brief.
