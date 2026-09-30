# v0.1 release readiness checklist

This checklist tracks release gates; it does not authorize publishing. The
development version remains `0.1.0.dev0`. No tag, release, or upload is performed
by the packaging smoke or CI. See [compatibility evidence](compatibility.md) for
the specific environments and the [SPEC](spec-v0.1.md#v01-definition-of-done) for
acceptance. Local checks below were run during Task #12, not across every platform.

- [x] Full pytest passes locally (326 tests); offline packaging cases pass (16).
- [x] Ruff, applicable formatting, and diff whitespace checks pass locally.
- [x] Wheel builds and artifact inspection passes with correct version, package,
  README, LICENSE, and console entry point.
- [x] Clean temporary venv installs the wheel and runs help/run/list/diff without
  importing checkout code or requiring PyTorch/NVIDIA.
- [x] Existing Linux CPU editable-install CI passes (main `9ad30ba`).
- [x] New Linux Python 3.10 installed-wheel CI step passes at `fd2c2fd`; its job
  link and observed Python 3.10.21 environment are in the compatibility document.
- [x] Local macOS arm64 no-PyTorch unit-test evidence is recorded.
- [ ] Real CPU PyTorch workflow is verified on macOS arm64 and Linux CPU with
  exact tested versions recorded; no-PyTorch smoke is insufficient for this gate.
- [ ] Real Linux + NVIDIA controlled validation passes with the environment and
  committed checkout recorded. **NOT RUN / PENDING; mocks do not satisfy this.**
- [ ] Tested Python/PyTorch/OS matrix is selected, verified, and published.
- [ ] Installation, metric meanings, unavailable values, excluded features, and
  bootstrap limitations are reviewed against actual behavior.
- [ ] Owner reviews proposed SPEC/ADR decisions required for v0.1 acceptance;
  deferred details are not silently frozen.
- [ ] Final release version and publishing date are explicitly decided by owner.
- [ ] If distributing an sdist, build it and verify installation/workflow from
  that artifact; Task #12 currently validates the wheel only.
- [ ] Final distribution contents are checked for accidental files or private
  data; repository diff contains no generated wheel/venv/run evidence.
- [ ] Owner authorizes any tag, GitHub Release, or PyPI upload separately.

Repeat tests and wheel smoke for the eventual release commit, not merely this
development snapshot. Compatibility tests with PyTorch and real NVIDIA execution
remain follow-up work; packaging readiness does not establish complete v0.1
acceptance. Do not mark pending gates complete without evidence.
