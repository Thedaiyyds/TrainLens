# Installed-wheel smoke check

A smoke test is a small end-to-end check of the essential workflow. This check
exercises a built wheel rather than the developer's editable installation.
It does not test PyTorch training or CUDA correctness.

From the checkout root, in a Python environment with this checkout installed:

```sh
python -m pip install -e '.[dev]'
python validation/packaging/run_smoke.py
```

The runner uses the existing [pip wheel flow](https://pip.pypa.io/en/stable/cli/pip_wheel/)
with build isolation and `--no-deps`. It checks distribution/version metadata,
the README embedded in metadata, the Apache-2.0 license file, package modules,
and `trainlens = trainlens.cli:app` console entry point. It then creates a new
venv without system site-packages and installs the wheel and its normal runtime
dependencies using pip. Build/install may need access to a package index; this
is a separate CI step, not a network-dependent unit test.

All CLI calls run from a newly created temporary project directory. Inherited
`PYTHONPATH`, `PYTHONHOME`, and `VIRTUAL_ENV` are removed. The runner checks
`trainlens.__file__` belongs to the new environment's site-packages, verifies
PyTorch is absent, and invokes that environment's actual console script:

1. `trainlens --help`
2. Two `trainlens run` invocations using that same venv Python and a tiny script.
3. Read the two persisted records through the installed RunStore.
4. `trainlens list` and `trainlens diff baseline experiment`.

The script also reports its imported TrainLens path so the training child cannot
silently use checkout code. Outcomes must succeed; CUDA metrics must have no
device values and explicitly report `pytorch_unavailable`, never invented zeros.
The diff must be Markdown with identities and unavailable metrics explained.

The runner prints JSON evidence with OS, architecture, Python, package version,
artifact filename/contents, and imported package path. It returns 0 on success
and nonzero on failure. Its wheel, venv, scripts, and `.trainlens` records are
temporary and cleaned up; paths in printed evidence are observations, not retained
artifacts. Setuptools may leave ignored `build/` and `src/*.egg-info/` files in
the checkout; the runner does not delete existing developer files.

Offline unit checks:

```sh
pytest tests/test_packaging_smoke.py
ruff check .
```

CI keeps editable-install pytest/Ruff checks and runs the wheel smoke in the same
Linux Python 3.10 job. No publishing workflow is added. sdist build/install is
not verified by this runner; it remains a release checklist item if distributed.
