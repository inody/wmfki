# Weak-form Mean-field Kernel Identification

This distribution provides the numerical code, declared protocols, tests,
sanitized frozen inputs, reference values, and eight reference figure PDFs for
the L4DC numerical results. Manuscript source and conference-template files are
not part of this distribution.

## Setup and complete frozen verification

```bash
make setup
make verify
```

`make verify` runs the numerical unit tests, validates every file against
`RELEASE_MANIFEST.json`, regenerates the eight figures in staging and compares
their SHA-256 hashes with `reference/figures/`, and rebuilds the six-method table
summary before checking all 36 reported numerical values against
`reference/table1.json`.

Exact plot rendering requires TeX Live 2026 with `fontenc`, `times`, `type1cm`,
and `type1ec` available to Matplotlib. The L4DC conference class and the
manuscript are not required. The exported GitHub Actions workflow uses
`macos-15`, Python 3.13.11, uv 0.9.26, the frozen `uv.lock`, and the
checksum-pinned MacTeX 2026.0324 package. It runs this same `make verify` target
with read-only repository permissions. The exact-PDF check deliberately fails
rather than relaxing or skipping a renderer-dependent mismatch. A passing result
is established only by an actual workflow run; local workflow generation does
not claim GitHub CI success.

The checks can also be run separately:

```bash
make test
make manifest-check
make figures-check
make metric-table-check
```

`make figures` installs regenerated, hash-validated PDFs into
`reference/figures/`. Generated records and temporary rendering files are
written below `outputs/`.

## Reduced and full numerical reproduction

```bash
make reproduce-plan REPRO_WORKERS=2
make reproduce-smoke REPRO_WORKERS=2
```

The plan lists the 25-stage numerical workflow. The smoke command exercises the
shared simulation, fitting, verification, and dependency-injection paths without
running the fixed R=100/R=20 production loops.

Run the complete numerical calculation in a fresh output directory with:

```bash
make reproduce REPRO_WORKERS=4 \
  REPRO_OUTPUT=outputs/reproduction-current/full
```

For this distribution, `REPRO_OUTPUT` must be a new run directory under
`outputs/`, such as the example above; other locations are rejected before
creating run state. Resume an interrupted run by adding `REPRO_RESUME=1`.
Resume is only for the same interrupted run: it validates a versioned state file, the numerical source
contract (code, configurations, `pyproject.toml`, `uv.lock`, and recorded
references), and the SHA-256 hashes and expected path set of every completed
stage before reusing it. Legacy or unversioned state is incompatible. Changed
source identity, incompatible state records, or mismatched recorded artifacts
cause an error instead of silent reuse. Validation covers explicitly recorded
artifacts, not every unrecorded producer intermediate. Start a new calculation
in a fresh output directory rather than migrating an incompatible state.
A new output directory does not bypass release-manifest checks: modified
source files must first be restored or packaged as a new consistent release.

The full workflow creates fresh trajectories, estimator fits, verification
reports, the six-method comparison, table verification, timing measurements,
and all eight figure PDFs. Timing remains part of this full workflow rather
than a detached benchmark. Fresh wall times depend on hardware, BLAS, thread
configuration, and system load; timing speedups and log--log slopes are
descriptive and are not reference-accuracy pass/fail thresholds. The timing
stage still checks its declared sampling, record count, and sampled `P=50N`
protocol. Frozen verification replays the recorded timing values and requires
all eight PDFs to match byte-for-byte, while PDFs from a fresh full run that
contain newly measured timings are not expected to match the frozen reference
PDF bytes. The workflow does not build a manuscript.

## License

WMFKI project material is licensed under the MIT License; see `LICENSE`.
External dependencies and plotting tools retain their upstream rights; see
`THIRD_PARTY_NOTICES.md`.
