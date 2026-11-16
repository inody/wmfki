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

Exact plot rendering requires a LaTeX installation with the `fontenc` and
`times` packages available to Matplotlib. The L4DC conference class and the
manuscript are not required.

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

Resume an interrupted run by adding `REPRO_RESUME=1`. The full workflow creates
fresh trajectories, estimator fits, verification reports, the six-method
comparison, table verification, and all eight figure PDFs. It does not build a
manuscript.

## License

WMFKI project material is licensed under the MIT License; see `LICENSE`.
External dependencies and plotting tools retain their upstream rights; see
`THIRD_PARTY_NOTICES.md`.
