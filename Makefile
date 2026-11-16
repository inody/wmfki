PYTHON ?= .venv/bin/python
REPRO_WORKERS ?= 4
REPRO_OUTPUT ?=
REPRO_RESUME ?=
MPLCONFIGDIR ?= outputs/matplotlib-cache
METRIC_OUTPUT := outputs/frozen-metric-table

.PHONY: setup test manifest-check figures figures-check metric-table \
	metric-table-check verify reproduce-plan reproduce-smoke reproduce \
	reproduce-current-plan reproduce-current-smoke reproduce-current-data \
	reproduce-current-artifacts reproduce-current-all

setup:
	uv sync --frozen

test:
	$(PYTHON) -m unittest discover -s test -p 'test_*.py'

manifest-check:
	$(PYTHON) scripts/verify_release_manifest.py --root .

figures:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/regenerate_l4dc_figures.py --config configs/experiments/l4dc_frozen_figures.json

figures-check:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/regenerate_l4dc_figures.py --config configs/experiments/l4dc_frozen_figures.json --check

metric-table:
	rm -rf "$(METRIC_OUTPUT)"
	$(PYTHON) scripts/build_cross_example_metric_table.py --config configs/experiments/l4dc_frozen_metric_table.json --output "$(METRIC_OUTPUT)"

metric-table-check: metric-table
	$(PYTHON) scripts/verify_l4dc_numerical_values.py --summary "$(METRIC_OUTPUT)/summary.json" --reference reference/table1.json --output "$(METRIC_OUTPUT)/numerical_verification.json"

verify: test manifest-check figures-check metric-table-check

reproduce-plan reproduce-current-plan:
	$(PYTHON) scripts/reproduce_l4dc_current.py plan --workers $(REPRO_WORKERS)

reproduce-smoke reproduce-current-smoke:
	$(PYTHON) scripts/reproduce_l4dc_current.py smoke --workers $(REPRO_WORKERS) $(if $(REPRO_OUTPUT),--output "$(REPRO_OUTPUT)",) $(if $(REPRO_RESUME),--resume,)

reproduce-current-data:
	$(PYTHON) scripts/reproduce_l4dc_current.py data --workers $(REPRO_WORKERS) $(if $(REPRO_OUTPUT),--output "$(REPRO_OUTPUT)",) $(if $(REPRO_RESUME),--resume,)

reproduce-current-artifacts:
	@test -n "$(REPRO_OUTPUT)" || (echo "REPRO_OUTPUT is required" >&2; exit 2)
	$(PYTHON) scripts/reproduce_l4dc_current.py artifacts --workers $(REPRO_WORKERS) --output "$(REPRO_OUTPUT)"

reproduce reproduce-current-all:
	$(PYTHON) scripts/reproduce_l4dc_current.py all --workers $(REPRO_WORKERS) $(if $(REPRO_OUTPUT),--output "$(REPRO_OUTPUT)",) $(if $(REPRO_RESUME),--resume,)
