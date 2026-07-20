PYTHON ?= python

.PHONY: reproduce-tables verify common clean

reproduce-tables:
	$(PYTHON) scripts/build_tables.py --results-root results --output-dir results/tables

common:
	$(PYTHON) scripts/select_common_evaluation_set.py --results-root results

verify: reproduce-tables common
	$(PYTHON) scripts/verify_release.py --results-root results

clean:
	rm -f results/summaries/*.csv results/tables/*.csv results/tables/*.tex results/expected/*.csv manifests/artifacts.csv manifests/checksums.sha256
