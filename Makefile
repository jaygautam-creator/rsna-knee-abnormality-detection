.PHONY: install test lint scan variant

install:            ## editable install with dev tools
	pip install -e ".[dev]"

test:               ## unit tests (no competition data needed)
	pytest -q

lint:
	ruff check src scripts tests

scan:               ## secret scan of the full git history
	gitleaks detect --redact -v

# make variant V=v94 OWNER=<kaggle-username>   ->  build/v94, ready for `kaggle kernels push -p build/v94`
variant:
	python scripts/build_variant.py pipeline/variants/$(V).json --owner $(OWNER) --out build/$(V)
