.PHONY: install lint typecheck test mlflow data dedup split import colab-package

install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not slow and not behavioral"

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

data:
	uv run python -m neuroscan.data.download --out data/raw

# Duplicate audit: data/duplicates.csv, data/duplicate_pairs.csv and the README figure
dedup:
	uv run python -m neuroscan.data.dedup --figure docs/figures/duplicate_pairs.png

# Fixed, duplicate-aware train/val/test split (versioned in git)
split:
	uv run python -m neuroscan.data.splits --out splits/v1.csv

# Import run bundle(s) into the local MLflow store: make import BUNDLE=runs/baseline-cnn
import:
	uv run python -m neuroscan.tracking.import_run $(BUNDLE)

# Source archive of the current commit for Colab (upload it to Drive: MyDrive/neuroscan/code/)
COMMIT := $(shell git rev-parse HEAD)
colab-package:
	@test -z "$$(git status --porcelain --untracked-files=no)" || \
		(echo "Commit your changes first: runs must point to an exact commit."; exit 1)
	mkdir -p dist
	git archive --format=tar.gz --prefix=neuroscan/ \
		--add-virtual-file=neuroscan/GIT_COMMIT:$(COMMIT) \
		-o dist/neuroscan-$(shell git rev-parse --short HEAD).tar.gz HEAD -- \
		src configs $(wildcard splits) pyproject.toml uv.lock README.md LICENSE
	@echo "Created dist/neuroscan-$(shell git rev-parse --short HEAD).tar.gz"
