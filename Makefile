.PHONY: setup test app eval baseline cryosegnet train lint clean

# uv is the toolchain (https://docs.astral.sh/uv). `uv sync` makes .venv + installs
# the project (editable) + the dev group + locks to uv.lock. `uv run` auto-syncs.

setup:
	uv sync

test:
	uv run pytest -q

app:
	uv run streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

eval:
	uv run python eval/run_eval.py --demo

baseline:
	uv run python scripts/run_baseline.py --backend cryosegnet

cryosegnet:
	uv run python scripts/run_cryosegnet.py --cryosegnet-dir $(CRYOSEGNET_DIR)

train:
	uv run python scripts/train_junk_classifier.py --empiar 10025

lint:
	uv run ruff check src app scripts eval tests || true

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info build dist
