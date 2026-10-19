PYTHON ?= .venv/bin/python
VENV ?= .venv

.PHONY: venv install run daily backtest test lint check serve clean

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip

install: venv
	$(VENV)/bin/python -m pip install -r requirements-dev.txt

run: install
	$(VENV)/bin/python -m app.daily

backtest: install
	$(VENV)/bin/python -m app.daily --backtest-days 365

serve:
	$(VENV)/bin/python -m uvicorn app.main:app --host $${ETORACLE_HOST:-127.0.0.1} --port $${ETORACLE_PORT:-8787}

test:
	$(PYTHON) -m pytest

lint:
	$(VENV)/bin/ruff check app tests

check: lint test

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
