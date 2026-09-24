.PHONY: install lint format test train notebook api dashboard docker-build docker-up docker-down

install:            ## Install all dependencies (runtime + dev)
	python -m pip install -r requirements/dev.txt

lint:               ## Run the linter
	ruff check src scripts tests

format:             ## Auto-format code
	ruff check --fix src scripts tests

test:               ## Run the test suite
	pytest

train:              ## Train the model (expects data/application_train.csv)
	python scripts/train.py --data-path data/application_train.csv

notebook:           ## Rebuild the walkthrough notebook (needs data/ and the API running)
	python scripts/build_walkthrough_notebook.py

api:                ## Run the API locally
	uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:          ## Run the Streamlit dashboard locally
	streamlit run src/dashboard.py --server.port 8501

docker-build:       ## Build the Docker images
	docker compose build

docker-up:          ## Start the full stack (API + dashboard)
	docker compose up --build

docker-down:        ## Stop the stack
	docker compose down
