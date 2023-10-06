pylint:
	pylint . run.py

isort:
	isort . --jobs=0

black:
	black .

fmt: isort black

test:
	pytest

clean: ## Remove __pycache__ folders
	@find . | grep __pycache__ | xargs rm -rf
