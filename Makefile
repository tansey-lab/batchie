python-unittest:
	pip install -e .[test]
	pytest ./src

.git/hooks/pre-commit:
	pip install pre-commit
	pre-commit install
