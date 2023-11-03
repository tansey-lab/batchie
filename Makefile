python-unittest:
	pip install -e .[test]
	coverage run -m pytest ./src
	coverage json

.git/hooks/pre-commit:
	pip install pre-commit
	pre-commit install
