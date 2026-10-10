.PHONY: install test baseline compare tune evaluate explain final predict

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

baseline:
	python -m churn.train

compare:
	python -m churn.compare

tune:
	python -m churn.tune

evaluate:
	python -m churn.evaluate

explain:
	python -m churn.explain

final:
	python -m churn.final

predict:
	python -m churn.predict