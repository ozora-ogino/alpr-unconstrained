export IMAGE_NAME=alpr-unconstrained
export ENTRYPOINT=bash

init: # Setup pre-commit
	pip3 install black pylint isort pre-commit pytest
	pre-commit install --hook-type pre-commit --hook-type pre-push

lint: # Lint all files in this repository
	pre-commit run --all-files --show-diff-on-failure

test: # Run tests
	pytest tests.py


build:
	docker build -t ${IMAGE_NAME} .

run:
	docker run --rm -it \
	-v ${PWD}/data:/opt/data \
	-v ${PWD}/src/:/opt/src ${IMAGE_NAME} \
	${ENTRYPOINT}
