export IMAGE_NAME=alpr-unconstrained
export ENTRYPOINT=bash
export DEVICE=cpu

init: # Setup pre-commit
	pip3 install black pylint isort pre-commit pytest
	pre-commit install --hook-type pre-commit --hook-type pre-push

lint: # Lint all files in this repository
	pre-commit run --all-files --show-diff-on-failure

test: # Run tests
	pytest tests.py


build:
	docker build -t ${IMAGE_NAME}:cpu --build-arg VERSION=1.15.5-py3 .
	docker build -t ${IMAGE_NAME}:gpu --build-arg VERSION=1.15.5-gpu .

run:
	docker run --rm -it \
	-v ${PWD}/data:/opt/data \
	-v ${PWD}/src/:/opt/src ${IMAGE_NAME}:${DEVICE} \
	${ENTRYPOINT}
