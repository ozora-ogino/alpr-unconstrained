# ALPR in Unscontrained Scenarios

This repository contains the unofficial implementation of CCV 2018 paper "License Plate Detection and Recognition in Unconstrained Scenarios" written in Python3.

The motivation of this project is refactoring [the author's official code](https://github.com/sergiomsilva/alpr-unconstrained) which written in Python2.

Additionally this project uses CCPD2019 instead of the author's private dataset used in his paper.

## How to use

1. Setting up repository with pre-commit and linter

```bash
make init
```

2. Build docker image

```bash
make build
```

3. Run docker

```bash
make run
```
