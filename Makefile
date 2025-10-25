# Simple build targets for the BLAST SQLite dumper

CC ?= gcc
NVCC ?= nvcc
SM ?= sm_86
CFLAGS ?= -O3 -fopenmp
NVCCFLAGS ?= -O3 -Xcompiler -fopenmp -arch=$(SM)
SRC := blast.cu

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
CLANG_FORMAT ?= clang-format

.PHONY: all cpu gpu clean install-dev format format-python format-cuda test

all: cpu

cpu: blast_cpu

blast_cpu: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o blast_cpu

gpu: blast_gpu

blast_gpu: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o blast_gpu

clean:
	rm -f blast_cpu blast_gpu

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

format: format-python format-cuda

format-python:
	$(PYTHON) -m black .

format-cuda:
	@cuda_sources="$(shell git ls-files '*.cu')"; \
	if [ -n "$$cuda_sources" ]; then \
		$(CLANG_FORMAT) -i $$cuda_sources; \
	fi

test:
	$(PYTHON) -m pytest
