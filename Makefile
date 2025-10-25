# Simple build targets for the BLAST SQLite dumper

CC ?= gcc
NVCC ?= nvcc
SM ?= sm_86
CFLAGS ?= -O3 -fopenmp
NVCCFLAGS ?= -O3 -Xcompiler -fopenmp -arch=$(SM)
SRC := blast.cu

.PHONY: all cpu gpu clean

all: cpu

cpu: blast_cpu

blast_cpu: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o blast_cpu

gpu: blast_gpu

blast_gpu: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o blast_gpu

clean:
	rm -f blast_cpu blast_gpu
