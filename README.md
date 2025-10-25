# 🧨 BLAST — A GPU-Accelerated SQLite3 Dumper

**BLAST** is a *bare-metal*, high-throughput SQLite3 file dumper written in C + CUDA.  
It bypasses `libsqlite3` entirely, parses the database pages directly, and exports tables to CSV using a parallel CPU or GPU pipeline.  

---

## 🚀 Features

- **Zero-dependency:** Reads raw `.db` files without linking against SQLite.
- **Parallel page decoding:** Multi-core OpenMP page walkers on CPU.
- **Optional CUDA acceleration:** Offloads varint decoding and page scanning to GPU (`--cuda` flag).
- **Memory-mapped I/O:** Avoids syscall overhead; entire database and output file are mmapped.
- **Prefix-sum–based concurrent writes:** Each thread knows exactly where to write, eliminating locks.
- **Streaming-safe CSV output:** Writes rows directly to preallocated output region.

---

## 🧱 Build

Requires:
- GCC / Clang for CPU build.
- NVIDIA CUDA toolkit (11.0 +) for GPU mode.
- An AVX2-capable CPU recommended.

### CPU-only build
```bash
gcc -O3 -fopenmp blast.cu -o blast
```

### GPU-enabled build
```bash
nvcc -O3 -Xcompiler -fopenmp -arch=sm_86 blast.cu -o blast
```
> Adjust `-arch=sm_XX` to match your GPU’s compute capability (e.g. `sm_70`, `sm_89`, etc.).

---

## ⚙️ Usage

```bash
./blast [--cuda] input.db output.csv
```

| Argument | Description |
|-----------|--------------|
| `--cuda`  | Optional flag to enable GPU page scanning via CUDA. |
| `input.db` | Path to a SQLite database file. |
| `output.csv` | Destination CSV file (overwritten). |

Example:
```bash
./blast mydata.db dump.csv
./blast --cuda big.db dump.csv
```

---

## 🧩 How It Works

### 1. Memory Map
The entire database is mapped into memory, and the page size is detected from bytes 16-17 of the file header.

### 2. Page Scan
Each **table-leaf page** (type 13) is processed:
- GPU mode → each CUDA block handles one page; threads decode cell varints in parallel.
- CPU mode → OpenMP threads iterate over pages concurrently.

### 3. Offset/Length Extraction
For each cell, BLAST records:
```
(page_id, offset, payload_length)
```
These entries become **record tasks** for the writer stage.

### 4. Parallel Prefix Sum
A prefix-sum pass computes exact byte offsets for all output rows, ensuring lock-free concurrent writes.

### 5. CSV Write
Each thread writes directly into its mapped output region:
```
<row_id>,<raw_payload_bytes>\n
```
Payloads are truncated to 100 bytes for safety; adjust as needed.
