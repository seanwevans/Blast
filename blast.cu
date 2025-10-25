/*
blast.cu

  Unified high-throughput SQLite table dumper

  Build
    nvcc -O3 -Xcompiler -fopenmp -march=native -arch=sm_86 blast.cu -o blast

  Usage
    blast [--cuda] [--simd|--nosimd] [--table <name>] input.db output.csv
*/

#include <fcntl.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__CUDACC__)
#define HD __host__ __device__
#else
#define HD
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#define PAGE_LEAF 13
#define PAGE_INTERIOR 5
#define MAX_CELLS 512
#define MAX_PATH_DEPTH 32

typedef struct {
  uint64_t row_id;    // synthetic id (page<<32 | cell)
  uint64_t rowid_val; // true SQLite rowid
  uint32_t page;
  uint16_t offset;
  uint16_t length;
} RecordTask;

typedef struct {
  uint32_t *pages;
  int count;
  int cap;
} PageList;

static void pagelist_init(PageList *pl) {
  pl->cap = 128;
  pl->count = 0;
  pl->pages = (uint32_t *)malloc(pl->cap * sizeof(uint32_t));
}

static void pagelist_push(PageList *pl, uint32_t page) {
  if (pl->count >= pl->cap) {
    pl->cap *= 2;
    pl->pages = (uint32_t *)realloc(pl->pages, pl->cap * sizeof(uint32_t));
  }
  pl->pages[pl->count++] = page;
}

static uint8_t *mmap_file(const char *path, size_t *sz) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    perror("open");
    exit(1);
  }
  struct stat st;
  if (fstat(fd, &st)) {
    perror("fstat");
    exit(1);
  }
  *sz = (size_t)st.st_size;
  uint8_t *p = (uint8_t *)mmap(NULL, *sz, PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  close(fd);
  return p;
}

static void *mmap_outfile(const char *path, size_t sz, int *fd_out) {
  int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    perror("open out");
    exit(1);
  }
  (void)posix_fallocate(fd, 0, (off_t)sz);
  void *m = mmap(NULL, sz, PROT_WRITE, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) {
    perror("mmap out");
    exit(1);
  }
  *fd_out = fd;
  return m;
}

static size_t detect_page_size(const uint8_t *db) {
  uint16_t ps = (uint16_t)((db[16] << 8) | db[17]);
  return ps == 1 ? 65536u : (size_t)ps;
}

static inline char *u64toa(uint64_t val, char *out) {
  char tmp[32];
  int i = 31;
  tmp[i--] = '\0';
  if (!val)
    tmp[i--] = '0';
  else
    while (val) {
      tmp[i--] = (char)('0' + (val % 10));
      val /= 10;
    }
  size_t len = (size_t)(31 - i);
  memcpy(out, tmp + i + 1, len - 1);
  return out + (len - 1);
}

static inline char *dtoa(double val, char *out) {
  char buf[32];
  int len = snprintf(buf, sizeof(buf), "%.17g", val);
  if (len < 0)
    len = 0;
  memcpy(out, buf, (size_t)len);
  return out + len;
}

static inline char *write_csv_text(const uint8_t *data, size_t len, char *out) {
  *out++ = '"';
  for (size_t i = 0; i < len; i++) {
    unsigned char c = data[i];
    if (c == '"') {
      *out++ = '"';
      *out++ = '"';
    } else {
      *out++ = (char)c;
    }
  }
  *out++ = '"';
  return out;
}

static inline char *write_blob_hex(const uint8_t *data, size_t len, char *out) {
  static const char hex[] = "0123456789ABCDEF";
  *out++ = 'X';
  *out++ = '\'';
  for (size_t i = 0; i < len; i++) {
    uint8_t byte = data[i];
    *out++ = hex[byte >> 4];
    *out++ = hex[byte & 0xF];
  }
  *out++ = '\'';
  return out;
}

HD static inline uint64_t read_varint_scalar(const uint8_t *p, int *len) {
  uint64_t v = 0;
  for (int i = 0; i < 9; i++) {
    uint8_t b = p[i];
    v = (v << 7) | (b & 0x7F);
    if (!(b & 0x80)) {
      *len = i + 1;
      return v;
    }
  }
  *len = 9;
  return v;
}

#ifdef __AVX2__
static inline uint64_t read_varint_avx2(const uint8_t *p, int *len) {
  __m128i v = _mm_loadu_si128((const __m128i *)p);
  __m128i mask = _mm_cmplt_epi8(v, _mm_set1_epi8(0x80));
  int m = _mm_movemask_epi8(mask);
  int idx = __builtin_ctz(m | 0x10000);
  *len = idx + 1;
  uint64_t x = 0;
  for (int i = 0; i <= idx; i++)
    x = (x << 7) | (p[i] & 0x7F);
  return x;
}
#endif

static int use_simd_runtime = 0;

static inline uint64_t read_varint_auto(const uint8_t *p, int *len) {
#if defined(__AVX2__)
  if (use_simd_runtime)
    return read_varint_avx2(p, len);
  else
    return read_varint_scalar(p, len);
#else
  (void)use_simd_runtime;
  return read_varint_scalar(p, len);
#endif
}

static char *decode_record(const uint8_t *payload, size_t payload_len,
                           uint64_t rowid_val, char *out) {
  int hdr_varlen = 0;
  uint64_t header_size = read_varint_auto(payload, &hdr_varlen);
  size_t header_end = (size_t)header_size;
  size_t header_pos = (size_t)hdr_varlen;
  size_t data_pos = header_end;
  int field_index = 0;

  while (header_pos < header_end && data_pos <= payload_len) {
    int vlen = 0;
    uint64_t serial_type = read_varint_auto(payload + header_pos, &vlen);
    header_pos += (size_t)vlen;
    if (field_index++ > 0)
      *out++ = ',';

    if (serial_type == 0) {
      if (field_index == 1)
        out = u64toa(rowid_val, out);
    } else if (serial_type == 7) {
      if (data_pos + 8 <= payload_len) {
        uint64_t bits = 0;
        for (int b = 0; b < 8; b++)
          bits = (bits << 8) | payload[data_pos + b];
        data_pos += 8;
        union {
          uint64_t u;
          double d;
        } conv;
        conv.u = bits;
        out = dtoa(conv.d, out);
      } else
        *out++ = '?';
    } else if (serial_type >= 1 && serial_type <= 6) {
      static const int blut[7] = {0, 1, 2, 3, 4, 6, 8};
      int blen = blut[serial_type];
      if (data_pos + (size_t)blen > payload_len)
        blen = (int)(payload_len - data_pos);
      uint64_t val = 0;
      for (int b = 0; b < blen; b++)
        val = (val << 8) | payload[data_pos + b];
      data_pos += (size_t)blen;
      out = u64toa(val, out);
    } else if (serial_type == 8) {
      *out++ = '0';
    } else if (serial_type == 9) {
      *out++ = '1';
    } else if (serial_type >= 12 && (serial_type % 2 == 0)) {
      size_t blen = (size_t)((serial_type - 12) / 2);
      if (data_pos + blen > payload_len)
        blen = payload_len - data_pos;
      out = write_blob_hex(payload + data_pos, blen, out);
      data_pos += blen;
    } else if ((serial_type >= 13) && (serial_type % 2 == 1)) {
      size_t tlen = (size_t)((serial_type - 13) / 2);
      if (data_pos + tlen > payload_len)
        tlen = payload_len - data_pos;
      out = write_csv_text(payload + data_pos, tlen, out);
      data_pos += tlen;
    } else
      *out++ = '?';
  }
  *out++ = '\n';
  return out;
}

#ifdef __CUDACC__
__global__ void gpu_scan_pages(const uint8_t *db, size_t page_sz,
                               size_t n_pages, RecordTask *out_tasks,
                               int *total) {
  int page_id = blockIdx.x;
  if (page_id >= (int)n_pages)
    return;
  const uint8_t *page = db + page_id * page_sz;
  if (page[0] != PAGE_LEAF)
    return;
  int nc = (page[3] << 8) | page[4];
  int base = atomicAdd(total, nc);
  for (int c = threadIdx.x; c < nc; c += blockDim.x) {
    uint16_t off = (page[8 + 2 * c] << 8) | page[8 + 2 * c + 1];
    int vlen1;
    uint64_t pay = read_varint_scalar(page + off, &vlen1);
    int vlen2;
    uint64_t rowid_val = read_varint_scalar(page + off + vlen1, &vlen2);
    RecordTask t;
    t.row_id = ((uint64_t)page_id << 32) | c;
    t.rowid_val = rowid_val;
    t.page = page_id;
    t.offset = (uint16_t)(off + vlen1 + vlen2);
    t.length = (uint16_t)pay;
    out_tasks[base + c] = t;
  }
}
#endif

static void collect_table_pages(const uint8_t *db, size_t db_sz, size_t page_sz,
                                uint32_t page_no, PageList *out) {
  if (page_no == 0 || (size_t)page_no * page_sz >= db_sz)
    return;
  const uint8_t *page = db + (size_t)(page_no - 1) * page_sz;
  uint8_t type = page[0];
  if (type == PAGE_LEAF) {
    pagelist_push(out, page_no - 1);
    return;
  }
  if (type != PAGE_INTERIOR)
    return;
  int nc = (page[3] << 8) | page[4];
  for (int i = 0; i < nc; i++) {
    uint16_t off = (page[12 + 2 * i] << 8) | page[12 + 2 * i + 1];
    uint32_t child = (page[off] << 24) | (page[off + 1] << 16) |
                     (page[off + 2] << 8) | (page[off + 3]);
    collect_table_pages(db, db_sz, page_sz, child, out);
  }
  uint32_t right =
      (page[8] << 24) | (page[9] << 16) | (page[10] << 8) | (page[11]);
  collect_table_pages(db, db_sz, page_sz, right, out);
}

static uint32_t find_table_rootpage_page(const uint8_t *db, size_t db_sz,
                                         size_t page_sz, uint32_t page_no,
                                         const char *table_name) {
  if (page_no == 0)
    return 0;
  size_t off = (size_t)(page_no - 1) * page_sz;
  if (off + page_sz > db_sz)
    return 0;

  const uint8_t *page = db + off;
  uint8_t type = page[0];
  int nc = (page[3] << 8) | page[4];

  if (type == PAGE_LEAF) {
    for (int c = 0; c < nc; c++) {
      uint16_t cell_off = (page[8 + 2 * c] << 8) | page[8 + 2 * c + 1];
      int vlen1;
      uint64_t payload_len = read_varint_scalar(page + cell_off, &vlen1);
      int vlen2;
      (void)read_varint_scalar(page + cell_off + vlen1, &vlen2);
      const uint8_t *payload = page + cell_off + vlen1 + vlen2;
      int hdrlen_bytes;
      uint64_t hdrlen = read_varint_scalar(payload, &hdrlen_bytes);
      size_t header_end = hdrlen;
      size_t hpos = (size_t)hdrlen_bytes;
      size_t dpos = header_end;
      char type_text[16] = {0}, name[128] = {0}, tbl[128] = {0};
      uint32_t root = 0;
      for (int col = 0; hpos < header_end; col++) {
        int slen;
        uint64_t st = read_varint_scalar(payload + hpos, &slen);
        hpos += (size_t)slen;
        if (st == 0)
          continue;
        if (st >= 13 && (st % 2)) {
          uint64_t len = (st - 13) / 2;
          if (dpos + len > payload_len)
            len = payload_len - dpos;
          if (col == 0)
            memcpy(type_text, payload + dpos, (len < 15 ? len : 15));
          if (col == 1)
            memcpy(name, payload + dpos, (len < 127 ? len : 127));
          if (col == 2)
            memcpy(tbl, payload + dpos, (len < 127 ? len : 127));
          dpos += (size_t)len;
        } else if (st >= 1 && st <= 6) {
          static const int blut[7] = {0, 1, 2, 3, 4, 6, 8};
          int blen = blut[st];
          if ((size_t)dpos + blen > payload_len)
            blen = (int)(payload_len - dpos);
          if (blen == 4 && col == 3) {
            root = (payload[dpos] << 24) | (payload[dpos + 1] << 16) |
                   (payload[dpos + 2] << 8) | payload[dpos + 3];
          }
          dpos += (size_t)blen;
        }
      }
      if (!strcmp(type_text, "table") && (!strcmp(name, table_name) ||
                                           !strcmp(tbl, table_name)))
        return root;
    }
  } else if (type == PAGE_INTERIOR) {
    for (int c = 0; c < nc; c++) {
      uint16_t cell_off = (page[12 + 2 * c] << 8) | page[12 + 2 * c + 1];
      uint32_t child = (page[cell_off] << 24) | (page[cell_off + 1] << 16) |
                       (page[cell_off + 2] << 8) | (page[cell_off + 3]);
      uint32_t res =
          find_table_rootpage_page(db, db_sz, page_sz, child, table_name);
      if (res)
        return res;
    }
    uint32_t right = (page[8] << 24) | (page[9] << 16) | (page[10] << 8) |
                     (page[11]);
    return find_table_rootpage_page(db, db_sz, page_sz, right, table_name);
  }

  return 0;
}

static uint32_t find_table_rootpage(const uint8_t *db, size_t db_sz,
                                    size_t page_sz, const char *table_name) {
  return find_table_rootpage_page(db, db_sz, page_sz, 1, table_name);
}

static void cpu_write_csv(const uint8_t *db, size_t page_sz, RecordTask *tasks,
                          int n_tasks, char *out, uint64_t *prefix) {
#pragma omp parallel for schedule(static, 16)
  for (int i = 0; i < n_tasks; i++) {
    const RecordTask *t = &tasks[i];
    const uint8_t *payload = db + (size_t)t->page * page_sz + t->offset;
    uint64_t start = prefix[i];
    char *dst = out + start;
    dst = u64toa(t->row_id, dst);
    *dst++ = ',';
    dst = u64toa(t->rowid_val, dst);
    *dst++ = ',';
    dst = decode_record(payload, (size_t)t->length, t->rowid_val, dst);
  }
}

static uint64_t *prefix_sum(uint32_t *lens, int n) {
  uint64_t *pref = (uint64_t *)malloc((size_t)(n + 1) * sizeof(uint64_t));
  pref[0] = 0;
  for (int i = 1; i <= n; i++)
    pref[i] = pref[i - 1] + (uint64_t)lens[i - 1];
  return pref;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(
        stderr,
        "usage: %s [--cuda] [--simd|--nosimd] [--table <name>] in.db out.csv\n",
        argv[0]);
    return 1;
  }

  int use_cuda = 0;
  const char *table_name = NULL;
  const char *in = NULL;
  const char *outp = NULL;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--cuda"))
      use_cuda = 1;
    else if (!strcmp(argv[i], "--simd"))
      use_simd_runtime = 1;
    else if (!strcmp(argv[i], "--nosimd"))
      use_simd_runtime = 0;
    else if (!strcmp(argv[i], "--table") && i + 1 < argc)
      table_name = argv[++i];
    else if (!in)
      in = argv[i];
    else if (!outp)
      outp = argv[i];
  }

  if (!in || !outp) {
    fprintf(stderr, "missing in/out\n");
    return 1;
  }

  size_t db_sz;
  uint8_t *db = mmap_file(in, &db_sz);
  size_t page_sz = detect_page_size(db);
  size_t n_pages = db_sz / page_sz;

  fprintf(stderr, "[+] pages=%zu size=%zu use_cuda=%d simd=%d\n", n_pages,
          page_sz, use_cuda, use_simd_runtime);

  uint32_t rootpage = 0;
  if (table_name) {
    rootpage = find_table_rootpage(db, db_sz, page_sz, table_name);
    if (!rootpage) {
      fprintf(stderr, "[!] table not found\n");
      return 1;
    }
    fprintf(stderr, "[+] table '%s' rootpage=%u\n", table_name, rootpage);
  }

  PageList leafs;
  pagelist_init(&leafs);
  if (table_name)
    collect_table_pages(db, db_sz, page_sz, rootpage, &leafs);
  else
    for (uint32_t i = 1; i < n_pages; i++)
      pagelist_push(&leafs, i);

  fprintf(stderr, "[+] collected %d leaf pages\n", leafs.count);

  RecordTask *tasks =
      (RecordTask *)malloc(leafs.count * MAX_CELLS * sizeof(RecordTask));
  int n_tasks = 0;

#ifdef __CUDACC__
  if (use_cuda) {
    RecordTask *d_tasks;
    int *d_total;
    int zero = 0;
    cudaMalloc(&d_tasks, leafs.count * MAX_CELLS * sizeof(RecordTask));
    cudaMalloc(&d_total, sizeof(int));
    cudaMemcpy(d_total, &zero, sizeof(int), cudaMemcpyHostToDevice);
    uint8_t *d_db;
    cudaMalloc(&d_db, db_sz);
    cudaMemcpy(d_db, db, db_sz, cudaMemcpyHostToDevice);
    for (int p = 0; p < leafs.count; p++)
      gpu_scan_pages<<<1, 64>>>(d_db, page_sz, leafs.pages[p] + 1, d_tasks,
                                d_total);
    cudaDeviceSynchronize();
    cudaMemcpy(&n_tasks, d_total, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tasks, d_tasks, n_tasks * sizeof(RecordTask),
               cudaMemcpyDeviceToHost);
    cudaFree(d_tasks);
    cudaFree(d_total);
    cudaFree(d_db);
    fprintf(stderr, "[+] GPU decoded %d tasks\n", n_tasks);
  } else
#endif
  {
#pragma omp parallel for schedule(static, 8) reduction(+ : n_tasks)
    for (int pi = 0; pi < leafs.count; pi++) {
      uint32_t pg = leafs.pages[pi];
      const uint8_t *page = db + pg * page_sz;
      int nc = (page[3] << 8) | page[4];
      for (int c = 0; c < nc && c < MAX_CELLS; c++) {
        uint16_t off = (page[8 + 2 * c] << 8) | page[8 + 2 * c + 1];
        int vlen1;
        uint64_t pay = read_varint_auto(page + off, &vlen1);
        int vlen2;
        uint64_t rowid_val = read_varint_auto(page + off + vlen1, &vlen2);
        RecordTask t;
        t.row_id = ((uint64_t)pg << 32) | c;
        t.rowid_val = rowid_val;
        t.page = pg;
        t.offset = (uint16_t)(off + vlen1 + vlen2);
        t.length = (uint16_t)pay;
        tasks[pi * MAX_CELLS + c] = t;
      }
      n_tasks += nc;
    }
    fprintf(stderr, "[+] CPU parsed %d tasks\n", n_tasks);
  }

  uint32_t *lens = (uint32_t *)malloc(n_tasks * sizeof(uint32_t));
  for (int i = 0; i < n_tasks; i++)
    lens[i] = (uint32_t)(tasks[i].length + 128);
  uint64_t *pref = prefix_sum(lens, n_tasks);
  uint64_t total_bytes = pref[n_tasks];
  fprintf(stderr, "[+] writing %lu bytes output\n", total_bytes);

  int fd_out;
  char *outbuf = (char *)mmap_outfile(outp, total_bytes, &fd_out);
  cpu_write_csv(db, page_sz, tasks, n_tasks, outbuf, pref);
  msync(outbuf, total_bytes, MS_SYNC);
  munmap(outbuf, total_bytes);
  close(fd_out);
  munmap(db, db_sz);
  free(tasks);
  free(lens);
  free(pref);
  free(leafs.pages);
  fprintf(stderr, "[+] done.\n");
  return 0;
}
