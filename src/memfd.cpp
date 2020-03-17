//
// Created by Tuowen Zhao on 5/30/19.
//

#include "memfd.h"
#include "brick-mpi.h"
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */

uint8_t *MEMFD::mmap_end = (uint8_t *) 0x600000000000L; // 11 zeros, one leading digit is 2^44 = 16TB
std::set<void *> MEMFD::allocated;

#ifndef USE_MEMFD
int MEMFD::shm_cnt = 0;
std::string MEMFD::shm_prefix = "noname";
#endif

void MEMFD::setup_prefix(const std::string &prefix, int rank) {
#ifndef USE_MEMFD
  shm_prefix = prefix + "." + std::to_string(rank) + ".";
#endif
}

void MEMFD::free(void *ptr, size_t length) {
  auto it_st = allocated.find(ptr);
  if (it_st == allocated.end())
    return;
  auto it_ed = allocated.find(ptr);
  while (*it_ed < (char *) ptr + length)
    it_ed++;
  allocated.erase(it_st, it_ed);
}

MEMFD::MEMFD(size_t length) : offset(0), len(length) {
#ifdef USE_MEMFD
  ring_fd = memfd_create("h", MFD_CLOEXEC);
#else
  shm_name = shm_prefix + std::to_string(shm_cnt++);
  ring_fd = shm_open(shm_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
#endif

  ftruncate(ring_fd, length);
  pagesize = sysconf(_SC_PAGESIZE);
}

void MEMFD::cleanup() {
  close(ring_fd);
#ifndef USE_MEMFD
  shm_unlink(shm_name.c_str());
#endif
}

void *MEMFD::map_pointer(void *hint, size_t fpos, size_t mlen) {
  bool update = (hint == nullptr);
  if (update)
    hint = (uint8_t *) mmap_end - mlen;
  void *res = mmap(hint, mlen, PROT_READ | PROT_WRITE, MAP_SHARED, ring_fd, fpos + offset);
  if (res == MAP_FAILED) printf("map to %p failed with error: %s\n", hint, strerror(errno));
  else if (res != hint) printf("hint failed!\n");
  allocated.insert(res);
  if (update)
    mmap_end = (uint8_t *) res;
  return res;
}

void *MEMFD::packed_pointer(const std::vector<size_t> &packed) {
  void *res = mmap_end;
  std::vector<size_t> packed_red;
  if (packed.size() > 1) {
    packed_red.push_back(packed[0]);
    packed_red.push_back(packed[1]);
    int ll = 0;
    for (int i = 2; i < packed.size(); i += 2) {
      if (packed_red[ll] + packed_red[ll + 1] == packed[i])
        packed_red[ll + 1] += packed[i + 1];
      else {
        packed_red.push_back(packed[i]);
        packed_red.push_back(packed[i + 1]);
        ll += 2;
      }
    }
  }
  for (int i = packed_red.size() - 2; i >= 0; i -= 2) {
    {
      size_t end = packed_red[i] + packed_red[i + 1];
      // Auto resizing doesn't work
      if (end > len)
        printf("specified file chunk reaches after the end of file");
      if (packed_red[i] % pagesize != 0 || packed_red[i + 1] % pagesize != 0)
        printf("Chunks must be page-aligned, %lu %lu\n", packed_red[i], packed_red[i + 1]);
    }
    res = map_pointer(nullptr, packed_red[i], packed_red[i + 1]);
  }
  return res;
}

MEMFD *MEMFD::duplicate(size_t off) {
  auto memfd = new MEMFD(this);
  memfd->offset = off;
  return memfd;
}

#ifndef DECOMP_PAGEUNALIGN

BrickStorage BrickStorage::mmap_alloc(long chunks, long step) {
  BrickStorage b;
  size_t size = chunks * step * sizeof(bElem);
  auto memfd = new MEMFD(size);
  b.chunks = chunks;
  b.step = step;
  // Brick compute use the canonical view
  b.dat = (bElem *) memfd->packed_pointer({0, size});
  b.mmap_info = memfd;
  return b;
}

BrickStorage BrickStorage::mmap_alloc(long chunks, long step, void *mmap_fd, size_t offset) {
  BrickStorage b;
  size_t size = chunks * step * sizeof(bElem);
  auto memfd = static_cast<MEMFD *>(mmap_fd)->duplicate(offset);
  b.chunks = chunks;
  b.step = step;
  b.dat = (bElem *) memfd->packed_pointer({0, size});
  b.mmap_info = memfd;
  return b;
}

#endif
