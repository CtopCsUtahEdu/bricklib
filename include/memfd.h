/**
 * @file
 * @brief Helper data structure for memory file
 */

#ifndef BRICK_MEMFD_H
#define BRICK_MEMFD_H

#include <cstdint>
#include <initializer_list>
#include <set>
#include <string>
#include <vector>

class MEMFD {
private:
  size_t len;
  int ring_fd;
  long pagesize;
  size_t offset;
#ifndef USE_MEMFD
  static std::string shm_prefix;
  static int shm_cnt;
  std::string shm_name;
#endif
public:
  static uint8_t *mmap_end;
  static std::set<void *> allocated;

  static void setup_prefix(const std::string &prefix, int rank);

  static void free(void *ptr, size_t length);

  MEMFD(MEMFD *memfd)
      : len(memfd->len), ring_fd(memfd->ring_fd), pagesize(memfd->pagesize), offset(0) {
#ifndef USE_MEMFD
    shm_name = memfd->shm_name;
#endif
  }

  MEMFD(size_t length);

  void *map_pointer(void *hint, size_t pos, size_t len);

  void *packed_pointer(const std::vector<size_t> &packed);

  void *packed_pointer(const std::initializer_list<size_t> &packed_init) {
    std::vector<size_t> packed = packed_init;
    return packed_pointer(packed);
  }

  MEMFD *duplicate(size_t offset);

  void cleanup();
};

// TODO Record and free pointers

#endif // BRICK_MEMFD_H
