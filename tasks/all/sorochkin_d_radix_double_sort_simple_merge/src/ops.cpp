#include "../include/ops.hpp"

#include <algorithm>
#include <array>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"

namespace {
template <typename T>
constexpr size_t Bytes() {
  return sizeof(T);
}
template <typename T>
constexpr size_t Bits() {
  return Bytes<T>() * CHAR_BIT;
}
class Bitutil {
 private:
  union du64 {
    double d;
    uint64_t u;
    static constexpr uint64_t kMask = 1ULL << ((sizeof(uint64_t) * CHAR_BIT) - 1);
  };

 public:
  static constexpr uint64_t AsU64(double x) {
    const du64 r{.d = x};
    return ((r.u & du64::kMask) != 0U) ? ~r.u : r.u | du64::kMask;
  }

  template <typename T>
    requires std::is_floating_point_v<T> or std::is_integral_v<T>
  static constexpr uint8_t ByteAt(const T &val, uint8_t idx) {
    return (val >> (idx * 8)) & 0xFF;
  }
};

void RadixSort(std::span<double> v) {
  constexpr size_t kBase = 1 << CHAR_BIT;

  std::vector<double> aux_buf(v.size());
  std::span<double> aux{aux_buf};

  std::array<std::size_t, kBase> count;

  for (std::size_t ib = 0; ib < Bytes<double>(); ++ib) {
    std::ranges::fill(count, 0);
    std::ranges::for_each(v, [&](auto el) { ++count[Bitutil::ByteAt(Bitutil::AsU64(el), ib)]; });
    std::partial_sum(count.begin(), count.end(), count.begin());
    std::ranges::for_each(std::ranges::reverse_view(v),
                          [&](auto el) { aux[--count[Bitutil::ByteAt(Bitutil::AsU64(el), ib)]] = el; });
    std::swap(v, aux);
  }
}
}  // namespace

bool sorochkin_d_radix_double_sort_simple_merge_all::SortTask::ValidationImpl() {
  return world_.rank() != 0 || (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool sorochkin_d_radix_double_sort_simple_merge_all::SortTask::PreProcessingImpl() {
  if (world_.rank() == 0) {
    std::span<double> src = {reinterpret_cast<double *>(task_data->inputs[0]), task_data->inputs_count[0]};
    input_.assign(src.begin(), src.end());
    output_.reserve(input_.size());
  }
  return true;
}

namespace {
std::vector<std::span<double>> Distribute(std::span<double> arr, std::size_t n) {
  std::vector<std::span<double>> chunks(n);
  const std::size_t delta = arr.size() / n;
  const std::size_t extra = arr.size() % n;

  auto *cur = arr.data();
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t sz = delta + ((i < extra) ? 1 : 0);
    chunks[i] = std::span{cur, cur + sz};
    cur += sz;
  }

  return chunks;
}
}  // namespace

void sorochkin_d_radix_double_sort_simple_merge_all::SortTask::Squash(boost::mpi::communicator &group) {
  const auto numprocs = static_cast<std::size_t>(group.size());
  for (std::size_t i = 1; i < numprocs; i *= 2) {
    if (group.rank() % (2 * i) == 0) {
      const int slave = group.rank() + static_cast<int>(i);
      if (slave < static_cast<int>(numprocs)) {
        int size{};
        group.recv(int(slave), 0, size);

        const std::size_t threshold = procchunk_.size();
        procchunk_.resize(threshold + size);
        group.recv(int(slave), 0, procchunk_.data() + threshold, size);

        std::ranges::inplace_merge(procchunk_, procchunk_.begin() + std::int64_t(threshold));
      }
    } else if ((group.rank() % i) == 0) {
      const int size = static_cast<int>(procchunk_.size());
      const int master = group.rank() - static_cast<int>(i);
      group.send(master, 0, size);
      group.send(master, 0, procchunk_.data(), size);
      break;
    }
  }
}

bool sorochkin_d_radix_double_sort_simple_merge_all::SortTask::RunImpl() {
  std::size_t totalsize{};
  if (world_.rank() == 0) {
    totalsize = input_.size();
  }
  boost::mpi::broadcast(world_, totalsize, 0);

  if (totalsize == 0) {
    return true;
  }

  const auto numprocs = std::min<std::size_t>(totalsize, world_.size());
  procchunk_.resize(totalsize);

  if (world_.rank() >= int(numprocs)) {
    world_.split(1);
    return true;
  }
  auto group = world_.split(0);

  if (group.rank() == 0) {
    std::vector<std::span<double>> procchunks = Distribute(input_, numprocs);
    procchunk_.assign(procchunks[0].begin(), procchunks[0].end());
    for (int i = 1; i < int(procchunks.size()); i++) {
      const auto &chunk = procchunks[i];
      const int chunksize = int(chunk.size());
      group.send(i, 0, chunksize);
      group.send(i, 0, chunk.data(), chunksize);
    }
  } else {
    int chunksize{};
    group.recv(0, 0, chunksize);
    procchunk_.resize(chunksize);
    group.recv(0, 0, procchunk_.data(), chunksize);
  }

  const auto numthreads = std::min<std::size_t>(procchunk_.size(), ppc::util::GetPPCNumThreads());
  std::vector<std::span<double>> chunks = Distribute(procchunk_, numthreads);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(numthreads); i++) {
    RadixSort(chunks[i]);
  }

  for (std::size_t i = 1; i < numthreads; i *= 2) {
    const auto multithreaded = chunks.front().size() > 48;
    const auto active_threads = numthreads - i;

#pragma omp parallel for if (multithreaded)
    for (int j = 0; j < static_cast<int>(active_threads); j += 2 * static_cast<int>(i)) {
      auto &left = chunks[j];
      auto &right = chunks[j + i];

      std::inplace_merge(left.begin(), left.end(), right.end());
      left = std::span{left.begin(), right.end()};
    }
  }

  Squash(group);

  return true;
}

bool sorochkin_d_radix_double_sort_simple_merge_all::SortTask::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(procchunk_, reinterpret_cast<double *>(task_data->outputs[0]));
  }
  return true;
}
