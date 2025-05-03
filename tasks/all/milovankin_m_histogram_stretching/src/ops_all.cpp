#include "../include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_reduce.h"

namespace milovankin_m_histogram_stretching_all {

bool TestTaskAll::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->inputs_count.empty() && task_data->inputs_count[0] != 0 &&
         !task_data->outputs.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskAll::PreProcessingImpl() {
  const uint8_t* input_data = task_data->inputs.front();
  const uint32_t input_size = task_data->inputs_count.front();

  img_.assign(input_data, input_data + input_size);
  return true;
}

struct MinMaxPair {
  uint8_t min_val;
  uint8_t max_val;

  MinMaxPair() : min_val(std::numeric_limits<uint8_t>::max()), max_val(0) {}

  MinMaxPair(uint8_t min, uint8_t max) : min_val(min), max_val(max) {}
};

bool TestTaskAll::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  // Each process finds local min/max of its portion of the data
  std::size_t chunk_size = img_.size() / size;
  std::size_t start_idx = rank * chunk_size;
  std::size_t end_idx = (rank == size - 1) ? img_.size() : (rank + 1) * chunk_size;

  // Image size is smaller than number of processes
  if (img_.size() < static_cast<std::size_t>(size)) {
    if (rank == 0) {
      chunk_size = img_.size();
      end_idx = img_.size();
    } else {
      chunk_size = 0;
      start_idx = end_idx = 0;
    }
  }

  // Skip if process has no data
  if (start_idx >= img_.size() || chunk_size == 0) {
    MinMaxPair dummy(std::numeric_limits<uint8_t>::max(), 0);
    MinMaxPair global_minmax;

    // Still participate in reduction
    boost::mpi::all_reduce(world_, dummy, global_minmax, [](const MinMaxPair& a, const MinMaxPair& b) -> MinMaxPair {
      return {std::min(a.min_val, b.min_val), std::max(a.max_val, b.max_val)};
    });
    return true;
  }

  // Use TBB to find local min/max in parallel within this process
  const std::vector<uint8_t>& img_ref = img_;
  const std::size_t grain_size = std::max(std::size_t(1024), chunk_size / 16);

  MinMaxPair local_minmax = tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(start_idx, end_idx, grain_size), MinMaxPair(),
      [&img_ref](const tbb::blocked_range<std::size_t>& range, MinMaxPair init) -> MinMaxPair {
        uint8_t local_min = init.min_val;
        uint8_t local_max = init.max_val;

        for (std::size_t i = range.begin(); i != range.end(); ++i) {
          uint8_t val = img_ref[i];
          local_min = std::min(val, local_min);
          local_max = std::max(val, local_max);
        }

        return {local_min, local_max};
      },
      [](MinMaxPair a, MinMaxPair b) -> MinMaxPair {
        return {std::min(a.min_val, b.min_val), std::max(a.max_val, b.max_val)};
      });

  // Find global min/max across all processes
  MinMaxPair global_minmax;
  boost::mpi::all_reduce(world_, local_minmax, global_minmax,
                         [](const MinMaxPair& a, const MinMaxPair& b) -> MinMaxPair {
                           return {std::min(a.min_val, b.min_val), std::max(a.max_val, b.max_val)};
                         });

  // Apply stretching in parallel
  uint8_t min_val = global_minmax.min_val;
  uint8_t max_val = global_minmax.max_val;

  if (min_val != max_val) {
    const int delta = max_val - min_val;
    std::vector<uint8_t>& img_ref_mut = img_;

    tbb::parallel_for(tbb::blocked_range<std::size_t>(start_idx, end_idx, grain_size),
                      [min_val, delta, &img_ref_mut](const tbb::blocked_range<std::size_t>& range) {
                        for (std::size_t i = range.begin(); i != range.end(); ++i) {
                          img_ref_mut[i] = static_cast<uint8_t>(((img_ref_mut[i] - min_val) * 255 + delta / 2) / delta);
                        }
                      });
  }

  if (size > 1) {
    if (rank == 0) {
      // Process 0 already has its chunk processed in place
      // Receive chunks from other processes
      for (int src_rank = 1; src_rank < size; ++src_rank) {
        std::size_t src_start = src_rank * chunk_size;
        std::size_t src_end = (src_rank == size - 1) ? img_.size() : (src_rank + 1) * chunk_size;
        std::size_t src_size = src_end - src_start;

        if (src_size > 0) {
          std::vector<uint8_t> temp_buffer(src_size);
          world_.recv(src_rank, 0, temp_buffer.data(), src_size);
          std::copy(temp_buffer.begin(), temp_buffer.end(), img_.begin() + src_start);
        }
      }
    } else {
      // Send the processed chunk to process 0
      if (chunk_size > 0) {
        world_.send(0, 0, img_.data() + start_idx, chunk_size);
      }
    }
  }

  return true;
}

bool TestTaskAll::PostProcessingImpl() {
  uint8_t* output_data = task_data->outputs[0];
  const uint32_t output_size = task_data->outputs_count[0];
  const uint32_t copy_size = std::min(output_size, static_cast<uint32_t>(img_.size()));

  if (world_.rank() == 0) {
    std::copy_n(img_.cbegin(), copy_size, output_data);
  }

  return true;
}

}  // namespace milovankin_m_histogram_stretching_all