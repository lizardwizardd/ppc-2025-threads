#include "../include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace milovankin_m_histogram_stretching_stl {

bool TestTaskSTL::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->inputs_count.empty() && task_data->inputs_count[0] != 0 &&
         !task_data->outputs.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskSTL::PreProcessingImpl() {
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

bool TestTaskSTL::RunImpl() {
  auto num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 0) {
    num_threads = 1;
  }
  num_threads = std::min(num_threads, img_.size());  // No more than img_.size()

  const std::size_t chunk_size = (img_.size() + num_threads - 1) / num_threads;

  // Find min and max in parallel
  std::vector<std::future<MinMaxPair>> minmax_futures;

  for (std::size_t i = 0; i < num_threads; ++i) {
    const std::size_t start = i * chunk_size;
    const std::size_t end = std::min(start + chunk_size, img_.size());

    minmax_futures.push_back(std::async(std::launch::async, [this, start, end]() {
      MinMaxPair result;
      for (std::size_t j = start; j < end; ++j) {
        uint8_t val = img_[j];
        result.min_val = std::min(val, result.min_val);
        result.max_val = std::max(val, result.max_val);
      }
      return result;
    }));
  }

  // Combine results
  MinMaxPair global_minmax;
  for (auto& future : minmax_futures) {
    MinMaxPair local = future.get();
    global_minmax.min_val = std::min(global_minmax.min_val, local.min_val);
    global_minmax.max_val = std::max(global_minmax.max_val, local.max_val);
  }

  const uint8_t min_val = global_minmax.min_val;
  const uint8_t max_val = global_minmax.max_val;

  // Apply stretching in parallel
  if (min_val != max_val) {
    const int delta = max_val - min_val;
    std::vector<std::thread> threads;

    for (std::size_t i = 0; i < num_threads; ++i) {
      const std::size_t start = i * chunk_size;
      const std::size_t end = std::min(start + chunk_size, img_.size());

      threads.emplace_back([this, start, end, min_val, delta]() {
        for (std::size_t j = start; j < end; ++j) {
          img_[j] = static_cast<uint8_t>(((img_[j] - min_val) * 255 + delta / 2) / delta);
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  uint8_t* output_data = task_data->outputs[0];
  const uint32_t output_size = task_data->outputs_count[0];
  const uint32_t copy_size = std::min(output_size, static_cast<uint32_t>(img_.size()));

  std::copy_n(img_.cbegin(), copy_size, output_data);
  return true;
}

}  // namespace milovankin_m_histogram_stretching_stl
