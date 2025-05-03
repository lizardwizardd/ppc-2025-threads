#include "../include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <thread>
#include <vector>

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
  const size_t num_threads = std::thread::hardware_concurrency();
  const std::size_t chunk_size = std::max(std::size_t(1024), img_.size() / (num_threads * 2));
  const std::size_t num_chunks = (img_.size() + chunk_size - 1) / chunk_size;

  // Find min and max in parallel
  std::vector<std::future<MinMaxPair>> minmax_futures;

  for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
    const std::size_t start = chunk * chunk_size;
    const std::size_t end = std::min(start + chunk_size, img_.size());

    minmax_futures.push_back(std::async(std::launch::async, [this, start, end]() {
      MinMaxPair result;
      for (std::size_t i = start; i < end; ++i) {
        uint8_t val = img_[i];
        result.min_val = std::min(val, result.min_val);
        result.max_val = std::max(val, result.max_val);
      }
      return result;
    }));
  }

  // Combine results from all threads
  MinMaxPair global_minmax;
  for (auto& future : minmax_futures) {
    MinMaxPair local_minmax = future.get();
    global_minmax.min_val = std::min(global_minmax.min_val, local_minmax.min_val);
    global_minmax.max_val = std::max(global_minmax.max_val, local_minmax.max_val);
  }

  uint8_t min_val = global_minmax.min_val;
  uint8_t max_val = global_minmax.max_val;

  // Apply stretching in parallel
  if (min_val != max_val) {
    const int delta = max_val - min_val;

    std::vector<std::thread> threads;

    for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
      const std::size_t start = chunk * chunk_size;
      const std::size_t end = std::min(start + chunk_size, img_.size());

      threads.emplace_back([this, start, end, min_val, delta]() {
        for (std::size_t i = start; i < end; ++i) {
          img_[i] = static_cast<uint8_t>(((img_[i] - min_val) * 255 + delta / 2) / delta);
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
