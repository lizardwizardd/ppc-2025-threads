﻿#include "all/ermilova_d_shell_sort_batcher_even_odd_merger/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <vector>

namespace {

std::vector<int> CreateSedgwickSequence(int n) {
  std::vector<int> gaps;
  int k = 0;
  while (true) {
    int gap =
        (k % 2 == 0) ? (9 * (1 << (2 * k))) - (9 * (1 << k)) + 1 : (8 * (1 << k)) - (6 * (1 << ((k + 1) / 2))) + 1;

    if (gap > n / 2) {
      break;
    }

    gaps.push_back(gap);
    k++;
  }

  if (gaps.empty() || gaps.back() != 1) {
    gaps.push_back(1);
  }

  std::ranges::reverse(gaps);
  return gaps;
}

void ShellSort(std::vector<int> &data, size_t start, size_t end) {
  auto partition_size = static_cast<int>(end - start + 1);
  auto gaps = CreateSedgwickSequence(partition_size);

  for (int gap : gaps) {
    for (size_t i = start + gap; i <= end; i++) {
      int temp = data[i];
      size_t j = i;
      while (j >= start + gap && data[j - gap] > temp) {
        data[j] = data[j - gap];
        j -= gap;
      }
      data[j] = temp;
    }
  }
}

void BatcherMerge(std::vector<int> &data, size_t start, size_t mid, size_t end) {
  std::vector<int> left(data.begin() + static_cast<std::ptrdiff_t>(start),
                        data.begin() + static_cast<std::ptrdiff_t>(mid));

  std::vector<int> right(data.begin() + static_cast<std::ptrdiff_t>(mid),
                         data.begin() + static_cast<std::ptrdiff_t>(end));
  size_t left_index = 0;
  size_t right_index = 0;
  size_t data_offset = start;

  size_t left_size = mid - start;
  size_t right_size = end - mid;

  for (size_t i = start; i < end; ++i) {
    if (i % 2 == 0) {
      if (left_index < left_size && (right_index >= right_size || left[left_index] <= right[right_index])) {
        data[data_offset++] = left[left_index++];
      } else {
        data[data_offset++] = right[right_index++];
      }
    } else {
      if (right_index < right_size && (left_index >= left_size || right[right_index] <= left[left_index])) {
        data[data_offset++] = right[right_index++];
      } else {
        data[data_offset++] = left[left_index++];
      }
    }
  }
}

void ParallelShellSortWithBatcherMerge(std::vector<int> &data) {
  size_t elements_count = data.size();
  if (elements_count <= 1) {
    return;
  }

  int threads_count = omp_get_max_threads();
  size_t block_size = (elements_count + threads_count - 1) / threads_count;

#pragma omp parallel
  {
    int thread_number = omp_get_thread_num();

    size_t start_block_index = static_cast<size_t>(thread_number) * block_size;
    size_t end_block_index = std::min(start_block_index + block_size, elements_count) - 1;
    if (start_block_index < elements_count) {
      ShellSort(data, start_block_index, end_block_index);
    }
  }

  for (size_t merge_size = block_size; merge_size < elements_count; merge_size *= 2) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(elements_count); i += static_cast<int>(2 * merge_size)) {
      size_t mid = std::min(i + merge_size, elements_count);
      size_t end = std::min(i + (2 * merge_size), elements_count);
      if (mid < end) {
        BatcherMerge(data, i, mid, end);
      }
    }
  }
}
}  // namespace
bool ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask::PreProcessingImpl() {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    auto input_task_size = task_data->inputs_count[0];
    auto *input_task_data = reinterpret_cast<int *>(task_data->inputs[0]);
    data_ = std::vector(input_task_data, input_task_data + input_task_size);
  }
  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask::ValidationImpl() {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    if (task_data->inputs_count.empty() || task_data->outputs_count.empty() || task_data->inputs_count[0] <= 0 ||
        task_data->inputs_count[0] != task_data->outputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask::RunImpl() {
  boost::mpi::communicator world;

  int rank = world.rank();
  int size = world.size();

  std::vector<int> full_data;
  std::vector<int> send_counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    full_data = data_;
    size_t total_size = full_data.size();
    size_t offset = 0;

    size_t base = total_size / size;
    size_t rem = total_size % size;
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
      send_counts[i] = static_cast<int>(base + (i < rem ? 1 : 0));
      displs[i] = static_cast<int>(offset);
      offset += send_counts[i];
    }

    for (int i = 1; i < size; ++i) {
      std::vector<int> chunk(full_data.begin() + displs[i], full_data.begin() + displs[i] + send_counts[i]);
      world.send(i, 0, chunk);
    }
  }

  std::vector<int> local_chunk;

  if (rank == 0) {
    local_chunk.assign(full_data.begin(), full_data.begin() + send_counts[0]);
  } else {
    world.recv(0, 0, local_chunk);
  }

  ParallelShellSortWithBatcherMerge(local_chunk);

  if (rank != 0) {
    world.send(0, 1, local_chunk);
  }

  if (rank == 0) {
    std::vector<int> result(full_data.size());
    std::ranges::copy(local_chunk, result.begin());

    for (int i = 1; i < size; ++i) {
      std::vector<int> tmp;
      world.recv(i, 1, tmp);
      std::ranges::copy(tmp, result.begin() + displs[i]);
    }

    data_ = result;

    for (int merge_step = 1; merge_step < size; merge_step *= 2) {
      int loop_end = size - merge_step;
#pragma omp parallel for schedule(static)
      for (int i = 0; i < loop_end; i += 2 * merge_step) {
        size_t left_start = displs[i];
        size_t mid = displs[i + merge_step];
        size_t right_end = (i + 2 * merge_step < size) ? displs[i + (2 * merge_step)] : data_.size();
        BatcherMerge(data_, left_start, mid, right_end);
      }
    }
  }

  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask::PostProcessingImpl() {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    auto *output_task_data = reinterpret_cast<int *>(task_data->outputs[0]);
    std::ranges::copy(data_, output_task_data);
  }
  return true;
}
