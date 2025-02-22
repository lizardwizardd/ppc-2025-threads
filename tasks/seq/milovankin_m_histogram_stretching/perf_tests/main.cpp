#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/milovankin_m_histogram_stretching/include/ops_seq.hpp"

static milovankin_m_histogram_stretching_seq::TestTaskSequential createTask(std::vector<uint8_t>& dataIn,
                                                                            std::vector<uint8_t>& dataOut) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(dataIn.data());
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(dataIn.size()));

  taskData->outputs.emplace_back(dataOut.data());
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(dataOut.size()));

  return milovankin_m_histogram_stretching_seq::TestTaskSequential(taskData);
}

TEST(milovankin_m_histogram_stretching_seq, test_pipeline_run) {
  std::vector<uint8_t> dataIn(123456789, 123);
  std::vector<uint8_t> dataOut(dataIn.size());
  dataIn.front() = 5;
  dataIn.back() = 155;

  auto task = std::make_shared<milovankin_m_histogram_stretching_seq::TestTaskSequential>(createTask(dataIn, dataOut));

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(milovankin_m_histogram_stretching_seq, test_task_run) {
  std::vector<uint8_t> dataIn(123456789, 123);
  std::vector<uint8_t> dataOut(dataIn.size());
  dataIn.front() = 5;
  dataIn.back() = 155;

  auto task = std::make_shared < milovankin_m_histogram_stretching_seq::TestTaskSequential>(createTask(dataIn, dataOut));

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
