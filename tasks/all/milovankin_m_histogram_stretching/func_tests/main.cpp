#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../include/ops_all.hpp"
#include "boost/mpi/collectives/broadcast.hpp"
#include "core/task/include/task.hpp"

namespace {

milovankin_m_histogram_stretching_all::TestTaskAll CreateParallelTask(std::vector<uint8_t>& data_in,
                                                                      std::vector<uint8_t>& data_out) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(data_in.data());
  task_data->inputs_count.emplace_back(static_cast<uint32_t>(data_in.size()));

  task_data->outputs.emplace_back(data_out.data());
  task_data->outputs_count.emplace_back(static_cast<uint32_t>(data_out.size()));

  return milovankin_m_histogram_stretching_all::TestTaskAll(task_data);
}

}  // namespace

TEST(milovankin_m_histogram_stretching_all, test_small_data) {
  boost::mpi::communicator world;

  // clang-format off
  std::vector<uint8_t> data_in = {
    50, 100, 100, 200,
    100, 50, 250, 100,
    100, 100, 200, 50,
    100, 200, 50, 250,
  };
  std::vector<uint8_t> data_expected = {
    0,   64,  64,  191,
    64,  0,   255, 64,
    64,  64,  191, 0,
    64,  191, 0,   255
  };
  // clang-format on
  std::vector<uint8_t> data_out(data_in.size());

  milovankin_m_histogram_stretching_all::TestTaskAll task = CreateParallelTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_EQ(data_out, data_expected);
  }
}

TEST(milovankin_m_histogram_stretching_all, test_single_element) {
  boost::mpi::communicator world;

  std::vector<uint8_t> data_in = {150};
  std::vector<uint8_t> data_out(1);

  auto task = CreateParallelTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    EXPECT_EQ(data_out[0], 150);
  }
}

TEST(milovankin_m_histogram_stretching_all, test_empty_data) {
  std::vector<uint8_t> data_in;
  std::vector<uint8_t> data_out;

  auto task = CreateParallelTask(data_in, data_out);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_all, test_validation_fail_different_buffer_sizes) {
  std::vector<uint8_t> data_in(10, 100);
  std::vector<uint8_t> data_out(5);

  auto task = CreateParallelTask(data_in, data_out);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_all, test_validation_fail_output_buffer_empty) {
  std::vector<uint8_t> data_in(10);
  std::vector<uint8_t> data_out;

  auto task = CreateParallelTask(data_in, data_out);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_all, test_filled_image) {
  boost::mpi::communicator world;

  // clang-format off
  std::vector<uint8_t> data_in(100, 123);
  std::vector<uint8_t> data_expected(100, 123);
  // clang-format on
  std::vector<uint8_t> data_out(data_in.size());

  milovankin_m_histogram_stretching_all::TestTaskAll task = CreateParallelTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(data_expected, data_out);
  }
}

TEST(milovankin_m_histogram_stretching_all, test_big_image) {
  boost::mpi::communicator world;

  std::vector<uint8_t> data_in(1024, 100);
  std::vector<uint8_t> data_out(data_in.size());

  data_in[0] = 50;
  data_in[1] = 125;

  std::vector<uint8_t> data_expected(data_in.size(), 170);
  data_expected[0] = 0;
  data_expected[1] = 255;

  milovankin_m_histogram_stretching_all::TestTaskAll task = CreateParallelTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(data_expected, data_out);
  }
}

TEST(milovankin_m_histogram_stretching_all, test_mpi_gather_logic) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (size <= 1) {
    // GTEST_SKIP() << "Skipping MPI gather test on single process.";
  }

  const std::size_t data_size = 20;
  std::vector<uint8_t> data_in(data_size);
  std::vector<uint8_t> data_out(data_size);
  std::vector<uint8_t> data_expected(data_size);

  if (rank == 0) {
    std::size_t chunk_size = data_size / size;
    std::size_t remainder = data_size % size;

    for (int r = 0; r < size; ++r) {
      std::size_t r_start_idx = r * chunk_size;
      std::size_t r_chunk_size = (r == size - 1) ? (chunk_size + remainder) : chunk_size;
      std::size_t r_end_idx = r_start_idx + r_chunk_size;

      // Ensure end index does not exceed data size
      r_end_idx = std::min(r_end_idx, data_size);

      // Fill the chunk corresponding to rank 'r' with value 'r'
      for (std::size_t i = r_start_idx; i < r_end_idx; ++i) {
        data_in[i] = static_cast<uint8_t>(r);        // Assign rank number as value
        data_expected[i] = static_cast<uint8_t>(r);  // Expected output is same for this simple case
      }
    }
  }

  boost::mpi::broadcast(world, data_in.data(), static_cast<int>(data_in.size()), 0);

  auto task = CreateParallelTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();  // Trigger the GatherResults logic
  task.PostProcessing();

  if (rank == 0) {
    ASSERT_EQ(data_out, data_expected);
  }
}