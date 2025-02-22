#include <gtest/gtest.h>
#include <stdint.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
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

TEST(milovankin_m_histogram_stretching_seq, test_small_data) {
  // clang-format off
  std::vector<uint8_t> dataIn = {
	50, 100, 100, 200,
	100, 50, 250, 100,
	100, 100, 200, 50,
	100, 200, 50, 250,
  };
  std::vector<uint8_t> dataExpected = {
	0,   64,  64,  191,
    64,  0,   255, 64,
    64,  64,  191, 0,
    64,  191, 0,   255
  };
  // clang-format on
  std::vector<uint8_t> dataOut(dataIn.size());

  milovankin_m_histogram_stretching_seq::TestTaskSequential task = createTask(dataIn, dataOut);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  ASSERT_EQ(dataOut, dataExpected);
}

TEST(milovankin_m_histogram_stretching_seq, test_single_element) {
  std::vector<uint8_t> dataIn = {150};
  std::vector<uint8_t> dataOut(1);

  auto task = createTask(dataIn, dataOut);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(dataOut[0], 150);
}

TEST(milovankin_m_histogram_stretching_seq, test_empty_data) {
  std::vector<uint8_t> dataIn;
  std::vector<uint8_t> dataOut;

  auto task = createTask(dataIn, dataOut);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_seq, test_validation_fail_different_buffer_sizes) {
  std::vector<uint8_t> dataIn(10, 100);
  std::vector<uint8_t> dataOut(5);  // Output buffer too small

  auto task = createTask(dataIn, dataOut);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_seq, test_validation_fail_output_buffer_empty) {
  std::vector<uint8_t> dataIn(10);
  std::vector<uint8_t> dataOut;

  auto task = createTask(dataIn, dataOut);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_seq, test_compare_with_opencv) {
  cv::Mat image = cv::imread(ppc::util::GetAbsolutePath("seq/milovankin_m_histogram_stretching/data/img_test.jpg"),
                             cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(image.empty());

  cv::Mat expectedImage;
  cv::normalize(image, expectedImage, 0, 255, cv::NORM_MINMAX);

  std::vector<uint8_t> dataIn(image.data, image.data + image.total());
  std::vector<uint8_t> dataOut(dataIn.size());
  std::vector<uint8_t> dataExpected(expectedImage.data, expectedImage.data + expectedImage.total());

  auto task = createTask(dataIn, dataOut);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  ASSERT_EQ(dataOut.size(), dataExpected.size());
  for (size_t i = 0; i < dataOut.size(); ++i) {
    ASSERT_EQ(dataOut[i], dataExpected[i]) << "Mismatch at pixel " << i;
  }
}
