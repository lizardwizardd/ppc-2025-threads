#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/pikarychev_i_hoare_sort_simple_merge/include/ops_seq.hpp"

namespace {
void PerformTest(int size, bool reverse) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-5000, 5000);

  std::vector<int> in(size);
  std::ranges::generate(in, [&] { return dist(gen); });

  std::vector<int> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&reverse));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto task = pikarychev_i_hoare_sort_simple_merge::HoareSeq<int>(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out, [&](const auto& a, const auto& b) { return reverse ? (a < b) : (a > b); }));
}
}  // namespace

TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_0) { PerformTest(0, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_1) { PerformTest(1, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_2) { PerformTest(2, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_3) { PerformTest(3, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_4) { PerformTest(4, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_5) { PerformTest(5, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_15) { PerformTest(15, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_16) { PerformTest(16, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_17) { PerformTest(17, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_18) { PerformTest(18, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_19) { PerformTest(19, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_20) { PerformTest(20, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_21) { PerformTest(21, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_22) { PerformTest(22, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_23) { PerformTest(23, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_24) { PerformTest(24, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_25) { PerformTest(25, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_35) { PerformTest(35, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_36) { PerformTest(36, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_37) { PerformTest(37, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_38) { PerformTest(38, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_39) { PerformTest(39, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_40) { PerformTest(40, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_41) { PerformTest(41, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_49) { PerformTest(49, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_50) { PerformTest(50, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_51) { PerformTest(51, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_95) { PerformTest(95, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_96) { PerformTest(96, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_97) { PerformTest(97, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_98) { PerformTest(98, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_99) { PerformTest(99, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_100) { PerformTest(100, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_101) { PerformTest(101, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_102) { PerformTest(102, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_106) { PerformTest(106, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_107) { PerformTest(107, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_115) { PerformTest(115, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_116) { PerformTest(116, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_125) { PerformTest(125, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_126) { PerformTest(126, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, standard_127) { PerformTest(127, false); }

TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_0) { PerformTest(0, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_1) { PerformTest(1, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_2) { PerformTest(2, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_3) { PerformTest(3, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_4) { PerformTest(4, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_5) { PerformTest(5, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_15) { PerformTest(15, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_16) { PerformTest(16, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_17) { PerformTest(17, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_18) { PerformTest(18, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_19) { PerformTest(19, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_20) { PerformTest(20, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_21) { PerformTest(21, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_22) { PerformTest(22, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_23) { PerformTest(23, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_24) { PerformTest(24, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_25) { PerformTest(25, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_35) { PerformTest(35, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_36) { PerformTest(36, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_37) { PerformTest(37, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_38) { PerformTest(38, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_39) { PerformTest(39, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_40) { PerformTest(40, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_41) { PerformTest(41, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_49) { PerformTest(49, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_50) { PerformTest(50, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_51) { PerformTest(51, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_95) { PerformTest(95, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_96) { PerformTest(96, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_97) { PerformTest(97, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_98) { PerformTest(98, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_99) { PerformTest(99, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_100) { PerformTest(100, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_101) { PerformTest(101, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_102) { PerformTest(102, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_106) { PerformTest(106, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_107) { PerformTest(107, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_115) { PerformTest(115, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_116) { PerformTest(116, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_125) { PerformTest(125, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_126) { PerformTest(126, true); }
TEST(pikarychev_i_hoare_sort_simple_merge_seq, reverse_127) { PerformTest(127, true); }