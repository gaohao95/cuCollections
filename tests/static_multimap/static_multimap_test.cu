/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/discard_iterator.h>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cuco/static_multimap.cuh>
#include <limits>

namespace {
namespace cg = cooperative_groups;

// Thrust logical algorithms (any_of/all_of/none_of) don't work with device
// lambdas: See https://github.com/thrust/thrust/issues/1062
template <typename Iterator, typename Predicate>
bool all_of(Iterator begin, Iterator end, Predicate p)
{
  auto size = thrust::distance(begin, end);
  return size == thrust::count_if(begin, end, p);
}

template <typename Iterator, typename Predicate>
bool any_of(Iterator begin, Iterator end, Predicate p)
{
  return thrust::count_if(begin, end, p) > 0;
}

template <typename Iterator, typename Predicate>
bool none_of(Iterator begin, Iterator end, Predicate p)
{
  return not all_of(begin, end, p);
}

template <typename Key, typename Value>
struct pair_equal {
  __host__ __device__ bool operator()(const cuco::pair_type<Key, Value>& lhs,
                                      const cuco::pair_type<Key, Value>& rhs) const
  {
    return lhs.first == rhs.first;
  }
};

}  // namespace

enum class dist_type { UNIQUE, DUAL, UNIFORM, GAUSSIAN };

template <dist_type Dist, typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end)
{
  auto num_items = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  switch (Dist) {
    case dist_type::UNIQUE: {
      for (auto i = 0; i < num_items; ++i) {
        output_begin[i] = i;
      }
      break;
    }
    case dist_type::DUAL: {
      for (auto i = 0; i < num_items; ++i) {
        output_begin[i] = i % (num_items / 2);
      }
      break;
    }
  }
}

TEMPLATE_TEST_CASE_SIG("Tests of insert_if",
                       "",
                       ((typename Key, typename Value, dist_type Dist), Key, Value, Dist),
                       (int32_t, int32_t, dist_type::UNIQUE))
{
  constexpr std::size_t num_keys{10000000};
  cuco::static_multimap<Key, Value> map{20000000, -1, -1};

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    h_pairs[i].first  = h_keys[i];
    h_pairs[i].second = h_keys[i];
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  auto pred_lambda = [] __device__(cuco::pair_type<Key, Value> pair) {
    return pair.first % 2 == 0;
  };
  map.insert_if(d_pairs.begin(), d_pairs.end(), pred_lambda);
  // map.print();

  auto num = map.pair_count(d_pairs.begin(), d_pairs.end(), pair_equal<Key, Value>{});

  auto out1_zip = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
  auto out2_zip = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

  REQUIRE(num * 2 == num_keys);

  auto size =
    map.pair_retrieve(d_pairs.begin(), d_pairs.end(), out1_zip, out2_zip, pair_equal<Key, Value>{});

  REQUIRE(size * 2 == num_keys);
}
