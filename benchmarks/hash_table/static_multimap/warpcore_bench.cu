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

#include <random>

#include <thrust/device_vector.h>
#include <nvbench/nvbench.cuh>

#include <key_generator.hpp>
#include <warpcore.cuh>

using namespace warpcore;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  dist_type,
  // Callable to generate input strings:
  // Short identifier used for tables, command-line args, etc.
  // Used when context is available to figure out the enum type.
  [](dist_type d) {
    switch (d) {
      case dist_type::GAUSSIAN: return "GAUSSIAN";
      case dist_type::GEOMETRIC: return "GEOMETRIC";
      case dist_type::UNIFORM: return "UNIFORM";
      default: return "ERROR";
    }
  },
  // Callable to generate descriptions:
  // If non-empty, these are used in `--list` to describe values.
  // Used when context may not be available to figure out the type from the
  // input string.
  // Just use `[](auto) { return std::string{}; }` if you don't want these.
  [](auto) { return std::string{}; })

/**
 * @brief A benchmark evaluating multi-value `insert` performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_warpcore_insert(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  using hash_table_t = MultiValueHashTable<Key,
                                           Value,
                                           defaults::empty_key<Key>(),
                                           defaults::tombstone_key<Key>(),
                                           defaults::probing_scheme_t<Key, 8>,
                                           storage::key_value::AoSStore<Key, Value>>;

  auto const num_keys  = state.get_int64("NumInputs");
  auto const occupancy = state.get_float64("Occupancy");

  std::size_t const capacity = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Key> d_values(h_keys);

  state.add_element_count(num_keys, "NumKeys");

  hash_table_t hash_table(capacity);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      hash_table.init();

      // Use timers to explicitly mark the target region
      timer.start();
      hash_table.insert(d_keys.data().get(), d_values.data().get(), num_keys, launch.get_stream());
      timer.stop();

      auto status = hash_table.pop_status(launch.get_stream());
      if (status.has_any_errors()) { std::cout << status << "\n"; }
    });
}

template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_warpcore_insert(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating multi-value `count` performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_warpcore_count(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  using hash_table_t = MultiValueHashTable<Key,
                                           Value,
                                           defaults::empty_key<Key>(),
                                           defaults::tombstone_key<Key>(),
                                           defaults::probing_scheme_t<Key, 8>,
                                           storage::key_value::AoSStore<Key, Value>>;

  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const capacity = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Key> d_values(h_keys);

  hash_table_t hash_table(capacity);

  hash_table.insert(d_keys.data().get(), d_values.data().get(), num_keys);

  generate_prob_keys<Key>(matching_rate, h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_prob_keys(h_keys);
  thrust::device_vector<std::size_t> d_offsets(num_keys);

  state.add_element_count(num_keys, "NumKeys");

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               std::size_t value_size_out = 0;

               timer.start();
               // nullptr to launch the dry-run count
               hash_table.retrieve(d_prob_keys.data().get(),
                                   num_keys,
                                   d_offsets.data().get(),
                                   d_offsets.data().get() + 1,
                                   nullptr,
                                   value_size_out,
                                   launch.get_stream());
               timer.stop();

               auto status = hash_table.pop_status(launch.get_stream());
               if (status.has_any_errors()) { std::cout << status << "\n"; }
             });
}

template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_warpcore_count(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating multi-value `find_all` performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_warpcore_retrieve(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  using hash_table_t = MultiValueHashTable<Key,
                                           Value,
                                           defaults::empty_key<Key>(),
                                           defaults::tombstone_key<Key>(),
                                           defaults::probing_scheme_t<Key, 8>,
                                           storage::key_value::AoSStore<Key, Value>>;

  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const capacity = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Key> d_values(h_keys);

  hash_table_t hash_table(capacity);

  hash_table.insert(d_keys.data().get(), d_values.data().get(), num_keys);

  generate_prob_keys<Key>(matching_rate, h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_prob_keys(h_keys);

  state.add_element_count(num_keys, "NumKeys");

  std::size_t value_size_out = 0;

  thrust::device_vector<std::size_t> d_offsets(num_keys);

  hash_table.retrieve(d_prob_keys.data().get(),
                      num_keys,
                      d_offsets.data().get(),
                      d_offsets.data().get() + 1,
                      nullptr,
                      value_size_out);
  cudaDeviceSynchronize();

  thrust::device_vector<Value> d_results(value_size_out);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               value_size_out = 0;

               timer.start();
               hash_table.retrieve(d_prob_keys.data().get(),
                                   num_keys,
                                   d_offsets.data().get(),
                                   d_offsets.data().get() + 1,
                                   d_results.data().get(),
                                   value_size_out,
                                   launch.get_stream());
               timer.stop();

               auto status = hash_table.pop_status(launch.get_stream());
               if (status.has_any_errors()) { std::cout << status << "\n"; }
             });
}

template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_warpcore_retrieve(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

using key_type   = nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>;
using value_type = nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>;
using d_type =
  nvbench::enum_type_list<dist_type::GAUSSIAN, dist_type::GEOMETRIC, dist_type::UNIFORM>;

using multiplicity = nvbench::enum_type_list<1, 2, 4, 8, 16, 32, 64, 128, 256>;

NVBENCH_BENCH_TYPES(nvbench_warpcore_insert,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>,
                                      multiplicity))
  .set_name("warpcore_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8});

NVBENCH_BENCH_TYPES(nvbench_warpcore_insert,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("warpcore_insert_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1));

NVBENCH_BENCH_TYPES(nvbench_warpcore_count,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>,
                                      multiplicity))
  .set_name("warpcore_count_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_warpcore_count,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("warpcore_count_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_warpcore_count,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("warpcore_count_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1});

NVBENCH_BENCH_TYPES(nvbench_warpcore_retrieve,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>,
                                      multiplicity))
  .set_name("warpcore_retrieve_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_warpcore_retrieve,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("warpcore_retrieve_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_warpcore_retrieve,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("warpcore_retrieve_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1});
