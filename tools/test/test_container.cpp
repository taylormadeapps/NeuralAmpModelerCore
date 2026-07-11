#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/container.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"

namespace test_container
{

class CountingDSP : public nam::DSP
{
public:
  struct State : public nam::ChannelState
  {
  };

  explicit CountingDSP(NAM_SAMPLE process_value)
  : DSP(1, 1, 48000.0)
  , process_value(process_value)
  {
  }

  void Reset(const double sampleRate, const int maxBufferSize) override
  {
    if (on_reset)
      on_reset();

    reset_count++;
    reset_sample_rate = sampleRate;
    reset_buffer_size = maxBufferSize;
    DSP::Reset(sampleRate, maxBufferSize);
  }

  void prewarm() override { prewarm_count++; }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override
  {
    (void)input;
    process_count++;
    for (int i = 0; i < num_frames; i++)
      output[0][i] = process_value;
  }

  std::unique_ptr<nam::ChannelState> createChannelState() const override
  {
    channel_state_create_count++;
    return std::make_unique<State>();
  }

  void processChannel(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames,
                      nam::ChannelState& state) override
  {
    (void)state;
    process(input, output, num_frames);
  }

  void prewarmChannelState(nam::ChannelState& state) override
  {
    (void)state;
    channel_state_prewarm_count++;
  }

  const NAM_SAMPLE process_value;
  int reset_count = 0;
  int prewarm_count = 0;
  int process_count = 0;
  mutable int channel_state_create_count = 0;
  int channel_state_prewarm_count = 0;
  double reset_sample_rate = 0.0;
  int reset_buffer_size = 0;
  std::function<void()> on_reset;
};

std::unique_ptr<nam::container::ContainerModel> build_counting_container(CountingDSP*& small, CountingDSP*& large)
{
  auto small_model = std::make_unique<CountingDSP>((NAM_SAMPLE)1.0);
  auto large_model = std::make_unique<CountingDSP>((NAM_SAMPLE)2.0);

  small = small_model.get();
  large = large_model.get();

  std::vector<nam::container::Submodel> submodels;
  submodels.push_back({0.5, std::move(small_model)});
  submodels.push_back({1.0, std::move(large_model)});

  return std::make_unique<nam::container::ContainerModel>(std::move(submodels), 48000.0);
}

NAM_SAMPLE process_one_sample(nam::DSP* dsp)
{
  NAM_SAMPLE input = (NAM_SAMPLE)0.0;
  NAM_SAMPLE output = (NAM_SAMPLE)-1.0;
  NAM_SAMPLE* in_ptr = &input;
  NAM_SAMPLE* out_ptr = &output;
  dsp->process(&in_ptr, &out_ptr, 1);
  return output;
}

std::vector<NAM_SAMPLE> process_blocks(nam::DSP& dsp, nam::ChannelState* state,
                                       const std::vector<NAM_SAMPLE>& input,
                                       const std::vector<int>& block_sizes)
{
  std::vector<NAM_SAMPLE> output(input.size(), (NAM_SAMPLE)0.0);
  int position = 0;
  int block_index = 0;
  while (position < (int)input.size())
  {
    const int requested = block_sizes[(size_t)(block_index % (int)block_sizes.size())];
    const int num_frames = std::min(requested, (int)input.size() - position);
    NAM_SAMPLE* input_ptr = const_cast<NAM_SAMPLE*>(input.data() + position);
    NAM_SAMPLE* output_ptr = output.data() + position;
    if (state == nullptr)
      dsp.process(&input_ptr, &output_ptr, num_frames);
    else
      dsp.processChannel(&input_ptr, &output_ptr, num_frames, *state);
    position += num_frames;
    block_index++;
  }
  return output;
}

// Helper: load a .nam file as JSON
nlohmann::json load_nam_json(const std::string& path)
{
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open " + path);
  nlohmann::json j;
  f >> j;
  return j;
}

// Build a SlimmableContainer JSON from 3 .nam files
nlohmann::json build_container_json(const std::string& small_path, const std::string& medium_path,
                                    const std::string& large_path)
{
  nlohmann::json small_model = load_nam_json(small_path);
  nlohmann::json medium_model = load_nam_json(medium_path);
  nlohmann::json large_model = load_nam_json(large_path);

  nlohmann::json container;
  container["version"] = "0.7.0";
  container["architecture"] = "SlimmableContainer";
  container["config"]["submodels"] = nlohmann::json::array({{{"max_value", 0.33}, {"model", small_model}},
                                                            {{"max_value", 0.66}, {"model", medium_model}},
                                                            {{"max_value", 1.0}, {"model", large_model}}});
  container["weights"] = nlohmann::json::array();
  container["sample_rate"] = 48000;
  return container;
}

// Helper to process audio through a DSP model and verify finite output
void process_and_verify(nam::DSP* dsp, int num_buffers, int buffer_size)
{
  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  for (int buf = 0; buf < num_buffers; buf++)
  {
    for (int i = 0; i < buffer_size; i++)
      input[i] = (NAM_SAMPLE)(0.1 * ((buf * buffer_size + i) % 100) / 100.0);

    dsp->process(&in_ptr, &out_ptr, buffer_size);

    for (int i = 0; i < buffer_size; i++)
      assert(std::isfinite(output[i]));
  }
}

// =====================================================================
// Tests
// =====================================================================

void test_container_loads_from_json()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  assert(dsp != nullptr);
}

void test_container_processes_audio()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  process_and_verify(dsp.get(), 3, 64);
}

void test_container_slimmable_selects_submodel()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_small(buffer_size);
  std::vector<NAM_SAMPLE> out_large(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // Process at minimum size (selects first submodel)
  slimmable->SetSlimmableSize(0.0);
  out_ptr = out_small.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Process at maximum size (selects last submodel)
  slimmable->SetSlimmableSize(1.0);
  out_ptr = out_large.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // The outputs should differ since different models are active
  // (Not guaranteed for every sample, but statistically they should differ)
  bool any_different = false;
  for (int i = 0; i < buffer_size; i++)
  {
    if (std::abs(out_small[i] - out_large[i]) > 1e-6)
    {
      any_different = true;
      break;
    }
  }
  assert(any_different);
}

void test_container_boundary_values()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 16;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.05);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // Test exact boundary values (max_value is exclusive: val < max_value selects that submodel)
  slimmable->SetSlimmableSize(0.32); // Should select first submodel (0.32 < 0.33)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.33); // Should select second submodel (0.33 is NOT < 0.33, but IS < 0.66)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.65); // Should select second submodel (0.65 < 0.66)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.66); // Should select third/last submodel (0.66 is NOT < 0.66, fallback)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(1.0); // Should select third/last submodel (fallback)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));
}

void test_container_slimmable_breakpoints()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  const auto breakpoints = slimmable->GetSlimmableSizeBreakpoints();
  assert(breakpoints.size() == 1);
  assert(breakpoints[0] == 0.5);
}

void test_container_empty_submodels_throws()
{
  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] = nlohmann::json::array();
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_container_last_max_value_must_cover_one()
{
  // Build a container where last max_value < 1.0
  auto small_json = load_nam_json("example_models/lstm.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] = nlohmann::json::array({{{"max_value", 0.5}, {"model", small_json}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  // Suppress the version warning
  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);

  std::cerr.rdbuf(originalCerr);
}

void test_container_unsorted_submodels_throws()
{
  auto small_json = load_nam_json("example_models/lstm.nam");
  auto medium_json = load_nam_json("example_models/wavenet.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] =
    nlohmann::json::array({{{"max_value", 0.8}, {"model", small_json}}, {{"max_value", 0.5}, {"model", medium_json}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);

  std::cerr.rdbuf(originalCerr);
}

void test_container_sample_rate_mismatch_throws()
{
  // Create two models with different sample rates
  auto model_48k = load_nam_json("example_models/lstm.nam");
  auto model_44k = load_nam_json("example_models/lstm.nam");
  model_44k["sample_rate"] = 44100;

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] =
    nlohmann::json::array({{{"max_value", 0.5}, {"model", model_44k}}, {{"max_value", 1.0}, {"model", model_48k}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);

  std::cerr.rdbuf(originalCerr);
}

void test_container_load_from_file()
{
  std::filesystem::path path("example_models/slimmable_container.nam");
  auto dsp = nam::get_dsp(path);
  assert(dsp != nullptr);
  process_and_verify(dsp.get(), 3, 64);
}

void test_container_default_is_max_size()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_default(buffer_size);
  std::vector<NAM_SAMPLE> out_max(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  // Ensure both predictions start from identical model state.
  dsp->Reset(sample_rate, buffer_size);
  // Process with default (should be max size)
  out_ptr = out_default.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Explicitly set to max
  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);
  slimmable->SetSlimmableSize(1.0);
  dsp->Reset(sample_rate, buffer_size);
  out_ptr = out_max.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Both should produce the same output (same model active)
  for (int i = 0; i < buffer_size; i++)
    assert(std::abs(out_default[i] - out_max[i]) < 1e-6);
}

void test_container_reset_prepares_every_submodel_and_prewarms_only_active()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);

  dsp->Reset(44100.0, 128);

  assert(small->reset_count == 1);
  assert(small->prewarm_count == 0);
  assert(small->reset_sample_rate == 44100.0);
  assert(small->reset_buffer_size == 128);

  assert(large->reset_count == 1);
  assert(large->prewarm_count == 1);
  assert(large->reset_sample_rate == 44100.0);
  assert(large->reset_buffer_size == 128);
}

void test_container_external_state_is_bound_to_its_submodel()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);
  dsp->Reset(48000.0, 64);

  auto large_state = dsp->createChannelState();
  assert(large_state != nullptr);
  dsp->prewarmChannelState(*large_state);
  assert(small->channel_state_create_count == 0);
  assert(large->channel_state_create_count == 1);
  assert(small->channel_state_prewarm_count == 0);
  assert(large->channel_state_prewarm_count == 1);

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);
  slimmable->SetSlimmableSize(0.0);

  auto small_state = dsp->createChannelState();
  assert(small_state != nullptr);
  dsp->prewarmChannelState(*small_state);
  assert(small->channel_state_create_count == 1);
  assert(large->channel_state_create_count == 1);
  assert(small->channel_state_prewarm_count == 1);
  assert(large->channel_state_prewarm_count == 1);

  NAM_SAMPLE input = (NAM_SAMPLE)0.0;
  NAM_SAMPLE output = (NAM_SAMPLE)-1.0;
  NAM_SAMPLE* input_ptr = &input;
  NAM_SAMPLE* output_ptr = &output;

  // The large state remains attached to the large model even though the
  // container's standalone active tier is now small.
  dsp->processChannel(&input_ptr, &output_ptr, 1, *large_state);
  assert(output == large->process_value);

  output = (NAM_SAMPLE)-1.0;
  dsp->processChannel(&input_ptr, &output_ptr, 1, *small_state);
  assert(output == small->process_value);
}

void test_container_external_state_matches_standalone_at_every_a2_tier()
{
  constexpr int max_buffer_size = 256;
  constexpr int total_frames = 2048;
  std::vector<NAM_SAMPLE> input(total_frames);
  for (int i = 0; i < total_frames; ++i)
    input[(size_t)i] = (NAM_SAMPLE)(0.2 * std::sin(2.0 * M_PI * 317.0 * i / 48000.0)
                                    + 0.07 * std::sin(2.0 * M_PI * 1291.0 * i / 48000.0));

  const std::vector<int> block_sizes = {7, 64, 3, 128, 255, 11};
  for (const double slimmable_size : {0.0, 1.0})
  {
    auto standalone = nam::get_dsp(std::filesystem::path("example_models/A2.nam"));
    auto shared = nam::get_dsp(std::filesystem::path("example_models/A2.nam"));
    auto* standalone_slimmable = dynamic_cast<nam::SlimmableModel*>(standalone.get());
    auto* shared_slimmable = dynamic_cast<nam::SlimmableModel*>(shared.get());
    assert(standalone_slimmable != nullptr);
    assert(shared_slimmable != nullptr);

    standalone_slimmable->SetSlimmableSize(slimmable_size);
    shared_slimmable->SetSlimmableSize(slimmable_size);
    standalone->Reset(48000.0, max_buffer_size);
    shared->Reset(48000.0, max_buffer_size);

    auto state = shared->createChannelState();
    assert(state != nullptr);
    shared->prewarmChannelState(*state);

    const auto standalone_output = process_blocks(*standalone, nullptr, input, block_sizes);
    const auto shared_output = process_blocks(*shared, state.get(), input, block_sizes);
    assert(standalone_output.size() == shared_output.size());
    for (size_t i = 0; i < standalone_output.size(); ++i)
      assert(std::abs(standalone_output[i] - shared_output[i]) < 1e-6);
  }
}

void test_container_switch_resets_before_activation()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);
  dsp->Reset(48000.0, 64);

  NAM_SAMPLE value_during_reset = (NAM_SAMPLE)-1.0;
  small->on_reset = [&]() { value_during_reset = process_one_sample(dsp.get()); };

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);
  slimmable->SetSlimmableSize(0.0);

  assert(small->reset_count == 2);
  assert(small->prewarm_count == 1);
  assert(small->reset_sample_rate == 48000.0);
  assert(small->reset_buffer_size == 64);
  assert(value_during_reset == large->process_value);
  assert(process_one_sample(dsp.get()) == small->process_value);
}

} // namespace test_container
