// Numerical verification for the A2 fast-path WaveNet:
// for a config that matches the A2 shape, the fast path must produce the same
// output as the generic WaveNet on the same input and weights.
//
// Built only when NAM_ENABLE_A2_FAST is defined at compile time.

#if defined(NAM_ENABLE_A2_FAST)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"

#include "NAM/dsp.h"
#include "NAM/wavenet/a2_fast.h"
#include "NAM/wavenet/model.h"

#include "allocation_tracking.h"

namespace test_a2_fast
{
namespace
{

// Build a JSON config with the A2 shape, parameterized by channel count
// (3 = A2 nano, 8 = A2 standard). Follows the real .nam schema so both the
// strict detector and the generic parser accept it.
nlohmann::json build_a2_config(int channels)
{
  using nlohmann::json;

  json activation = json::array();
  json gating_mode = json::array();
  json secondary = json::array();
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    activation.push_back({{"type", "LeakyReLU"}, {"negative_slope", nam::wavenet::a2_fast::kLeakySlope}});
    gating_mode.push_back("none");
    secondary.push_back(nullptr);
  }

  json kernel_sizes = json::array();
  json dilations = json::array();
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    kernel_sizes.push_back(nam::wavenet::a2_fast::kKernelSizes[i]);
    dilations.push_back(nam::wavenet::a2_fast::kDilations[i]);
  }

  json film_inactive = {{"active", false}, {"shift", true}, {"groups", 1}};

  json layer;
  layer["input_size"] = 1;
  layer["condition_size"] = 1;
  layer["channels"] = channels;
  layer["bottleneck"] = channels;
  layer["kernel_sizes"] = kernel_sizes;
  layer["dilations"] = dilations;
  layer["activation"] = activation;
  layer["gating_mode"] = gating_mode;
  layer["secondary_activation"] = secondary;
  layer["head"] = {{"out_channels", 1}, {"kernel_size", nam::wavenet::a2_fast::kHeadKernelSize}, {"bias", true}};
  layer["head1x1"] = {{"active", false}, {"out_channels", 1}, {"groups", 1}};
  layer["layer1x1"] = {{"active", true}, {"groups", 1}};
  layer["conv_pre_film"] = film_inactive;
  layer["conv_post_film"] = film_inactive;
  layer["input_mixin_pre_film"] = film_inactive;
  layer["input_mixin_post_film"] = film_inactive;
  layer["activation_pre_film"] = film_inactive;
  layer["activation_post_film"] = film_inactive;
  layer["layer1x1_post_film"] = film_inactive;
  layer["head1x1_post_film"] = film_inactive;
  layer["groups_input"] = 1;
  layer["groups_input_mixin"] = 1;

  json config;
  config["layers"] = json::array({layer});
  config["head_scale"] = nam::wavenet::a2_fast::kHeadScale;
  return config;
}

// Weight count for the A2 layer array with the given channel count.
// Must match the order and sizes expected by both the generic WaveNet parser
// and A2FastModel::_load_weights.
int a2_weight_count(int channels)
{
  const int bn = channels;
  int total = /*rechannel*/ channels;
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    const int K = nam::wavenet::a2_fast::kKernelSizes[i];
    total += bn * channels * K + bn; // conv1d weights + bias
    total += bn;                     // input mixin (no bias)
    total += channels * bn + channels; // layer1x1 + bias
  }
  total += channels * nam::wavenet::a2_fast::kHeadKernelSize + 1; // head rechannel + bias
  total += 1; // trailing head_scale (read by WaveNet::set_weights_)
  return total;
}

std::vector<float> make_deterministic_weights(int count, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
  std::vector<float> w(count);
  for (auto& x : w)
    x = dist(rng);
  return w;
}

std::vector<NAM_SAMPLE> make_test_input(int num_frames, double sample_rate)
{
  std::vector<NAM_SAMPLE> in(num_frames);
  // Two-tone signal so the network sees varied content.
  for (int i = 0; i < num_frames; i++)
  {
    const double t = static_cast<double>(i) / sample_rate;
    in[i] = static_cast<NAM_SAMPLE>(0.25 * std::sin(2.0 * M_PI * 220.0 * t)
                                    + 0.10 * std::sin(2.0 * M_PI * 1230.0 * t));
  }
  return in;
}

std::vector<NAM_SAMPLE> run_dsp(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, int block_size)
{
  // Reset also prewarms.
  dsp.Reset(48000.0, block_size);
  std::vector<NAM_SAMPLE> out(input.size(), static_cast<NAM_SAMPLE>(0));
  int pos = 0;
  const int total = static_cast<int>(input.size());
  while (pos < total)
  {
    const int n = std::min(block_size, total - pos);
    const NAM_SAMPLE* in_ptr = input.data() + pos;
    NAM_SAMPLE* out_ptr = out.data() + pos;
    const NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, n);
    pos += n;
  }
  return out;
}

std::vector<NAM_SAMPLE> run_dsp_blocks(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input,
                                       const std::vector<int>& block_sizes)
{
  std::vector<NAM_SAMPLE> out(input.size(), static_cast<NAM_SAMPLE>(0));
  int pos = 0;
  int block_index = 0;
  const int total = static_cast<int>(input.size());
  while (pos < total)
  {
    const int block_size = block_sizes[static_cast<size_t>(block_index % static_cast<int>(block_sizes.size()))];
    const int n = std::min(block_size, total - pos);
    const NAM_SAMPLE* in_ptr = input.data() + pos;
    NAM_SAMPLE* out_ptr = out.data() + pos;
    const NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, n);
    pos += n;
    block_index++;
  }
  return out;
}

std::vector<NAM_SAMPLE> run_dsp_channel_state_blocks(nam::DSP& dsp, nam::ChannelState& state,
                                                     const std::vector<NAM_SAMPLE>& input,
                                                     const std::vector<int>& block_sizes)
{
  std::vector<NAM_SAMPLE> out(input.size(), static_cast<NAM_SAMPLE>(0));
  int pos = 0;
  int block_index = 0;
  const int total = static_cast<int>(input.size());
  while (pos < total)
  {
    const int block_size = block_sizes[static_cast<size_t>(block_index % static_cast<int>(block_sizes.size()))];
    const int n = std::min(block_size, total - pos);
    const NAM_SAMPLE* in_ptr = input.data() + pos;
    NAM_SAMPLE* out_ptr = out.data() + pos;
    const NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.processChannel(const_cast<NAM_SAMPLE**>(in_arr), out_arr, n, state);
    pos += n;
    block_index++;
  }
  return out;
}

void compare(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b, int channels, int block_size,
             double tol)
{
  assert(a.size() == b.size());
  double max_diff = 0.0;
  int max_i = 0;
  for (size_t i = 0; i < a.size(); i++)
  {
    const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > max_diff)
    {
      max_diff = d;
      max_i = static_cast<int>(i);
    }
  }
  if (!(max_diff < tol))
  {
    std::cerr << "A2FastModel<" << channels << "> diverges from generic WaveNet "
              << "(block=" << block_size << "): max |diff| = " << max_diff << " at i=" << max_i
              << " (generic=" << a[max_i] << ", fast=" << b[max_i] << ")" << std::endl;
    assert(false);
  }
}

void compare_channel_state(const std::vector<NAM_SAMPLE>& standalone, const std::vector<NAM_SAMPLE>& shared,
                           int channels, double tol)
{
  assert(standalone.size() == shared.size());
  double max_diff = 0.0;
  int max_i = 0;
  for (size_t i = 0; i < standalone.size(); i++)
  {
    const double d = std::fabs(static_cast<double>(standalone[i]) - static_cast<double>(shared[i]));
    if (d > max_diff)
    {
      max_diff = d;
      max_i = static_cast<int>(i);
    }
  }
  if (!(max_diff < tol))
  {
    std::cerr << "A2FastModel<" << channels << "> external ChannelState diverges from standalone process: max |diff| = "
              << max_diff << " at i=" << max_i << " (standalone=" << standalone[max_i]
              << ", shared=" << shared[max_i] << ")" << std::endl;
    assert(false);
  }
}

} // namespace

void test_detector_matches_nano()
{
  auto cfg = build_a2_config(3);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 3);
}

void test_detector_matches_standard()
{
  auto cfg = build_a2_config(8);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 8);
}

void test_detector_rejects_wrong_channels()
{
  auto cfg = build_a2_config(3);
  cfg["layers"][0]["channels"] = 4;
  cfg["layers"][0]["bottleneck"] = 4;
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_rejects_wrong_kernel_sizes()
{
  auto cfg = build_a2_config(8);
  cfg["layers"][0]["kernel_sizes"][0] = 7; // tweak first entry
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_rejects_wrong_activation()
{
  auto cfg = build_a2_config(8);
  cfg["layers"][0]["activation"][0] = {{"type", "Tanh"}};
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_rejects_gating()
{
  auto cfg = build_a2_config(3);
  cfg["layers"][0]["gating_mode"][0] = "gated";
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_matches_generic(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  // Fast path: build through the explicit A2 factory.
  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  // Generic path: call parse_config_json directly so the dispatcher's A2 shortcut
  // doesn't kick in. This gives us the reference WaveNet against the same config.
  auto generic_cfg = nam::wavenet::parse_config_json(cfg, 48000.0);
  std::vector<float> w_gen = weights;
  auto generic_dsp = generic_cfg.create(std::move(w_gen), 48000.0);

  // Exercise with two block sizes to catch any off-by-one in the ring-buffer rewind.
  const int total = 2048;
  const auto input = make_test_input(total, 48000.0);
  for (int block : {64, 256})
  {
    const auto out_generic = run_dsp(*generic_dsp, input, block);
    const auto out_fast = run_dsp(*fast_dsp, input, block);
    // 1e-5 is what nam2c_verify.py uses as its byte-exactness threshold; allow a little
    // slack because the generic path sums through Eigen and may reorder FMAs.
    compare(out_generic, out_fast, channels, block, /*tol=*/5e-5);
  }
}

void test_matches_generic_nano()
{
  test_matches_generic(3);
}

void test_matches_generic_standard()
{
  test_matches_generic(8);
}

void test_channel_state_matches_standalone(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA250000u + channels);

  auto standalone_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_standalone = weights;
  auto standalone_dsp = standalone_cfg->create(std::move(w_standalone), 48000.0);

  auto shared_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_shared = weights;
  auto shared_dsp = shared_cfg->create(std::move(w_shared), 48000.0);

  const int max_buffer = 256;
  standalone_dsp->Reset(48000.0, max_buffer);
  shared_dsp->Reset(48000.0, max_buffer);

  auto shared_state = shared_dsp->createChannelState();
  assert(shared_state);
  shared_dsp->prewarmChannelState(*shared_state);

  const int total = 4096;
  const auto input = make_test_input(total, 48000.0);
  const std::vector<int> block_sizes = {7, 64, 3, 128, 255, 11};

  const auto out_standalone = run_dsp_blocks(*standalone_dsp, input, block_sizes);
  const auto out_shared = run_dsp_channel_state_blocks(*shared_dsp, *shared_state, input, block_sizes);
  compare_channel_state(out_standalone, out_shared, channels, /*tol=*/1e-6);
}

void test_channel_state_matches_standalone_nano()
{
  test_channel_state_matches_standalone(3);
}

void test_channel_state_matches_standalone_standard()
{
  test_channel_state_matches_standalone(8);
}

// Real-time safety: once the DSP has been Reset (buffers sized, prewarmed),
// subsequent process() calls must not allocate or free heap memory. Uses the
// same allocation-tracking infrastructure as the generic WaveNet RT-safety
// tests (tools/test/allocation_tracking.{h,cpp}) — overridden malloc/free and
// global new/delete increment counters while tracking is enabled.
void test_process_realtime_safe(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  // Exercise several block sizes all within a single pre-sized state so the
  // internal "num_frames > max_buffer_size" guard in process() never fires
  // (which would legitimately reallocate).
  const int max_buffer = 256;
  fast_dsp->Reset(48000.0, max_buffer);

  const int total = 4 * max_buffer;
  const auto input = make_test_input(total, 48000.0);
  std::vector<NAM_SAMPLE> output(total, 0.0);

  // Warm up caches / any lazy init with one untracked pass.
  {
    const NAM_SAMPLE* in = input.data();
    NAM_SAMPLE* out = output.data();
    const NAM_SAMPLE* in_arr[] = {in};
    NAM_SAMPLE* out_arr[] = {out};
    fast_dsp->process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, max_buffer);
  }

  for (int block : {1, 32, 64, 128, 256})
  {
    std::string test_name = "A2FastModel<" + std::to_string(channels) + ">::process block="
                            + std::to_string(block);
    allocation_tracking::run_allocation_test_no_allocations(
      nullptr,
      [&]() {
        int pos = 0;
        while (pos + block <= total)
        {
          const NAM_SAMPLE* in = input.data() + pos;
          NAM_SAMPLE* out = output.data() + pos;
          const NAM_SAMPLE* in_arr[] = {in};
          NAM_SAMPLE* out_arr[] = {out};
          fast_dsp->process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, block);
          pos += block;
        }
      },
      nullptr,
      test_name.c_str());
  }
}

void test_process_realtime_safe_nano()
{
  test_process_realtime_safe(3);
}

void test_process_realtime_safe_standard()
{
  test_process_realtime_safe(8);
}

void test_process_channel_realtime_safe(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  const int max_buffer = 256;
  fast_dsp->Reset(48000.0, max_buffer);
  auto state = fast_dsp->createChannelState();
  assert(state);
  fast_dsp->prewarmChannelState(*state);

  const int total = 4 * max_buffer;
  const auto input = make_test_input(total, 48000.0);
  std::vector<NAM_SAMPLE> output(total, 0.0);

  {
    const NAM_SAMPLE* in = input.data();
    NAM_SAMPLE* out = output.data();
    const NAM_SAMPLE* in_arr[] = {in};
    NAM_SAMPLE* out_arr[] = {out};
    fast_dsp->processChannel(const_cast<NAM_SAMPLE**>(in_arr), out_arr, max_buffer, *state);
  }

  for (int block : {1, 32, 64, 128, 256})
  {
    std::string test_name = "A2FastModel<" + std::to_string(channels) + ">::processChannel block="
                            + std::to_string(block);
    allocation_tracking::run_allocation_test_no_allocations(
      nullptr,
      [&]() {
        int pos = 0;
        while (pos + block <= total)
        {
          const NAM_SAMPLE* in = input.data() + pos;
          NAM_SAMPLE* out = output.data() + pos;
          const NAM_SAMPLE* in_arr[] = {in};
          NAM_SAMPLE* out_arr[] = {out};
          fast_dsp->processChannel(const_cast<NAM_SAMPLE**>(in_arr), out_arr, block, *state);
          pos += block;
        }
      },
      nullptr,
      test_name.c_str());
  }
}

void test_process_channel_realtime_safe_nano()
{
  test_process_channel_realtime_safe(3);
}

void test_process_channel_realtime_safe_standard()
{
  test_process_channel_realtime_safe(8);
}

} // namespace test_a2_fast

#endif // NAM_ENABLE_A2_FAST
