#include <algorithm>
#include <string>
#include <vector>
#include <memory>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include "registry.h"
#include "lstm.h"

nam::lstm::LSTMCell::LSTMCell(const int input_size, const int hidden_size, std::vector<float>::iterator& weights)
{
  // Resize arrays
  this->_w.resize(4 * hidden_size, input_size + hidden_size);
  this->_b.resize(4 * hidden_size);
  this->_xh.resize(input_size + hidden_size);
  this->_ifgo.resize(4 * hidden_size);
  this->_c.resize(hidden_size);

  // Assign in row-major because that's how PyTorch goes.
  for (int i = 0; i < this->_w.rows(); i++)
    for (int j = 0; j < this->_w.cols(); j++)
      this->_w(i, j) = *(weights++);
  for (int i = 0; i < this->_b.size(); i++)
    this->_b[i] = *(weights++);
  const int h_offset = input_size;
  for (int i = 0; i < hidden_size; i++)
    this->_xh[i + h_offset] = *(weights++);
  for (int i = 0; i < hidden_size; i++)
    this->_c[i] = *(weights++);

  // Save initial state templates for multi-channel cloning.
  // Zero the input portion (only hidden portion was loaded from weights).
  for (int i = 0; i < input_size; i++)
    this->_xh[i] = 0.0f;
  this->_initial_xh = this->_xh;
  this->_initial_c = this->_c;
}

void nam::lstm::LSTMCell::process_(const Eigen::VectorXf& x)
{
  const long hidden_size = this->_get_hidden_size();
  const long input_size = this->_get_input_size();
  // Assign inputs
  this->_xh(Eigen::seq(0, input_size - 1)) = x;
  // The matmul
  this->_ifgo = this->_w * this->_xh + this->_b;
  // Elementwise updates (apply nonlinearities here)
  const long i_offset = 0;
  const long f_offset = hidden_size;
  const long g_offset = 2 * hidden_size;
  const long o_offset = 3 * hidden_size;
  const long h_offset = input_size;

  if (activations::Activation::using_fast_tanh)
  {
    for (auto i = 0; i < hidden_size; i++)
      this->_c[i] =
        activations::fast_sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
        + activations::fast_sigmoid(this->_ifgo[i + i_offset]) * activations::fast_tanh(this->_ifgo[i + g_offset]);

    for (int i = 0; i < hidden_size; i++)
      this->_xh[i + h_offset] =
        activations::fast_sigmoid(this->_ifgo[i + o_offset]) * activations::fast_tanh(this->_c[i]);
  }
  else
  {
    for (auto i = 0; i < hidden_size; i++)
      this->_c[i] = activations::sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
                    + activations::sigmoid(this->_ifgo[i + i_offset]) * tanhf(this->_ifgo[i + g_offset]);

    for (int i = 0; i < hidden_size; i++)
      this->_xh[i + h_offset] = activations::sigmoid(this->_ifgo[i + o_offset]) * tanhf(this->_c[i]);
  }
}

// --- LSTMCell shared-weight multi-channel API --------------------------------

nam::lstm::LSTMCellState nam::lstm::LSTMCell::createState() const
{
  LSTMCellState state;
  state._xh = this->_initial_xh;
  state._ifgo.resize(this->_ifgo.size());
  state._c = this->_initial_c;
  return state;
}

void nam::lstm::LSTMCell::process_(const Eigen::VectorXf& x, LSTMCellState& state) const
{
  const long hidden_size = this->_get_hidden_size();
  const long input_size = this->_get_input_size();
  // Assign inputs
  state._xh(Eigen::seq(0, input_size - 1)) = x;
  // The matmul — shared weights, external state
  state._ifgo = this->_w * state._xh + this->_b;
  // Elementwise updates (apply nonlinearities)
  const long i_offset = 0;
  const long f_offset = hidden_size;
  const long g_offset = 2 * hidden_size;
  const long o_offset = 3 * hidden_size;
  const long h_offset = input_size;

  if (activations::Activation::using_fast_tanh)
  {
    for (auto i = 0; i < hidden_size; i++)
      state._c[i] =
        activations::fast_sigmoid(state._ifgo[i + f_offset]) * state._c[i]
        + activations::fast_sigmoid(state._ifgo[i + i_offset]) * activations::fast_tanh(state._ifgo[i + g_offset]);

    for (int i = 0; i < hidden_size; i++)
      state._xh[i + h_offset] =
        activations::fast_sigmoid(state._ifgo[i + o_offset]) * activations::fast_tanh(state._c[i]);
  }
  else
  {
    for (auto i = 0; i < hidden_size; i++)
      state._c[i] = activations::sigmoid(state._ifgo[i + f_offset]) * state._c[i]
                    + activations::sigmoid(state._ifgo[i + i_offset]) * tanhf(state._ifgo[i + g_offset]);

    for (int i = 0; i < hidden_size; i++)
      state._xh[i + h_offset] = activations::sigmoid(state._ifgo[i + o_offset]) * tanhf(state._c[i]);
  }
}

// --- LSTMCell batched processing (N channels, one GEMM) ---------------------

void nam::lstm::LSTMCell::processBatch_(Eigen::MatrixXf& batch_xh, Eigen::MatrixXf& batch_ifgo,
                                         LSTMCellState** states, int N) const
{
  const long hs = this->_get_hidden_size();
  const long is = this->_get_input_size();

  // 1. Stack N _xh vectors as columns into batch_xh.
  for (int ch = 0; ch < N; ++ch)
    batch_xh.col(ch) = states[ch]->_xh;

  // 2. Batched GEMM: batch_ifgo = _w * batch_xh + _b (broadcast).
  //    Pre-fill with bias, then GEMM adds the product.
  batch_ifgo.leftCols(N).colwise() = this->_b;

#if defined(__APPLE__)
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              (int)this->_w.rows(), N, (int)this->_w.cols(),
              1.0f,
              this->_w.data(), (int)this->_w.rows(),
              batch_xh.data(), (int)batch_xh.rows(),
              1.0f,  // beta=1.0: adds to pre-filled bias
              batch_ifgo.data(), (int)batch_ifgo.rows());
#else
  batch_ifgo.leftCols(N).noalias() += this->_w * batch_xh.leftCols(N);
#endif

  // 3. Per-channel gate updates (elementwise — _c is channel-specific).
  const long i_offset = 0;
  const long f_offset = hs;
  const long g_offset = 2 * hs;
  const long o_offset = 3 * hs;
  const long h_offset = is;

  if (activations::Activation::using_fast_tanh)
  {
    for (int ch = 0; ch < N; ++ch)
    {
      auto& st = *states[ch];
      const float* ifgo = batch_ifgo.col(ch).data();

      for (long i = 0; i < hs; i++)
        st._c[i] =
          activations::fast_sigmoid(ifgo[f_offset + i]) * st._c[i]
          + activations::fast_sigmoid(ifgo[i_offset + i]) * activations::fast_tanh(ifgo[g_offset + i]);

      for (long i = 0; i < hs; i++)
        st._xh[h_offset + i] =
          activations::fast_sigmoid(ifgo[o_offset + i]) * activations::fast_tanh(st._c[i]);
    }
  }
  else
  {
    for (int ch = 0; ch < N; ++ch)
    {
      auto& st = *states[ch];
      const float* ifgo = batch_ifgo.col(ch).data();

      for (long i = 0; i < hs; i++)
        st._c[i] = activations::sigmoid(ifgo[f_offset + i]) * st._c[i]
                    + activations::sigmoid(ifgo[i_offset + i]) * tanhf(ifgo[g_offset + i]);

      for (long i = 0; i < hs; i++)
        st._xh[h_offset + i] = activations::sigmoid(ifgo[o_offset + i]) * tanhf(st._c[i]);
    }
  }
}

nam::lstm::LSTM::LSTM(const int in_channels, const int out_channels, const int num_layers, const int input_size,
                      const int hidden_size, std::vector<float>& weights, const double expected_sample_rate)
: DSP(in_channels, out_channels, expected_sample_rate)
{
  // Allocate input and output vectors
  this->_input.resize(input_size);
  this->_output.resize(out_channels);

  std::vector<float>::iterator it = weights.begin();
  for (int i = 0; i < num_layers; i++)
    this->_layers.push_back(LSTMCell(i == 0 ? input_size : hidden_size, hidden_size, it));

  // Load head weight as matrix (out_channels x hidden_size)
  // Weights are stored row-major: first row (output 0), then row 1 (output 1), etc.
  this->_head_weight.resize(out_channels, hidden_size);
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    for (int h = 0; h < hidden_size; h++)
    {
      this->_head_weight(out_ch, h) = *(it++);
    }
  }

  // Load head bias as vector (out_channels)
  this->_head_bias.resize(out_channels);
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    this->_head_bias(out_ch) = *(it++);
  }

  assert(it == weights.end());
}

void nam::lstm::LSTM::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  for (int i = 0; i < num_frames; i++)
  {
    // Copy multi-channel input to _input vector
    for (int ch = 0; ch < in_channels; ch++)
    {
      this->_input(ch) = input[ch][i];
    }

    // Process sample (stores result in _output)
    this->_process_sample();

    // Copy multi-channel output from _output to output arrays
    for (int ch = 0; ch < out_channels; ch++)
    {
      output[ch][i] = this->_output(ch);
    }
  }
}

int nam::lstm::LSTM::PrewarmSamples()
{
  int result = (int)(0.5 * mExpectedSampleRate);
  // If the expected sample rate wasn't provided, it'll be -1.
  // Make sure something still happens.
  return result <= 0 ? 1 : result;
}

void nam::lstm::LSTM::_process_sample()
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  if (this->_layers.size() == 0)
  {
    // No layers - pass input through to output (using first in_channels of output)
    const int channels_to_copy = std::min(in_channels, out_channels);
    for (int ch = 0; ch < channels_to_copy; ch++)
      this->_output(ch) = this->_input(ch);
    // Zero-fill remaining output channels if in_channels < out_channels
    for (int ch = channels_to_copy; ch < out_channels; ch++)
      this->_output(ch) = 0.0f;
    return;
  }

  this->_layers[0].process_(this->_input);
  for (size_t i = 1; i < this->_layers.size(); i++)
    this->_layers[i].process_(this->_layers[i - 1].get_hidden_state());

  // Compute output using head weight matrix and bias vector
  // _output = _head_weight * hidden_state + _head_bias
  const Eigen::VectorXf& hidden_state = this->_layers[this->_layers.size() - 1].get_hidden_state();

  // Compute matrix-vector product: (out_channels x hidden_size) * (hidden_size) = (out_channels)
  // Store directly in _output (which is already sized correctly in constructor)
  this->_output.noalias() = this->_head_weight * hidden_state;

  // Add bias: (out_channels) += (out_channels)
  this->_output.noalias() += this->_head_bias;
}

// --- LSTM shared-weight multi-channel API ------------------------------------

nam::lstm::LSTMChannelState nam::lstm::LSTM::createTypedChannelState() const
{
  LSTMChannelState state;
  state.layers.reserve(this->_layers.size());
  for (const auto& layer : this->_layers)
    state.layers.push_back(layer.createState());
  state._input.resize(this->_input.size());
  state._input.setZero();
  state._output.resize(this->_output.size());
  state._output.setZero();
  return state;
}

void nam::lstm::LSTM::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames,
                              LSTMChannelState& state) const
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  for (int i = 0; i < num_frames; i++)
  {
    for (int ch = 0; ch < in_channels; ch++)
      state._input(ch) = input[ch][i];

    this->_process_sample(state);

    for (int ch = 0; ch < out_channels; ch++)
      output[ch][i] = state._output(ch);
  }
}

void nam::lstm::LSTM::_process_sample(LSTMChannelState& state) const
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  if (this->_layers.size() == 0)
  {
    const int channels_to_copy = std::min(in_channels, out_channels);
    for (int ch = 0; ch < channels_to_copy; ch++)
      state._output(ch) = state._input(ch);
    for (int ch = channels_to_copy; ch < out_channels; ch++)
      state._output(ch) = 0.0f;
    return;
  }

  this->_layers[0].process_(state._input, state.layers[0]);
  for (size_t i = 1; i < this->_layers.size(); i++)
    this->_layers[i].process_(
      this->_layers[i - 1].get_hidden_state(state.layers[i - 1]),
      state.layers[i]);

  const Eigen::VectorXf& hidden_state =
    this->_layers[this->_layers.size() - 1].get_hidden_state(
      state.layers[this->_layers.size() - 1]);

  state._output.noalias() = this->_head_weight * hidden_state;
  state._output.noalias() += this->_head_bias;
}

void nam::lstm::LSTM::prewarmTypedChannelState(LSTMChannelState& state) const
{
  const int prewarmSamples = (int)(0.5 * mExpectedSampleRate);
  const int samples = prewarmSamples <= 0 ? 1 : prewarmSamples;
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // Allocate temporary silence buffers on the message thread
  std::vector<std::vector<NAM_SAMPLE>> inputBuffers(in_channels);
  std::vector<std::vector<NAM_SAMPLE>> outputBuffers(out_channels);
  std::vector<NAM_SAMPLE*> inputPtrs(in_channels);
  std::vector<NAM_SAMPLE*> outputPtrs(out_channels);

  for (int ch = 0; ch < in_channels; ch++)
  {
    inputBuffers[ch].resize(samples, 0.0);
    inputPtrs[ch] = inputBuffers[ch].data();
  }
  for (int ch = 0; ch < out_channels; ch++)
  {
    outputBuffers[ch].resize(samples, 0.0);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  this->process(inputPtrs.data(), outputPtrs.data(), samples, state);
}

// --- DSP virtual interface overrides -----------------------------------------

std::unique_ptr<nam::ChannelState> nam::lstm::LSTM::createChannelState() const
{
  return std::make_unique<LSTMChannelState>(createTypedChannelState());
}

void nam::lstm::LSTM::processChannel(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames,
                                     ChannelState& state)
{
  auto& lstmState = static_cast<LSTMChannelState&>(state);
  process(input, output, num_frames, lstmState);
}

void nam::lstm::LSTM::prewarmChannelState(ChannelState& state)
{
  auto& lstmState = static_cast<LSTMChannelState&>(state);
  prewarmTypedChannelState(lstmState);
}

// --- Batched multi-channel processing ----------------------------------------

void nam::lstm::LSTM::prepareBatch(int maxBatchSize)
{
  if (maxBatchSize <= 0)
    return;
  _max_batch_size = maxBatchSize;

  // Allocate per-layer batch scratch matrices.
  _batch_layer_scratch.resize(this->_layers.size());
  for (size_t l = 0; l < this->_layers.size(); l++)
  {
    // _w is (4*dh × (dx+dh)) — use friend access to get dimensions.
    const int xh_rows = (int)this->_layers[l]._w.cols();   // dx + dh
    const int ifgo_rows = (int)this->_layers[l]._w.rows();  // 4 * dh

    _batch_layer_scratch[l].xh.resize(xh_rows, maxBatchSize);
    _batch_layer_scratch[l].ifgo.resize(ifgo_rows, maxBatchSize);
  }

  // Head projection scratch: hidden_size from last layer.
  if (!this->_layers.empty())
  {
    const int hidden_size = (int)this->_layers.back()._get_hidden_size();
    _batch_hidden.resize(hidden_size, maxBatchSize);
  }
  _batch_output.resize(this->_head_weight.rows(), maxBatchSize);
}

void nam::lstm::LSTM::setPreferSmallBatchProcessing(bool enabled)
{
  _prefer_small_batch_processing.store(enabled, std::memory_order_relaxed);
}

void nam::lstm::LSTM::processBatchChannels(float* const* monoInputs, float* const* monoOutputs,
                                            int numFrames, ChannelState** states, int numChannels)
{
  // Fallback to sequential for small batch sizes or if batch scratch not allocated.
  const bool preferSmallBatch = _prefer_small_batch_processing.load(std::memory_order_relaxed);
  if (((!preferSmallBatch) && numChannels <= 2) || _max_batch_size < numChannels)
  {
    DSP::processBatchChannels(monoInputs, monoOutputs, numFrames, states, numChannels);
    return;
  }

  const int N = numChannels;
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();
  const int numLayers = (int)this->_layers.size();

  // Collect typed state pointers.
  // Use a small stack array for typical sizes, heap-allocate for large N.
  LSTMChannelState* lstmStates[64];
  std::vector<LSTMChannelState*> lstmStatesHeap;
  LSTMChannelState** sp;
  if (N <= 64)
  {
    sp = lstmStates;
  }
  else
  {
    lstmStatesHeap.resize(N);
    sp = lstmStatesHeap.data();
  }
  for (int ch = 0; ch < N; ++ch)
    sp[ch] = static_cast<LSTMChannelState*>(states[ch]);

  // Per-layer cell state pointers (reused each sample).
  LSTMCellState* cellStates[64];
  std::vector<LSTMCellState*> cellStatesHeap;
  LSTMCellState** cp;
  if (N <= 64)
    cp = cellStates;
  else
  {
    cellStatesHeap.resize(N);
    cp = cellStatesHeap.data();
  }

  // Process frame by frame — LSTM is per-sample recurrent.
  for (int f = 0; f < numFrames; ++f)
  {
    // 1. Set each channel's input from the mono buffers.
    for (int ch = 0; ch < N; ++ch)
    {
      for (int c = 0; c < in_channels; ++c)
        sp[ch]->_input(c) = monoInputs[ch][f];
    }

    if (numLayers == 0)
    {
      // No layers — pass input to output.
      for (int ch = 0; ch < N; ++ch)
      {
        const int toCopy = std::min(in_channels, out_channels);
        for (int c = 0; c < toCopy; ++c)
          sp[ch]->_output(c) = sp[ch]->_input(c);
        for (int c = toCopy; c < out_channels; ++c)
          sp[ch]->_output(c) = 0.0f;
      }
    }
    else
    {
      // --- Layer 0: input is the model input ---
      for (int ch = 0; ch < N; ++ch)
      {
        auto& ls = sp[ch]->layers[0];
        // Write input into _xh (first `input_size` elements).
        ls._xh.head(sp[ch]->_input.size()) = sp[ch]->_input;
        cp[ch] = &ls;
      }
      this->_layers[0].processBatch_(
          _batch_layer_scratch[0].xh, _batch_layer_scratch[0].ifgo, cp, N);

      // --- Subsequent layers: input is previous layer's hidden state ---
      for (int l = 1; l < numLayers; ++l)
      {
        for (int ch = 0; ch < N; ++ch)
        {
          // Get hidden state from previous layer.
          const Eigen::VectorXf h = this->_layers[l - 1].get_hidden_state(sp[ch]->layers[l - 1]);
          // Write into this layer's _xh.
          sp[ch]->layers[l]._xh.head(h.size()) = h;
          cp[ch] = &sp[ch]->layers[l];
        }
        this->_layers[l].processBatch_(
            _batch_layer_scratch[l].xh, _batch_layer_scratch[l].ifgo, cp, N);
      }

      // --- Head projection: batched across all N channels ---
      // Stack hidden states from last layer.
      for (int ch = 0; ch < N; ++ch)
      {
        _batch_hidden.col(ch) =
            this->_layers[numLayers - 1].get_hidden_state(sp[ch]->layers[numLayers - 1]);
      }

      // batch_output = _head_weight * batch_hidden + _head_bias (broadcast)
      _batch_output.leftCols(N).colwise() = this->_head_bias;
#if defined(__APPLE__)
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  (int)this->_head_weight.rows(), N, (int)this->_head_weight.cols(),
                  1.0f,
                  this->_head_weight.data(), (int)this->_head_weight.rows(),
                  _batch_hidden.data(), (int)_batch_hidden.rows(),
                  1.0f,
                  _batch_output.data(), (int)_batch_output.rows());
#else
      _batch_output.leftCols(N).noalias() += this->_head_weight * _batch_hidden.leftCols(N);
#endif

      // Scatter outputs back to per-channel state.
      for (int ch = 0; ch < N; ++ch)
        sp[ch]->_output = _batch_output.col(ch);
    }

    // 2. Copy per-channel output to the mono output buffers.
    for (int ch = 0; ch < N; ++ch)
    {
      for (int c = 0; c < out_channels; ++c)
        monoOutputs[ch][f] = sp[ch]->_output(c);
    }
  }
}

// Config parser
nam::lstm::LSTMConfig nam::lstm::parse_config_json(const nlohmann::json& config)
{
  LSTMConfig c;
  c.num_layers = config["num_layers"];
  c.input_size = config["input_size"];
  c.hidden_size = config["hidden_size"];
  // Default to 1 channel in/out for backward compatibility
  c.in_channels = config.value("in_channels", 1);
  c.out_channels = config.value("out_channels", 1);
  return c;
}

// LSTMConfig::create()
std::unique_ptr<nam::DSP> nam::lstm::LSTMConfig::create(std::vector<float> weights, double sampleRate)
{
  return std::make_unique<nam::lstm::LSTM>(in_channels, out_channels, num_layers, input_size, hidden_size, weights,
                                           sampleRate);
}

// Config parser for ConfigParserRegistry
std::unique_ptr<nam::ModelConfig> nam::lstm::create_config(const nlohmann::json& config, double sampleRate)
{
  (void)sampleRate;
  auto c = std::make_unique<LSTMConfig>();
  auto parsed = parse_config_json(config);
  *c = parsed;
  return c;
}

// Register the config parser
namespace
{
static nam::ConfigParserHelper _register_LSTM("LSTM", nam::lstm::create_config);
}
