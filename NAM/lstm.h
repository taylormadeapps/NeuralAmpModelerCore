#pragma once
// LSTM implementation

#include <map>
#include <vector>
#include <memory>
#include <atomic>

#include <Eigen/Dense>

#include "dsp.h"

namespace nam
{
namespace lstm
{

// =============================================================================
// Per-channel state structs — separated from weights for shared-weight
// multi-channel processing. One state per audio channel; the LSTMCell / LSTM
// classes hold the shared (immutable after load) weight matrices.
// =============================================================================

/// \brief Mutable per-channel state for a single LSTM cell.
struct LSTMCellState
{
  Eigen::VectorXf _xh;    ///< Concatenated input + hidden state
  Eigen::VectorXf _ifgo;  ///< Gate activations (scratch — recomputed each sample)
  Eigen::VectorXf _c;     ///< Cell state
};

/// \brief A single LSTM cell
class LSTMCell
{
public:
  /// \brief Constructor
  /// \param input_size Size of the input vector
  /// \param hidden_size Size of the hidden state
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  LSTMCell(const int input_size, const int hidden_size, std::vector<float>::iterator& weights);

  /// \brief Get the current hidden state
  /// \return Hidden state vector
  Eigen::VectorXf get_hidden_state() const { return this->_xh(Eigen::placeholders::lastN(this->_get_hidden_size())); };

  /// \brief Process a single input vector
  /// \param x Input vector
  void process_(const Eigen::VectorXf& x);

  // --- Shared-weight multi-channel API ----------------------------------------

  /// \brief Create a fresh per-channel state from initial conditions.
  LSTMCellState createState() const;

  /// \brief Process using shared weights (this) and external per-channel state.
  void process_(const Eigen::VectorXf& x, LSTMCellState& state) const;

  /// \brief Get hidden state from external per-channel state.
  Eigen::VectorXf get_hidden_state(const LSTMCellState& state) const
  {
    return state._xh(Eigen::placeholders::lastN(this->_get_hidden_size()));
  }

  /// \brief Batched processing: N channels' _xh vectors → one GEMM → per-channel gate updates.
  /// batch_xh and batch_ifgo must be pre-allocated to at least (rows × N) columns.
  void processBatch_(Eigen::MatrixXf& batch_xh, Eigen::MatrixXf& batch_ifgo,
                     LSTMCellState** states, int N) const;

private:
  // Parameters
  // xh -> ifgo
  // (dx+dh) -> (4*dh)
  Eigen::MatrixXf _w;
  Eigen::VectorXf _b;

  // State
  // Concatenated input and hidden state
  Eigen::VectorXf _xh;
  // Input, Forget, Cell, Output gates
  Eigen::VectorXf _ifgo;

  // Cell state
  Eigen::VectorXf _c;

  // Initial state templates (saved from constructor for cloning to new channels)
  Eigen::VectorXf _initial_xh;
  Eigen::VectorXf _initial_c;

  long _get_hidden_size() const { return this->_b.size() / 4; };
  long _get_input_size() const { return this->_xh.size() - this->_get_hidden_size(); };

  // Allow LSTM to access _w dimensions for batch scratch allocation.
  friend class LSTM;
};

/// \brief Mutable per-channel state for a complete LSTM model.
/// One of these per audio channel; the LSTM class holds the shared weights.
struct LSTMChannelState : public ChannelState
{
  std::vector<LSTMCellState> layers;
  Eigen::VectorXf _input;
  Eigen::VectorXf _output;
};

/// \brief A multi-layer LSTM model
///
/// A multi-layer LSTM processes audio frame-by-frame, maintaining hidden states
/// across layers. Each layer processes the hidden state from the previous layer as input.
class LSTM : public DSP
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param num_layers Number of LSTM layers
  /// \param input_size Size of the input to each LSTM cell
  /// \param hidden_size Size of the hidden state in each LSTM cell
  /// \param weights Model weights vector
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  LSTM(const int in_channels, const int out_channels, const int num_layers, const int input_size, const int hidden_size,
       std::vector<float>& weights, const double expected_sample_rate = -1.0);

  /// \brief Destructor
  ~LSTM() = default;

  /// \brief Process audio frames
  /// \param input Input audio buffers
  /// \param output Output audio buffers
  /// \param num_frames Number of frames to process
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

  // --- Shared-weight multi-channel API ----------------------------------------

  /// \brief Create a fresh channel state (per-cell states + I/O vectors).
  LSTMChannelState createTypedChannelState() const;

  /// \brief Process using shared weights and external channel state.
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames,
               LSTMChannelState& state) const;

  /// \brief Prewarm an external channel state by processing silence.
  void prewarmTypedChannelState(LSTMChannelState& state) const;

  // --- DSP virtual interface overrides ----------------------------------------
  std::unique_ptr<ChannelState> createChannelState() const override;
  void processChannel(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames, ChannelState& state) override;
  void prewarmChannelState(ChannelState& state) override;
  void processBatchChannels(float* const* monoInputs, float* const* monoOutputs,
                            int numFrames, ChannelState** states, int numChannels) override;
  void prepareBatch(int maxBatchSize) override;
  void setPreferSmallBatchProcessing(bool enabled) override;

protected:
  // Hacky, but a half-second seems to work for most models.
  int PrewarmSamples() override;

  Eigen::MatrixXf _head_weight; // (out_channels x hidden_size)
  Eigen::VectorXf _head_bias; // (out_channels)
  std::vector<LSTMCell> _layers;

  void _process_sample();
  void _process_sample(LSTMChannelState& state) const;

  // Input to the LSTM.
  // Since this is assumed to not be a parametric model, its shape should be (in_channels,)
  Eigen::VectorXf _input;
  // Output from _process_sample - multi-channel output vector (size out_channels)
  Eigen::VectorXf _output;

  // Batch scratch matrices for N-channel processing (pre-allocated on message thread).
  // Per-layer: one pair of (xh, ifgo) matrices sized for the layer's dimensions.
  struct BatchLayerScratch
  {
    Eigen::MatrixXf xh;    // (dx+dh × max_batch)
    Eigen::MatrixXf ifgo;  // (4*dh × max_batch)
  };
  std::vector<BatchLayerScratch> _batch_layer_scratch;
  Eigen::MatrixXf _batch_hidden;  // (dh × max_batch) — last layer hidden states
  Eigen::MatrixXf _batch_output;  // (out_ch × max_batch) — head projection result
  int _max_batch_size = 0;
  std::atomic<bool> _prefer_small_batch_processing { false };
};

/// \brief Configuration for an LSTM model
struct LSTMConfig : public ModelConfig
{
  int num_layers;
  int input_size;
  int hidden_size;
  int in_channels;
  int out_channels;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

/// \brief Parse LSTM configuration from JSON
/// \param config JSON configuration object
/// \return LSTMConfig
LSTMConfig parse_config_json(const nlohmann::json& config);

/// \brief Config parser for ConfigParserRegistry
std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);

}; // namespace lstm
}; // namespace nam
