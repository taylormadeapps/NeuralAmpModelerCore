#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "dsp.h"
#include "model_config.h"
#include "slimmable.h"

namespace nam
{
namespace container
{

struct Submodel
{
  double max_value;
  std::unique_ptr<DSP> model;
};

struct ContainerChannelState : public ChannelState
{
  // External state is permanently bound to the submodel that was active when
  // the state was created. A quality change therefore requires constructing
  // and publishing a new set of channel states; an existing state can never be
  // paired with a different submodel's weights.
  size_t submodel_index = 0;
  std::unique_ptr<ChannelState> submodel_state;
};

/// \brief A container model that holds multiple submodels at different sizes
///
/// SetSlimmableSize selects the active submodel based on the max_value thresholds.
/// Each submodel covers values up to (but not including) its max_value.
/// The last submodel is the fallback for values at or above the last threshold.
class ContainerModel : public DSP, public SlimmableModel
{
public:
  /// \brief Constructor
  /// \param submodels Vector of submodels sorted by max_value ascending
  /// \param expected_sample_rate Expected sample rate in Hz
  ContainerModel(std::vector<Submodel> submodels, const double expected_sample_rate);

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;
  RuntimeImplementation GetRuntimeImplementation() const override;
  std::unique_ptr<ChannelState> createChannelState() const override;
  void processChannel(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames, ChannelState& state) override;
  void prewarmChannelState(ChannelState& state) override;
  void processBatchChannels(NAM_SAMPLE* const* monoInputs, NAM_SAMPLE* const* monoOutputs,
                            int numFrames, ChannelState** states, int numChannels) override;
  void prepareBatch(int maxBatchSize) override;
  void prewarm() override;
  void Reset(const double sampleRate, const int maxBufferSize) override;
  void SetPrewarmOnReset(const bool prewarmOnReset) override;
  void SetSlimmableSize(const double val) override;
  std::vector<double> GetSlimmableSizeBreakpoints() const override;
  int GetPrewarmSamples() override;

private:
  size_t _get_index_for_slimmable_size(const double val) const;

  std::vector<Submodel> _submodels;
  std::atomic<size_t> _active_index{0};
  std::mutex _slim_set_mutex;

  DSP& _model_at(const size_t index) { return *_submodels[index].model; }
  const DSP& _model_at(const size_t index) const { return *_submodels[index].model; }
};

// Config / registration

struct ContainerConfig : public ModelConfig
{
  nlohmann::json raw_config;
  double sample_rate;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);

} // namespace container
} // namespace nam
