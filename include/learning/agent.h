#pragma once
// The rat's learning system: a recurrent DQN-style policy (double-DQN
// targets, Adam, epsilon-greedy, experience replay, soft target updates)
// with an auxiliary predictive world model, intrinsic curiosity, bounded
// episodic memory, structural plasticity and consolidation support.
//
// All training is single-threaded and deterministic given the RNG seed.
// The recurrent state is truncated-BPTT(1): each replay transition stores
// the h_prev used when it was collected (stale-state approximation).

#include "learning/agent_state.h"
#include "learning/neural_net.h"
#include "learning/replay_buffer.h"
#include "organism/development.h"
#include "organism/episodic_memory.h"
#include "organism/novelty.h"
#include "simulation/observation.h"
#include "simulation/rat.h"
#include "utility/config.h"
#include "utility/rng.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sir {

class Agent {
 public:
  Agent(const Config& cfg, Rng& rng);

  // Dimensions.
  int inputSize() const { return input_size_; }
  int rnnSize() const { return cfg_.rnn_hidden; }
  int policyOut() const { return 4; }
  int predOut() const { return 16; }
  static int predOutStatic() { return 16; }

  // --- decision (epsilon-greedy over online Q(obs, h); advances h_) ---
  struct Decision {
    Action action = Action::Up;
    bool explored = false;
  };
  Decision selectAction(const float* obs);

  // Diagnostics/tests: Q-values for `obs` using the current recurrent state
  // (does not alter it). Returns policy_out values.
  void qValuesFor(const float* obs, float* q_out) const;

  // Recurrent working memory: transient state, reset at session start and on
  // major lifecycle boundaries (maze regeneration).
  const float* recurrentState() const { return h_.data(); }
  void resetRecurrent();
  // The prediction of the most recent decision step (used by computeIntrinsics).
  const float* lastPrediction() const { return last_pred_.data(); }

  // --- phase 1: intrinsic signals (call after sim.step, before reward
  // composition). Computes novelty, prediction error, curiosity reward and
  // updates the uncertainty EMA. `reward_ext` is the externally determined
  // reward (navigation + homeostatic), which is what the model predicts.
  void computeIntrinsics(const float* s, const float* s2,
                         const float homeo_targets[3], float reward_ext);

  // --- phase 2: store transition, maybe train, maybe update episodic memory.
  // `r_total` is the full composite reward (ext + curiosity + prediction
  // bonus + collapse penalty); `reward_ext` is the external reward used as
  // the world-model prediction target.
  void observeAndTrain(const float* s, const float* h_prev, Action action,
                       float r_total, float reward_ext, const float* s2,
                       bool done, const float homeo_targets[3], bool cheese,
                       bool wall_hit, uint64_t sim_steps);

  // --- intrinsic signal accessors (for reward composition and metrics) ---
  float lastNovelty() const { return last_novelty_; }
  float lastCuriosityReward() const { return last_curiosity_reward_; }
  float lastPredError() const { return last_pred_error_; }
  float uncertainty() const { return uncertainty_; }
  float predictionLossEma() const { return pred_loss_ema_; }

  // --- consolidation ---
  bool consolidationAvailable() const;
  // Runs up to max_ops bounded training batches (mix of episodic memory and
  // replay). Returns the number of ops executed.
  int consolidationTrainOps(int max_ops);

  // --- structural plasticity (consolidation time only) ---
  // Prunes weak connections within hard limits, reactivates a bounded number
  // of previously useful dormant connections, strengthens useful ones, then
  // validates; rolls back on instability. `eval_obs` is a recent observation
  // buffer (eval_passes entries) used for the bounded evaluation.
  void plasticityEvaluate(const std::vector<float>& eval_obs, int eval_passes);

  // --- metrics ---
  double epsilon() const { return epsilon_; }
  uint64_t trainingUpdates() const { return training_updates_; }
  uint64_t invalidUpdates() const { return invalid_updates_; }
  uint64_t lifetimeSteps() const { return lifetime_steps_; }
  uint64_t exploredCount() const { return explored_count_; }
  size_t episodicSize() const { return episodic_.size(); }
  size_t episodicCapacity() const { return episodic_.capacity(); }
  uint64_t episodicReplacements() const { return episodic_.replacements(); }
  uint64_t prunedTotal() const { return pruned_total_; }
  uint64_t rewiredTotal() const { return rewired_total_; }
  uint64_t structuralAccepted() const { return structural_accepted_; }
  uint64_t structuralRejected() const { return structural_rejected_; }
  uint64_t consolidationCycles() const { return consolidation_cycles_; }
  uint64_t consolidationTrainOpsTotal() const { return consolidation_train_ops_total_; }
  size_t activeConnections() const { return online_.activeCount(); }
  size_t dormantConnections() const { return online_.dormantCount(); }
  double currentEpsilon() const { return epsilon_; }

  // --- persistence ---
  void exportState(AgentState& out) const;
  bool importState(const AgentState& in);  // validates everything; false = incompatible
  void setEpsilon(double e) { epsilon_ = e; }

 private:
  struct Sample {
    const float* s;
    const float* s2;
    const float* h_prev;
    const float* tgt;  // homeostatic targets (null for episodic entries)
    int action;
    float reward;      // composite reward (RL target)
    float reward_ext;  // external reward (world-model target)
    bool done;
  };

  bool trainBatchOn(const std::vector<Sample>& samples,
                    const std::vector<size_t>* replay_idx);  // true = applied
  void gatherSamples(size_t count, std::vector<Sample>* out, bool prefer_episodic);
  float computeEpsilon(uint64_t steps) const;
  void adamUpdate(const float* grad);
  void softUpdateTarget();
  bool validateAndRestore(const std::vector<float>& snapshot);
  bool allMomentsFinite() const;
  void updatePredictionTargets(const float* s, const float* s2,
                               const float* homeo_targets, float reward_ext,
                               float* out) const;

  const Config& cfg_;
  Rng& rng_;
  Development development_;

  NeuralNet online_;
  NeuralNet target_;
  std::vector<float> adam_m_, adam_v_;
  ReplayBuffer replay_;
  EpisodicMemory episodic_;
  NoveltyEstimator novelty_;

  // working memory
  std::vector<float> h_;
  std::vector<float> last_pred_;

  // intrinsic state
  float last_novelty_ = 0.0f;
  float last_curiosity_reward_ = 0.0f;
  float last_pred_error_ = 0.0f;
  float uncertainty_ = 0.5f;
  float pred_loss_ema_ = 0.5f;
  float novelty_ema_ = 0.0f;

  // counters
  uint64_t lifetime_steps_ = 0;
  uint64_t training_updates_ = 0;
  uint64_t invalid_updates_ = 0;
  uint64_t explored_count_ = 0;
  uint64_t pruned_total_ = 0;
  uint64_t rewired_total_ = 0;
  uint64_t structural_accepted_ = 0;
  uint64_t structural_rejected_ = 0;
  uint64_t consolidation_cycles_ = 0;
  uint64_t consolidation_train_ops_total_ = 0;
  double epsilon_ = 0.9;
  int input_size_ = 0;
};

}  // namespace sir
