#include "app/application.h"

#include <SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <thread>

namespace sir {

std::atomic<int> g_signal_count{0};

namespace {

double clampd(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

uint64_t Application::nowMs() const {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

int Application::run(const Options& opts) {
  session_start_ms_ = nowMs();

  // --- configuration ---
  std::string cfg_path = opts.config_path.empty() ? "config/default.cfg" : opts.config_path;
  std::string cfg_error, cfg_warnings;
  const int applied = cfg_.loadFromFile(cfg_path, &cfg_error, &cfg_warnings);
  if (applied < 0) {
    log_.warn("config file not readable (" + cfg_path + "); using built-in defaults: " +
              cfg_error);
  } else if (!cfg_warnings.empty()) {
    log_.warn("config warnings for " + cfg_path + ":\n" + cfg_warnings);
  }
  // Environment seed override (SIR_SEED) wins over the config file.
  if (opts.seed_override != 0) cfg_.seed = opts.seed_override;
  debug_ = cfg_.debug_display;  // config default for the diagnostics panel

  // --- directories + logger ---
  try {
    std::filesystem::create_directories(cfg_.checkpoint_dir);
    std::filesystem::create_directories(
        std::filesystem::path(cfg_.log_file).parent_path());
  } catch (const std::exception& e) {
    fprintf(stderr, "cannot create data directories: %s\n", e.what());
    return 1;
  }
  if (!log_.open(cfg_.log_file, cfg_.log_max_bytes)) {
    fprintf(stderr, "cannot open log file: %s\n", cfg_.log_file.c_str());
    return 1;
  }

  // --- randomness + checkpoint recovery (order matters: restore the RNG
  // state BEFORE constructing simulation/agent) ---
  const uint32_t seed =
      opts.seed_override != 0 ? opts.seed_override
                              : (cfg_.seed == 0 ? randomSeed() : cfg_.seed);
  rng_ = std::make_unique<Rng>(seed);
  store_ = std::make_unique<CheckpointStore>(cfg_, log_);

  AgentState loaded;
  const LoadResult lr = store_->load(&loaded);
  bool recovered = false;
  if (lr == LoadResult::Ok) {
    if (!loaded.rng_state.empty()) {
      if (!rng_->restoreState(loaded.rng_state)) {
        log_.warn("checkpoint RNG state rejected; keeping current seed");
      } else {
        recovered = true;
      }
    }
    loaded_runtime_ms_ = loaded.active_runtime_ms;
    consolidation_cycles_ = loaded.consolidation_cycles;
    last_consolidation_step_ = loaded.last_consolidation_step;
    checkpoints_saved_ = loaded.checkpoints_saved;
  } else if (lr == LoadResult::Incompatible) {
    log_.warn("checkpoint topology incompatible with configuration; a new organism starts");
  } else if (lr == LoadResult::Corrupt) {
    log_.warn("no valid checkpoint found; a new organism starts");
  } else {
    log_.info("no checkpoint found; a new organism starts");
  }

  sim_ = std::make_unique<Simulation>(cfg_, *rng_);
  agent_ = std::make_unique<Agent>(cfg_, *rng_);

  if (lr == LoadResult::Ok) {
    if (agent_->importState(loaded)) {
      sim_->setLifetimeSteps(loaded.lifetime_steps);
      sim_->setCheeseTotal(loaded.cheese_total);
      sim_->setMazeGenerations(loaded.maze_generations);
      sim_->homeostasis().restore(loaded.homeo);
      metrics_.setWallHitsTotal(0);  // transient within a session
      log_.info("organism state restored (age " + std::to_string(loaded.lifetime_steps) +
                " steps, cheese " + std::to_string(loaded.cheese_total) + ")");
    } else {
      log_.warn("checkpoint state rejected by agent validation; fresh organism starts");
      const uint32_t fresh_seed =
          opts.seed_override != 0 ? opts.seed_override
                                  : (cfg_.seed == 0 ? randomSeed() : cfg_.seed);
      rng_ = std::make_unique<Rng>(fresh_seed);
      sim_ = std::make_unique<Simulation>(cfg_, *rng_);
      agent_ = std::make_unique<Agent>(cfg_, *rng_);
      loaded_runtime_ms_ = 0;
      consolidation_cycles_ = 0;
      last_consolidation_step_ = 0;
      checkpoints_saved_ = 0;
    }
  }
  log_.info("seed=" + std::to_string(rng_->seed()) +
            " rng_restored=" + (recovered ? "yes" : "no") +
            " config_keys=" + std::to_string(applied));

  // --- rendering ---
  if (!opts.headless) {
    std::string rerr;
    if (!renderer_.init(cfg_, &rerr)) {
      log_.error("renderer init failed: " + rerr);
      fprintf(stderr, "error: %s\n", rerr.c_str());
      return 1;
    }
  }

  // Observation ring for plasticity evaluation (bounded).
  recent_obs_.assign(static_cast<size_t>(64) * agent_->inputSize(), 0.0f);

  last_autosave_ms_ = nowMs();
  last_sim_step_ms_ = nowMs();
  last_fps_ms_ = nowMs();
  last_sps_ms_ = nowMs();
  const double step_dt = 1.0 / static_cast<double>(std::max(1, cfg_.sim_steps_per_second));
  const uint64_t frame_dt_ms =
      static_cast<uint64_t>(1000.0 / static_cast<double>(std::max(1, cfg_.render_frames_per_second)));
  uint64_t next_frame_ms = nowMs();
  bool quit = false;

  log_.info("session started");

  while (!quit) {
    pollSignals();
    if (g_signal_count.load() > 0) break;

    // --- events ---
    SDL_Event ev;
    while (!opts.headless && SDL_PollEvent(&ev)) {
      if (ev.type == SDL_QUIT) quit = true;
      if (ev.type == SDL_KEYDOWN) {
        switch (ev.key.keysym.sym) {
          case SDLK_SPACE:
            paused_ = !paused_;
            break;
          case SDLK_r:
            if (!consolidating_) {
              sim_->regenerateMaze();
              agent_->resetRecurrent();
              log_.info("manual maze regeneration (weights untouched)");
            }
            break;
          case SDLK_s:
            saveCheckpointNow("manual");
            break;
          case SDLK_d:
            debug_ = !debug_;
            break;
          case SDLK_ESCAPE:
            quit = true;
            break;
          default:
            break;
        }
      }
    }

    // --- simulation steps (fixed timestep, bounded catch-up) ---
    if (!paused_) {
      const uint64_t t = nowMs();
      const uint64_t elapsed = t - last_sim_step_ms_;
      last_sim_step_ms_ = t;
      if (!opts.headless) {
        sim_accum_ += static_cast<double>(elapsed) / 1000.0;
        int catch_up = 0;
        while (sim_accum_ >= step_dt &&
               catch_up < cfg_.max_catchup_steps_per_frame) {
          if (consolidating_) {
            doConsolidationStep();
          } else {
            doSimStep();
          }
          sim_accum_ -= step_dt;
          ++catch_up;
        }
        if (sim_accum_ >= step_dt) sim_accum_ = 0.0;  // drop backlog, stay live
      } else {
        // Headless: run unthrottled (diagnostic mode) but stay responsive to
        // signals by bounding work per iteration.
        int n = 0;
        while (n < 500 && !quit) {
          if (g_signal_count.load() > 0) { quit = true; break; }
          if (consolidating_) doConsolidationStep();
          else doSimStep();
          ++n;
          if (opts.max_steps > 0 &&
              sim_->lifetimeSteps() >= opts.max_steps) {
            quit = true;
            break;
          }
        }
      }
      // Step-rate tracking for the panel.
      ++steps_since_sps_;
      if (t - last_sps_ms_ >= 1000) {
        sim_sps_ = static_cast<double>(steps_since_sps_) * 1000.0 /
                   static_cast<double>(t - last_sps_ms_);
        steps_since_sps_ = 0;
        last_sps_ms_ = t;
      }
      if (opts.max_steps > 0 && sim_->lifetimeSteps() >= opts.max_steps) {
        log_.info("SIR_MAX_STEPS reached; shutting down cleanly");
        quit = true;
      }
    }

    // --- render ---
    if (!opts.headless) {
      const uint64_t t = nowMs();
      if (t >= next_frame_ms) {
        next_frame_ms = t + frame_dt_ms;
        ++frames_rendered_;
        ++frames_since_fps_;
        // Frame-rate measurement for the panel (observed, real values).
        if (t - last_fps_ms_ >= 500) {
          fps_ = static_cast<double>(frames_since_fps_) * 1000.0 /
                 static_cast<double>(t - last_fps_ms_);
          frames_since_fps_ = 0;
          last_fps_ms_ = t;
        }
        PanelData panel;
        fillPanel(panel);
        renderer_.render(*sim_, panel);
        if (!screenshot_done_ && !opts.screenshot_path.empty() &&
            frames_rendered_ >= 30) {
          if (renderer_.saveScreenshot(opts.screenshot_path)) {
            log_.info("screenshot saved: " + opts.screenshot_path);
          } else {
            log_.warn("screenshot failed: " + opts.screenshot_path);
          }
          screenshot_done_ = true;
        }
      }
      SDL_Delay(4);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // --- autosave ---
    const uint64_t t = nowMs();
    if (t - last_autosave_ms_ >=
        static_cast<uint64_t>(cfg_.autosave_interval_seconds) * 1000ULL) {
      last_autosave_ms_ = t;
      saveCheckpointNow("autosave");
    }
  }

  // --- clean shutdown ---
  if (consolidating_) endConsolidation();
  saveCheckpointNow("shutdown");
  log_.info("shutdown: steps=" + std::to_string(sim_->lifetimeSteps()) +
            " cheese=" + std::to_string(sim_->cheeseTotal()) +
            " training=" + std::to_string(agent_->trainingUpdates()));
  if (!opts.headless) renderer_.shutdown();
  return 0;
}

void Application::pollSignals() {
  const int n = g_signal_count.load();
  if (n > 1) {
    // Second signal: exit immediately (async-signal-safe path).
    _Exit(130);
  }
}

void Application::saveCheckpointNow(const char* reason) {
  AgentState state;
  agent_->exportState(state);
  state.homeo = sim_->homeostasis().snapshot();
  state.cheese_total = sim_->cheeseTotal();
  state.lifetime_steps = sim_->lifetimeSteps();
  state.maze_generations = sim_->mazeGenerations();
  state.checkpoints_saved = static_cast<uint32_t>(checkpoints_saved_ + 1);
  state.consolidation_cycles = static_cast<uint32_t>(consolidation_cycles_);
  state.last_consolidation_step = last_consolidation_step_;
  state.active_runtime_ms =
      loaded_runtime_ms_ + (nowMs() - session_start_ms_);
  state.timestamp_utc = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  if (store_->save(state)) {
    ++checkpoints_saved_;
    log_.info("checkpoint saved (" + std::string(reason) + ")");
  } else {
    log_.error("checkpoint save failed (" + std::string(reason) + ")");
  }
}

void Application::doSimStep() {
  // Phase 1: observe + decide. NOTE: sim.observe() and agent.recurrentState()
  // return pointers into internal buffers that are mutated by the following
  // calls (sim.step() shifts the frame history; selectAction() advances the
  // recurrent state). Copy both into locals so the stored transition is the
  // true pre-action state (aliasing fix).
  std::vector<float> s_local(agent_->inputSize());
  std::copy(sim_->observe(), sim_->observe() + agent_->inputSize(),
            s_local.begin());
  std::vector<float> h_prev_local(agent_->rnnSize());
  std::copy(agent_->recurrentState(),
            agent_->recurrentState() + agent_->rnnSize(), h_prev_local.begin());
  const float* s = s_local.data();
  const float* h_prev = h_prev_local.data();
  const Agent::Decision d = agent_->selectAction(s);
  const int action = static_cast<int>(d.action);
  // Episode length: steps since the previous cheese, including this step.
  const int steps_before_cheese = sim_->stepsSinceLastCheese();

  // Phase 2: apply the action; the body (homeostasis) reacts to events.
  const Simulation::StepOutcome out = sim_->step(d.action);
  const float* s2 = sim_->observe();

  // Phase 3: world model + curiosity.
  const Homeostasis::Snapshot& ha = out.homeo_after;
  const Homeostasis::Snapshot& hb = out.homeo_before;
  const float homeo_targets[3] = {ha.energy - hb.energy, ha.hunger - hb.hunger,
                                  ha.stress - hb.stress};
  const double homeo_reward =
      -cfg_.homeo_reward_energy * (1.0 - ha.energy) -
      cfg_.homeo_reward_hunger * ha.hunger -
      cfg_.homeo_reward_fatigue * ha.fatigue -
      cfg_.homeo_reward_stress * ha.stress * ha.stress +
      cfg_.homeo_reward_satisfaction * (ha.satisfaction - 0.5);
  const float reward_ext =
      out.reward_nav + static_cast<float>(homeo_reward);
  agent_->computeIntrinsics(s, s2, homeo_targets, reward_ext);

  // Phase 4: compose the total reward (bounded, documented in README).
  const double curiosity =
      static_cast<double>(agent_->lastCuriosityReward()) *
      (0.5 + 0.5 * ha.curiosity_need);
  const double pred_bonus =
      cfg_.prediction_reward_gain * (1.0 - agent_->uncertainty());
  // Scent-proximity shaping: bonus proportional to the strongest scent
  // channel after the action (newest observation frame = current sensory
  // state). Derived only from the rat's own perception; rewards being near
  // the cheese and transfers across cheese re-placements.
  const int last_frame = (cfg_.observation_frames - 1) * kObservationBase;
  const float scent_after = std::max(
      {s2[last_frame + 8], s2[last_frame + 9], s2[last_frame + 10], s2[last_frame + 11]});
  const double scent_proximity =
      cfg_.scent_proximity_reward_gain * static_cast<double>(scent_after);
  double collapse = 0.0;
  if (metrics_.actionEntropy() < 0.2 && metrics_.recentAvgReward() < -0.05 &&
      sim_->lifetimeSteps() > 1000) {
    collapse = -cfg_.collapse_penalty;
  }
  const float r_total = static_cast<float>(
      clampd(static_cast<double>(reward_ext) + curiosity + pred_bonus +
                 scent_proximity + collapse,
             -12.0, 12.0));

  // Phase 5: learn + remember.
  agent_->observeAndTrain(s, h_prev, d.action, r_total, reward_ext, s2,
                          out.cheese_reached, homeo_targets, out.cheese_reached,
                          out.wall_hit, sim_->lifetimeSteps());

  // Feed learning signals back into the body + observation context.
  sim_->setNoveltySignal(agent_->lastNovelty());
  sim_->homeostasis().applyNovelty(agent_->lastNovelty());
  sim_->homeostasis().setUncertainty(agent_->uncertainty());

  // Metrics.
  metrics_.recordStep(
      r_total, out.wall_hit,
      /*revisit=*/sim_->lastRevisit(), action, prev_action_, d.explored, ha,
      agent_->lastNovelty(), agent_->lastCuriosityReward(), agent_->lastPredError());
  prev_action_ = action;
  if (out.cheese_reached) {
    metrics_.recordCheese(steps_before_cheese + 1);
    ++cheese_since_last_consolidation_;
    if (adaptation_remaining_ >= 0) {
      metrics_.recordAdaptation(adaptation_remaining_);
      adaptation_remaining_ = -1;
    }
  }
  if (out.maze_regenerated) {
    adaptation_remaining_ = 0;
    agent_->resetRecurrent();
  } else if (adaptation_remaining_ >= 0) {
    ++adaptation_remaining_;
  }

  // Keep a bounded ring of recent observations for plasticity evaluation.
  const int input = agent_->inputSize();
  std::copy(s2, s2 + input, recent_obs_.data() + recent_obs_head_ * input);
  recent_obs_head_ = (recent_obs_head_ + 1) % 64;

  // Consolidation trigger check.
  if (shouldConsolidate()) beginConsolidation();
}

bool Application::shouldConsolidate() const {
  if (consolidating_) return false;
  const uint64_t steps = sim_->lifetimeSteps();
  if (steps - last_consolidation_step_ < cfg_.consolidation_min_interval_steps)
    return false;
  if (cheese_since_last_consolidation_ >= cfg_.consolidation_every_cheeses)
    return true;
  if (sim_->homeostasis().fatigue() >= cfg_.consolidation_fatigue_threshold)
    return true;
  return false;
}

void Application::beginConsolidation() {
  consolidating_ = true;
  consolidation_steps_left_ = cfg_.consolidation_max_steps;
  consolidation_ops_left_ = cfg_.consolidation_max_train_ops;
  sim_->setResting(true);
  last_consolidation_step_ = sim_->lifetimeSteps();
  ++consolidation_cycles_;
  log_.info("consolidation begins (cycle " + std::to_string(consolidation_cycles_) + ")");
}

void Application::doConsolidationStep() {
  // Rest tick: the body rests (fatigue recovery), time passes, no movement.
  Homeostasis::Events ev;
  ev.resting = true;
  sim_->homeostasis().update(ev);
  sim_->setNoveltySignal(agent_->lastNovelty());

  // Bounded training ops, spread across ticks (no CPU spikes).
  if (consolidation_ops_left_ > 0 && (consolidation_steps_left_ % 2) == 0) {
    const int done = agent_->consolidationTrainOps(1);
    consolidation_ops_left_ -= done;
    sim_->homeostasis().setUncertainty(agent_->uncertainty());
  }

  // Metrics: resting steps still count as steps with zero reward.
  const Homeostasis::Snapshot hs = sim_->homeostasis().snapshot();
  metrics_.recordStep(0.0f, false, false, prev_action_, prev_action_, false, hs,
                      agent_->lastNovelty(), 0.0f, agent_->lastPredError());

  --consolidation_steps_left_;
  if (consolidation_steps_left_ <= 0 || consolidation_ops_left_ <= 0) {
    endConsolidation();
  }
}

void Application::endConsolidation() {
  consolidating_ = false;
  sim_->setResting(false);
  // Structural plasticity evaluation (bounded, rollback-protected).
  agent_->plasticityEvaluate(recent_obs_, cfg_.plasticity_eval_passes);
  agent_->resetRecurrent();
  cheese_since_last_consolidation_ = 0;
  log_.info("consolidation ends; ops=" +
            std::to_string(agent_->consolidationTrainOpsTotal()) +
            " accepted=" + std::to_string(agent_->structuralAccepted()) +
            " rejected=" + std::to_string(agent_->structuralRejected()));
  saveCheckpointNow("consolidation");
}

void Application::fillPanel(PanelData& p) {
  p.paused = paused_;
  p.consolidating = consolidating_;
  p.debug = debug_;
  p.fps = fps_;
  p.sim_steps_per_sec = sim_sps_;
  p.lifetime_steps = sim_->lifetimeSteps();
  p.cheese_total = sim_->cheeseTotal();
  p.maze_generations = sim_->mazeGenerations();
  p.checkpoints_saved = static_cast<uint32_t>(checkpoints_saved_);
  p.episode_count = metrics_.episodeCount();
  p.steps_since_cheese = sim_->stepsSinceLastCheese();
  p.epsilon = agent_->currentEpsilon();
  p.avg_reward = metrics_.recentAvgReward();
  p.wall_rate = metrics_.wallRatePer1000Steps();
  p.revisit_rate = metrics_.revisitRatePer1000Steps();
  p.mean_steps = metrics_.meanStepsPerCheese();
  p.median_steps = metrics_.medianStepsPerCheese();
  p.adaptation_steps = metrics_.meanAdaptationSteps();
  p.entropy = metrics_.actionEntropy();
  p.repeat_rate = metrics_.repeatedActionRate();
  p.exploration_rate = metrics_.explorationRate();
  p.pred_loss = metrics_.avgPredLoss();
  p.uncertainty = agent_->uncertainty();
  p.avg_novelty = metrics_.avgNovelty();
  p.curiosity_rate = metrics_.avgCuriosityReward();
  p.avg_energy = metrics_.avgEnergy();
  p.min_energy = metrics_.minEnergy();
  p.avg_hunger = metrics_.avgHunger();
  p.extreme_hunger = metrics_.extremeHungerFraction();
  p.avg_fatigue = metrics_.avgFatigue();
  p.avg_stress = metrics_.avgStress();
  p.stress_free_steps = sim_->homeostasis().stressFreeSteps();
  p.training_updates = agent_->trainingUpdates();
  p.invalid_updates = agent_->invalidUpdates();
  p.explored_total = agent_->exploredCount();
  p.episodic_used = agent_->episodicSize();
  p.episodic_cap = agent_->episodicCapacity();
  p.episodic_replacements = agent_->episodicReplacements();
  p.active_conn = agent_->activeConnections();
  p.dormant_conn = agent_->dormantConnections();
  p.pruned = agent_->prunedTotal();
  p.rewired = agent_->rewiredTotal();
  p.struct_ok = agent_->structuralAccepted();
  p.struct_rejected = agent_->structuralRejected();
  p.consolidation_cycles = consolidation_cycles_;
  p.consolidation_ops = agent_->consolidationTrainOpsTotal();
  p.runtime_s = (loaded_runtime_ms_ + nowMs() - session_start_ms_) / 1000;
}

}  // namespace sir
