// Frozen-weight evaluation on fresh maze seeds. No updates, replay sampling,
// consolidation, privileged coordinates, or planning during evaluation.
#include "learning/agent.h"
#include "persistence/checkpoint.h"
#include "simulation/simulation.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace sir;
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: sir_eval CONFIG [steps=5000] [first_seed=1001] [seeds=10]\n";
    return 2;
  }
  Config cfg; std::string error,warnings;
  if (cfg.loadFromFile(argv[1],&error,&warnings)<0) { std::cerr<<error<<'\n'; return 2; }
  Logger log; AgentState learned;
  CheckpointStore store(cfg,log);
  if (store.load(&learned)!=LoadResult::Ok) return 1;
  const int steps=argc>2?std::max(1,std::atoi(argv[2])):5000;
  const int first=argc>3?std::max(1,std::atoi(argv[3])):1001;
  const int seeds=argc>4?std::clamp(std::atoi(argv[4]),1,1000):10;
  // The memory-augmented modes share exactly the same exploration controls.
  cfg.exploration_start=cfg.exploration_end=.05;
  std::cout<<"mode,seed,steps,cheese,wall_hits\n";
  for (const std::string mode : {"random_legal","untrained_memory","trained_memory","trained_greedy"}) {
    for (int seed=first;seed<first+seeds;++seed) {
      Rng env_rng(seed), policy_rng(seed+100000);
      Simulation sim(cfg,env_rng); Agent agent(cfg,policy_rng);
      if (mode.starts_with("trained") && !agent.importState(learned)) return 1;
      AgentState before,after; agent.exportState(before);
      int walls=0;
      for(int t=0;t<steps;++t) {
        const float* obs=sim.observe();
        Action action;
        if (mode=="random_legal") {
          constexpr int wall[4]={0,4,6,2}; int legal[4],n=0;
          const int frame=(cfg.observation_frames-1)*kObservationBase;
          for(int k=0;k<4;++k) if(obs[frame+wall[k]]<.5f) legal[n++]=k;
          action=static_cast<Action>(n?legal[policy_rng.uniformInt(0,n-1)]:0);
        } else action=agent.selectAction(obs,mode!="trained_greedy").action;
        const auto outcome=sim.step(action);
        walls+=outcome.wall_hit;
        if(outcome.cheese_reached || outcome.maze_regenerated) agent.resetRecurrent();
      }
      agent.exportState(after);
      if(before.online_params!=after.online_params || before.adam_m!=after.adam_m ||
         before.training_steps!=after.training_steps) return 3;
      std::cout<<mode<<','<<seed<<','<<steps<<','<<sim.cheeseTotal()<<','<<walls<<'\n';
    }
  }
}
