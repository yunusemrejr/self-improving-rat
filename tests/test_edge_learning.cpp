#include "test_framework.h"
#include "learning/agent.h"
#include "learning/neural_net.h"
#include "learning/replay_buffer.h"
#include <algorithm>
#include <limits>

using namespace sir;

TEST(replay_tree_wrap_restore_and_failure_are_exact) {
  ReplayBuffer replay(30, 16, 7, .6f);
  float s[30]{}, h[16]{}, ht[3]{};
  for (int i = 0; i < 20; ++i) {
    s[28] = i / 20.0f;
    replay.push(s, s, h, i % 4, float(i), float(i), i % 5 == 0, ht);
    replay.updatePriority(replay.size()-1, float(i+1));
  }
  replay.markBoundary();
  std::vector<float> values, copied;
  std::vector<uint32_t> meta, cm;
  replay.serializeTo(values, meta);
  ReplayBuffer restored(30, 16, 7, .6f);
  CHECK(restored.restoreFrom(values, meta));
  restored.serializeTo(copied, cm);
  CHECK(values == copied); CHECK(meta == cm);
  Rng r1(88), r2(88); std::vector<size_t> a,b; std::vector<float> wa,wb;
  replay.sampleIndices(r1, 10000, &a, &wa);
  restored.sampleIndices(r2, 10000, &b, &wb);
  // Restore canonicalizes ring storage; validate the same probability law.
  int counts_a[7]{}, counts_b[7]{};
  for(size_t i:a) ++counts_a[i];
  for(size_t i:b) ++counts_b[i];
  for(int i=0;i<7;++i) CHECK(std::abs(counts_a[i]-counts_b[i]) <= 3);
  for(size_t i=0;i<a.size();++i) CHECK(std::isfinite(wa[i]) && std::isfinite(wb[i]));
  for (size_t i:a) CHECK(i < 7);
  auto bad=meta; bad[0]=4;
  CHECK(!restored.restoreFrom(values,bad));
  restored.serializeTo(copied, cm); CHECK(copied==values); CHECK(cm==meta);
  ReplayBuffer small(30,16,3);
  CHECK(small.restoreFrom(values,meta)); CHECK(small.size()==3);
  CHECK_NEAR(small.reward(0),17,0); CHECK(small.boundary(2));
  replay.updatePriority(0,std::numeric_limits<float>::quiet_NaN());
  replay.sampleIndices(r1,16,&a,&wa);
  for(float w:wa) CHECK(std::isfinite(w) && w>0);
}

TEST(agent_masks_observed_walls_in_greedy_and_exploration_with_stacked_frames) {
  Config cfg; cfg.observation_frames=2; cfg.exploration_start=1; cfg.exploration_end=1;
  Rng rng(10); Agent agent(cfg,rng);
  float s[60]{}, q[4]={100,90,80,-10};
  // Only right is free in the newest frame, even though older frame is open.
  s[30]=s[34]=s[36]=1;
  CHECK(agent.greedyAction(s,q)==3);
  for(int i=0;i<1000;++i) CHECK(agent.selectAction(s).action==Action::Right);
  s[32]=1; CHECK(agent.greedyAction(s,q)==0); // all-blocked defensive fallback
}

TEST(agent_td_bootstrap_uses_post_observation_recurrence_and_masks) {
  Config cfg; cfg.batch_size=1; cfg.train_interval_steps=1; cfg.sequence_train_interval=0;
  cfg.exploration_start=0; cfg.exploration_end=0;
  Rng rng(12); Agent agent(cfg,rng); AgentState state; agent.exportState(state);
  Rng nr(4); NeuralNet online(30,16,4,16,nr), target(30,16,4,16,nr);
  online.setParams(state.online_params.data()); target.setParams(state.target_params.data());
  float s[30]{}, next[30]{}, hp[16], ht[3]{};
  for(int i=0;i<16;++i) hp[i]=.1f*(i%3);
  s[19]=.7f; s[9]=.4f; next[8]=.8f;
  next[4]=next[6]=next[2]=1; // only up legal for bootstrap
  float h[16], th[16], nh[16], q[4], nq[4], tq[4];
  online.forward(s,hp,h,q,nullptr); target.forward(s,hp,th,nullptr,nullptr);
  online.forward(next,h,nh,nq,nullptr); target.forward(next,th,nh,tq,nullptr);
  const float expected=std::max(.001f,std::fabs(q[3]-(.7f+float(cfg.discount_factor)*tq[0])));
  agent.observeAndTrain(s,hp,Action::Right,.7f,.7f,next,false,ht,false,false,1);
  agent.exportState(state);
  CHECK_NEAR(state.replay_floats.back(),expected,1e-6);
  CHECK(agent.invalidUpdates()==0);
}

TEST(agent_sequence_does_not_backpropagate_across_episode_boundary) {
  Config cfg; cfg.bptt_chunk_len=8; cfg.sequence_train_interval=0;
  Rng ra(4),rb(4); Agent a(cfg,ra), b(cfg,rb); AgentState state;
  a.exportState(state);
  ReplayBuffer replay(30,16,8); float obs[30]{},h[16]{},ht[3]{};
  for(int i=0;i<8;++i) replay.push(obs,obs,h,0,i==0?1:0,0,i==0,ht);
  replay.serializeTo(state.replay_floats,state.replay_meta);
  CHECK(a.importState(state));
  const size_t stride=30*2+16+6;
  for(size_t i=1;i<8;++i) state.replay_floats[i*stride+stride-2]=10; // future rewards
  CHECK(b.importState(state));
  CHECK(a.consolidationSequenceOps(1)==1); CHECK(b.consolidationSequenceOps(1)==1);
  AgentState sa,sb; a.exportState(sa); b.exportState(sb);
  CHECK(sa.online_params==sb.online_params); CHECK(sa.adam_m==sb.adam_m);
}

TEST(agent_replay_survives_restart_and_import_rejection_is_transactional) {
  Config cfg; Rng ra(5),rb(8); Agent a(cfg,ra), b(cfg,rb);
  float obs[30]{},h[16]{},ht[3]{};
  for(int i=0;i<64;++i) a.observeAndTrain(obs,h,Action::Right,.1f,.1f,obs,false,ht,false,false,i);
  AgentState saved; a.exportState(saved);
  CHECK(b.importState(saved)); CHECK(b.replaySize()==64);
  CHECK(b.consolidationTrainOps(1)==1); CHECK(b.invalidUpdates()==0);
  AgentState before,after; b.exportState(before);
  saved.replay_meta[0]=100;
  CHECK(!b.importState(saved)); b.exportState(after);
  CHECK(before.online_params==after.online_params); CHECK(before.replay_meta==after.replay_meta);
  saved=before; saved.adam_v[0]=-1;
  CHECK(!b.importState(saved));
}

TEST(plasticity_utility_retains_history_and_mask_blocks_regrowth) {
  Rng rng(5); NeuralNet net(30,16,4,16,rng);
  std::vector<float> grad(net.paramCount(),1), util(net.paramCount());
  net.updateUtility(grad.data(),.9); net.getUtility(util.data());
  float first=util[0]; CHECK(first>0);
  std::fill(grad.begin(),grad.end(),0); net.updateUtility(grad.data(),.9);
  net.getUtility(util.data()); CHECK_NEAR(util[0],first*.9,1e-7);
  net.zeroParam(0); std::fill(grad.begin(),grad.end(),1); net.maskDense(grad.data());
  CHECK(grad[0]==0); CHECK(grad[1]==1);
}

TEST(episodic_cheese_memories_survive_lower_significance_arrivals) {
  EpisodicMemory mem(30,16,2);
  EpisodicEntry good; good.s.resize(30); good.s2.resize(30); good.h.resize(16);
  good.significance=3; good.flags=kFlagCheese;
  mem.add(good); mem.add(good);
  auto weak=good; weak.significance=.8f; weak.flags=kFlagHighPredError;
  CHECK(!mem.add(weak)); CHECK(mem[0].flags==kFlagCheese); CHECK(mem[1].flags==kFlagCheese);
  std::vector<float> f; std::vector<uint32_t> m; mem.serializeTo(f,m);
  EpisodicMemory restored(30,16,2); m[1]=kFlagHighPredError;
  CHECK(restored.restoreFrom(f,m)); // all defined event flags are accepted
}

#include "learning/reward.h"

TEST(scent_reward_cannot_be_farmed_by_lingering_or_a_round_trip) {
  Config cfg; float a[30]{},b[30]{}; a[8]=.5; b[8]=.8;
  CHECK(scentPotentialReward(cfg,a,a,false)<0);
  const double there=scentPotentialReward(cfg,a,b,false);
  const double back=scentPotentialReward(cfg,b,a,false);
  const double expected=cfg.scent_proximity_reward_gain *
      (cfg.discount_factor*cfg.discount_factor-1)*a[8];
  CHECK_NEAR(there+cfg.discount_factor*back,expected,1e-7);
  CHECK_NEAR(scentPotentialReward(cfg,a,b,true),-cfg.scent_proximity_reward_gain*a[8],1e-7);
}

TEST(episodic_uncertainty_explores_despite_overconfident_q_values) {
  Config cfg; cfg.exploration_start=cfg.exploration_end=0;
  Rng rng(44); Agent agent(cfg,rng); AgentState state; agent.exportState(state);
  std::fill(state.online_params.begin(),state.online_params.end(),0);
  const size_t bias=GruLayer::paramCount(30,16)+16*4;
  state.online_params[bias]=1000; state.online_params[bias+1]=-1000;
  CHECK(agent.importState(state));
  float obs[30]{}; obs[2]=obs[6]=1; // up/down are the two legal choices
  int down=0;
  for(int i=0;i<150;++i) down+=agent.selectAction(obs).action==Action::Down;
  CHECK(down>0);
  CHECK(agent.selectAction(obs,false).action==Action::Up); // raw greedy eval
  AgentState after; agent.exportState(after);
  CHECK(after.online_params==state.online_params);
}
