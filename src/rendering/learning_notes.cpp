#include "rendering/learning_notes.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace sir {
namespace {
std::string number(double value, int digits = 2) {
  std::ostringstream s; s << std::fixed << std::setprecision(digits) << value;
  return s.str();
}
std::string probabilityText(double p) {
  // Never round an uncertain event to "certain", or a small chance to zero.
  if (p>0 && p<0.001) return "less than 0.1%";
  if (p<1 && p>0.999) return "more than 99.9%";
  return number(100*p,1)+"%";
}
std::string tenChoiceChance(double epsilon) {
  if (epsilon==0 || epsilon==1) return probabilityText(epsilon);
  const double p=-std::expm1(10*std::log1p(-epsilon));
  return probabilityText(std::min(p,std::nextafter(1.0,0.0)));
}
constexpr size_t kCount = 26;
}

size_t learningNoteCount() { return kCount; }

LearningNote learningNote(size_t i, const Config& c, double epsilon) {
  const std::string method = "HOW IT LEARNS";
  const std::string possibility = "POSSIBLE BEHAVIOR - NOT A FORECAST";
  switch (i) {
    case 0: return {"A score is not a promise", method,
      "The rat gives each move a Q value: a guess at the reward that move could lead to. A bigger score ranks a move higher; it is not a success percentage.",
      R"(Q(s,a))",
      "Q names the score. s is the sensed situation, including working memory; a is one action, such as left. The parentheses mean: find the score for this particular situation and move.",
      "src/learning/agent.cpp: selectAction, greedyAction"};
    case 1: return {"The coin flip behind exploration", "PROBABILITY AT NOTE OPEN",
      "When this note opened, the chance of entering random exploration was " + probabilityText(epsilon) +
      (c.episodic_action_bonus>0 ? ". The score-based branch adds a memory bonus; random moves favor less-tried directions." : ". The score-based branch uses Q values alone; random moves are equally weighted.") +
      " A random choice can still pick the highest-scoring move.",
      R"(P(\mathrm{random}) = \epsilon)",
      "P means probability, from 0 (never) to 1 (certain). The curved e is epsilon, the exploration rate. The equals sign says these are the same number. This is not the chance of finding cheese.",
      "src/learning/agent.cpp: computeEpsilon, selectAction"};
    case 2: return {"Small chances add up", "CONDITIONAL EXAMPLE",
      "If the opening exploration rate stayed fixed for ten independent choices, the chance of at least one random-mode choice would be " + tenChoiceChance(epsilon) + ". Real rates can change.",
      R"(P(A) = 1 - (1-\epsilon)^{10})",
      "A means at least one random choice; P is its chance. Epsilon is the random-choice rate. 1-epsilon is the chance of no random choice. The raised 10 means multiply that chance ten times; 1 minus it counts all the other cases.",
      "src/learning/agent.cpp: epsilon Bernoulli branch; conditional probability identity"};
    case 3: return {"Future rewards count a little less", method,
      "Discounting makes distant rewards count less in today's score. With your gamma of " + number(c.discount_factor) + ", a reward of 10 after ten discount steps contributes " + number(10*std::pow(c.discount_factor,10)) + ". This is an example, not a prediction.",
      R"(\mathrm{value} = \gamma^{k} r)",
      "Gamma, the y-like letter, is the discount factor. k counts steps into the future. r is the reward. The raised k means repeated multiplication. Writing terms side by side means multiply them.",
      "src/learning/agent.cpp: trainBatchOn"};
    case 4: return {"Learning from a better guess", method,
      "For a nonterminal one-step update, Double-DQN combines the reward just received with a prediction of what comes next. The online network chooses the next action; the target network scores it.",
      R"(y = r + \gamma Q_{target}(s',a_{best}))",
      "y is the training target; r is the received reward; gamma discounts the future. Q_target is the slower network's score. s-prime means the next observation. a_best is the online network's highest-scoring allowed next move.",
      "src/learning/agent.cpp: trainBatchOn; one-step special case"};
    case 5: return {"A surprise becomes an error signal", method,
      "Temporal-difference error measures the gap between a training target and the current Q score. Its size helps decide which experiences deserve more replay; its sign tells learning which way to adjust.",
      R"(\delta = y - Q(s,a))",
      "Delta, the curled d-like letter, is the error. y is the target; Q(s,a) is the current score for senses s and action a. Positive means the target is higher. The code also uses the opposite sign when taking loss gradients.",
      "src/learning/agent.cpp: TD error, Huber gradient, priority update"};
    case 6: return {"Some memories get more practice", method,
      "Prioritized replay revisits surprising experiences more often. The current priority exponent is " + number(c.per_priority_alpha) + ". Smaller exponents flatten the preference; zero makes stored experiences equally weighted.",
      R"(P(i) = \frac{p_i^{\alpha}}{\sum_j p_j^{\alpha}})",
      "P(i) is the sampling weight expressed as a probability for memory i. p is its priority, usually a clipped error size; new memories start high. Alpha sets how strongly priority matters. The large Sigma means add over all memories j; the fraction divides one weight by their total.",
      "src/learning/replay_buffer.cpp: updatePriority, setPriority, sampleIndices; batch sampling mixture"};
    case 7: return {"Ten times the error is not ten times the replay", "CONDITIONAL EXAMPLE",
      "For example priorities of 10 and 1, your alpha gives a sampling-weight ratio of " + number(std::pow(10,c.per_priority_alpha)) + ". Tempering priorities helps one surprising experience avoid monopolizing practice.",
      R"(\frac{P(i)}{P(j)} = (\frac{p_i}{p_j})^{\alpha})",
      "i and j label two memories. P is each memory's probability in the replay distribution; p is its stored priority. A fraction is division. The raised alpha is an exponent: it changes how sharply unequal priorities become unequal weights.",
      "src/learning/replay_buffer.cpp: setPriority"};
    case 8: return {"Correcting the replay imbalance", method,
      c.per_is_beta > 0 ? "Frequently sampled memories get less weight per update. This importance-sampling correction balances extra practice; its beta exponent moves toward 1 as the exploration schedule progresses." : "Importance-sampling correction exists here, but your beta setting is zero, so it is disabled. Every sampled transition gets the same correction weight.",
      R"(w_i = (N P(i))^{-\beta})",
      "w_i is the correction weight of memory i. N is the number stored; P(i) is its replay probability. Beta controls the correction. The negative exponent makes larger sampling probabilities receive smaller weights. The batch then scales weights so its largest is 1.",
      "src/learning/agent.cpp: observeAndTrain, importance weights"};
    case 9: return {"Memory inside the network", method,
      "A GRU is a gated recurrent unit: it carries a small working memory from one observation to the next. For one memory component, a gate blends the old value with a new candidate.",
      R"(h_{new} = (1-z)h_{old} + z c)",
      "h is a memory value; the small words below it label old and new. c is a candidate replacement. z is a gate between 0 and 1: near 0 keeps the old value, near 1 uses the candidate. Adjacent terms are multiplied.",
      "src/learning/gru.cpp: forward"};
    case 10: return {"How a gate stays between zero and one", method,
      "Sigmoid turns any input into a value between zero and one. GRU gates use this to mix memory smoothly; the prediction head uses it to keep its outputs bounded.",
      R"(\sigma(x) = \frac{1}{1+e^{-x}})",
      "Sigma, the rounded Greek letter, names the sigmoid function. x is its input. e is the number about 2.718, not an error or energy value here. The raised -x means an exponential; the fraction divides 1 by the whole expression below it.",
      "src/learning/gru.cpp: sigmoid; src/learning/neural_net.cpp: forward"};
    case 11: return {"Practice can reach backward through time", method,
      (c.sequence_train_interval>0 && c.bptt_chunk_len>=2 ? "Online sequence training is enabled: up to " + std::to_string(c.bptt_chunk_len) + " connected steps per chunk, every " + std::to_string(c.sequence_train_interval) + " steps. Backpropagation through time adjusts earlier memory operations using later errors." : "Online sequence training is disabled by your cadence or chunk setting. The implementation can train connected sequences, so later errors can adjust earlier memory operations."),
      R"(h_t = \mathrm{GRU}(x_t,h_{t-1}))",
      "t labels a time step. x_t is the observation at that step; h_t is the updated memory. t-1 means the previous step. GRU names the memory-update rule. Chunks and recorded boundaries limit how far gradients travel.",
      "src/learning/agent.cpp: trainSequenceBatch, observeAndTrain"};
    case 12: return {"Why replay warms up its memory", method,
      "Old experiences were collected by older network weights. Burn-in rereads up to " + std::to_string(c.replay_burn_in) + " preceding transitions before a sequence update, rebuilding context with today's weights. Zero disables that warm-up.",
      "",
      "Burn-in does not train on those warm-up steps. It prepares the recurrent state used by the training chunk. Both networks rebuild their own context. Stored starting states can still be stale; this reduces the problem rather than eliminating it.",
      "src/learning/agent.cpp: burnIn"};
    case 13: return {"Why familiar situations lose their shine", method,
      "One part of curiosity counts coarse observation codes. The count-based novelty component shrinks as a code repeats, before it is blended with prediction error. This is not a count of unique maze cells.",
      R"(\mathrm{novelty} = \frac{1}{\sqrt{n}})",
      "n is the stored visit count for that code. The hooked line is a square root: the number that multiplies by itself to make n. At n=4, the root is 2, so 1 divided by 2 gives 0.5. The bounded count table may later compact or saturate.",
      "src/organism/novelty.cpp: observe, compact; src/learning/agent.cpp: computeIntrinsics"};
    case 14: return {"Giving neglected moves another chance", method,
      c.episodic_action_bonus>0 ? "A UCB-style bonus raises the appeal of less-tried moves at the rat's own position. Repeated visits raise the bonus for moves still neglected, encouraging the rat to challenge a stale favorite." : "Your episodic action bonus is disabled. When enabled, this UCB-style term raises the appeal of less-tried moves at the rat's own position; it is not a promise of finding food.",
      R"(b = c \sqrt{\frac{2\log(2+v)}{1+n}})",
      "b is the bonus; c is its configured strength. v counts all attempts here; n counts attempts of this move. log is the natural logarithm, which grows slowly. The square root softens the result. When enabled, b is added to locally scaled Q scores, not to reward.",
      "src/learning/agent.cpp: selectAction, UCB scores"};
    case 15: return {"A sense of smell, without reward farming", method,
      "Potential-based shaping rewards changes in scent value instead of paying the rat just for staying near cheese. The app uses only scent it can sense, and makes the next potential zero when cheese is collected.",
      R"(F = c (\gamma \Phi(s') - \Phi(s)))",
      "F is the extra reward; c is its strength. Gamma discounts the future. Capital Phi, the circle with a vertical stroke, is the sum of scent channels. s is now; s-prime is next. Subtraction compares them. Circling cannot farm a continuing scent bonus.",
      "include/learning/reward.h: scentPotentialReward"};
    case 16: return {"A gentler response to a huge error", method,
      "Huber loss grows quadratically for small errors and linearly for large ones. This keeps a very surprising transition from demanding an unlimited update. The small-error part shown here applies within your threshold of " + number(c.td_huber_delta) + ".",
      R"(L(e) = \frac{e^{2}}{2})",
      "L is the loss, a number training tries to reduce. e is prediction minus target, not the exponential constant on other notes. The raised 2 means e times itself. Divide by 2. Beyond the threshold, a linear branch replaces this formula.",
      "src/learning/agent.cpp: trainBatchOn, trainSequenceBatch"};
    case 17: return {"A slower network steadies the target", method,
      "Soft target updates make the target network follow the learning network gradually, instead of replacing it abruptly. Your mixing rate tau is " + number(c.target_update_tau,3) + ". Masks also keep dormant connections at zero.",
      R"(\theta_{target}' = (1-\tau)\theta_{target} + \tau\theta_{online})",
      "Theta, the oval with a crossbar, stands for weights. The labels say which network; the prime mark means updated. Tau is the mixing fraction. For each weight, the formula blends the old target value with the latest online value.",
      "src/learning/agent.cpp: softUpdateTarget"};
    case 18: return {"Keeping useful connections in mind", method,
      "A connection's utility trace remembers how much that weight has mattered in recent gradients. Keeping this moving average across updates gives structural plasticity evidence for choosing weak and useful connections.",
      R"(u' = d u + (1-d)|g w|)",
      "u is utility and u-prime is its new value. d is the old-history fraction. g is the gradient, the loss's local sensitivity; w is the connection weight. Vertical bars mean absolute value: ignore the sign. This is a heuristic, not proof a connection is essential.",
      "src/learning/neural_net.cpp: updateUtility; src/learning/agent.cpp: plasticityEvaluate"};
    case 19: return {"Learning slows with age, but keeps going", method,
      "The developmental schedule scales the configured learning rate as lifetime steps accumulate. A built-in floor keeps at least one tenth of the base rate. This enables continued adaptation; it cannot guarantee improvement.",
      R"(\eta_{used} = \eta_{base}\max(0.1,1-c A))",
      "Eta, the n-like Greek letter, is the learning rate. A is age normalized from 0 to 1; c is the age-slowdown setting. max means choose the larger number, so 0.1 is a floor. Multiplication scales the base rate; Adam then uses its moment estimates.",
      "src/organism/development.cpp: learningRateScale; src/learning/agent.cpp: adamUpdate"};
    case 20: return {"What survives closing the window", method,
      "Checkpoints retain both networks, optimizer history, significant experiences and recent replay. Restarting starts a new maze, but preserves learned parameters. Replay snapshots have an 8 MiB budget; they are not infinite diaries.",
      "",
      "Optimizer history helps the next update continue smoothly. Replay means stored transitions that can be practiced again. Transient working memory and position-action counts reset on restart. A checksum detects file damage; backups help recovery.",
      "src/persistence/checkpoint.cpp; src/learning/agent.cpp: exportState, importState"};
    case 21: return {"Could a shortcut emerge?", possibility,
      "Yes: a learned preference might favor a useful detour or a shorter route. Braided mazes can offer alternatives. The app does not run a route planner inside the agent, and does not estimate a probability of discovering a shortcut.",
      "",
      "An emergent route would be a pattern produced by repeated observations, rewards and updates. To call it improvement, compare steps per cheese on matched maze seeds. An attractive single run cannot tell you how often that behavior will recur.",
      "src/simulation/maze.cpp: generate; src/learning/agent.cpp: selectAction; tools/evaluate.cpp"};
    case 22: return {"Why success here may fail elsewhere", possibility,
      "The rat can learn a habit that helps in one maze and fails in another. That is a generalization problem. This experiment's frozen trained policies still underperformed a legal random walk on the documented unseen-maze evaluation.",
      "",
      "Frozen means weights stay fixed while being evaluated. Unseen means those maze seeds were not the training seeds. A legal random walk picks a free direction at random. No reliable probability of future mastery has been established.",
      "tools/evaluate.cpp; docs/validation-20260909.md: Frozen weights on unseen mazes"};
    case 23: return {"Openings are random; mastery is not a coin toss", "GENERATOR PROBABILITY",
      "Your maze braiding setting is " + probabilityText(c.maze_braid_probability) + ". For each eligible dead end, this is the chance the generator attempts an extra opening. It can create loops and alternative routes.",
      R"(P(\mathrm{attempt}) = p_{braid})",
      "P means probability. p_braid is the configured braiding fraction; the small label names its purpose. An attempted opening still needs a valid neighboring wall to remove. This number is not the chance of solving the maze or learning a shortcut.",
      "src/simulation/maze.cpp: generate, braiding pass"};
    case 24: return {"Learning from food without pretending to feel", method,
      "Cheese restores the simulated energy store. Moving spends energy, while hunger and fatigue follow their own update rules. These body variables influence observations and reward; they do not establish feelings or consciousness.",
      R"(0 \le E \le 1)",
      "E names the energy variable. Each less-than-or-equal sign sets a bound: energy stays between 0 and 1, including both ends. A value of 0.8 means 80% of this simulated scale, not measured calories or a biological survival probability.",
      "src/organism/homeostasis.cpp: update; src/app/application.cpp: doSimStep"};
    case 25: return {"A reward can stop at an episode boundary", method,
      "Collecting cheese ends the current food-seeking episode. Its final target uses the reward just received, without guessing value beyond that terminal. Rest and manual maze resets instead truncate sequence context.",
      R"(y_{terminal} = r)",
      "y is the training target; the small word terminal labels an episode-ending step. r is the composite reward received. Equals means no future-value term is added. Nonterminal truncations can still keep a next-state value estimate.",
      "src/learning/agent.cpp: trainBatchOn, resetRecurrent; src/app/application.cpp: doSimStep"};
  }
  throw std::out_of_range("learning note index");
}

std::vector<std::string> wrapNoteText(const std::string& text, size_t columns) {
  columns = std::max<size_t>(1,columns);
  std::istringstream words(text); std::string word,line;
  std::vector<std::string> lines;
  while(words>>word) {
    if(!line.empty() && line.size()+1+word.size()>columns) { lines.push_back(line); line.clear(); }
    while(word.size()>columns) { if(!line.empty()) {lines.push_back(line);line.clear();} lines.push_back(word.substr(0,columns)); word.erase(0,columns); }
    if(!word.empty()) { if(!line.empty()) line+=' '; line+=word; }
  }
  if(!line.empty()) lines.push_back(line);
  return lines;
}
uint64_t noteReadingTime(const LearningNote& note) {
  std::istringstream words(note.explanation+" "+note.notation); std::string word; uint64_t n=0;
  while(words>>word) ++n;
  // About 150 words/minute, plus time to decode the formula.
  return std::clamp<uint64_t>(n*400+(note.latex.empty()?4000:9000),24000,60000);
}
LearningNoteDeck::LearningNoteDeck(uint32_t seed) : rng_(seed),order_(kCount) {
  std::iota(order_.begin(),order_.end(),0); std::shuffle(order_.begin(),order_.end(),rng_);
}
void LearningNoteDeck::start(uint64_t now,uint64_t duration) { deadline_=now+duration; held_remaining_=duration; }
bool LearningNoteDeck::due(uint64_t now) const { return !held_ && now>=deadline_; }
uint64_t LearningNoteDeck::remaining(uint64_t now) const { return held_?held_remaining_:(now<deadline_?deadline_-now:0); }
void LearningNoteDeck::toggleHold(uint64_t now) {
  if(held_) {held_=false;deadline_=now+held_remaining_;}
  else {held_remaining_=remaining(now);held_=true;}
}
void LearningNoteDeck::next() {
  const auto old=current();
  if(++cursor_==order_.size()) {
    cursor_=0;std::shuffle(order_.begin(),order_.end(),rng_);
    if(order_[0]==old) std::swap(order_[0],order_[1]);
  }
}
}  // namespace sir
