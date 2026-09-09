#include "test_framework.h"
#include "rendering/learning_notes.h"
#include "rendering/note_typesetting.h"
#include "utility/rng.h"

#include <algorithm>
#include <set>
#include <stdexcept>

using namespace sir;

TEST(notes_shuffle_without_repeats_or_training_randomness) {
  Rng training(42), control(42);
  LearningNoteDeck deck(9182), same(9182), other(1256);
  bool differs = false;
  size_t previous = learningNoteCount();
  for (int bag=0; bag<20; ++bag) {
    std::set<size_t> seen;
    for (size_t i=0; i<learningNoteCount(); ++i) {
      CHECK(deck.current() != previous);
      CHECK(deck.current() == same.current());
      differs |= deck.current() != other.current();
      seen.insert(deck.current());
      previous = deck.current();
      deck.next(); same.next(); other.next();
      CHECK(training.nextU32() == control.nextU32());
    }
    CHECK(seen.size() == learningNoteCount());
  }
  CHECK(differs);
}

TEST(notes_hold_freezes_remaining_reading_time) {
  LearningNoteDeck deck(42);
  deck.start(100,30000);
  CHECK(!deck.due(30099));
  CHECK(deck.due(30100));
  deck.toggleHold(10100);
  CHECK(deck.held());
  CHECK(deck.remaining(90000)==20000);
  CHECK(!deck.due(90000));
  deck.toggleHold(90000);
  CHECK(!deck.due(109999));
  CHECK(deck.due(110000));
  deck.toggleHold(100000);
  const auto previous=deck.current();
  deck.next(); deck.start(100000,42000);
  CHECK(deck.current()!=previous);
  CHECK(deck.held());
  CHECK(!deck.due(900000));
  CHECK(deck.remaining(900000)==42000);
  deck.toggleHold(900000);
  CHECK(!deck.due(941999));
  CHECK(deck.due(942000));
}

TEST(notes_probability_examples_follow_configuration) {
  Config cfg;
  CHECK(learningNote(2,cfg,0.1).explanation.find("65.1%")!=std::string::npos);
  CHECK(learningNote(2,cfg,0).explanation.find("0.0%")!=std::string::npos);
  CHECK(learningNote(2,cfg,1).explanation.find("100.0%")!=std::string::npos);
  CHECK(learningNote(2,cfg,0.9).explanation.find("more than 99.9%")!=std::string::npos);
  CHECK(learningNote(2,cfg,0.999999).explanation.find("more than 99.9%")!=std::string::npos);
  CHECK(learningNote(2,cfg,0.0000001).explanation.find("less than 0.1%")!=std::string::npos);
  CHECK(learningNote(1,cfg,0.123).explanation.find("12.3%")!=std::string::npos);
  cfg.discount_factor=0.5;
  CHECK(learningNote(3,cfg,0).explanation.find("0.01")!=std::string::npos);
  cfg.per_priority_alpha=0.5;
  CHECK(learningNote(7,cfg,0).explanation.find("3.16")!=std::string::npos);
  cfg.per_priority_alpha=0;
  CHECK(learningNote(7,cfg,0).explanation.find("1.00")!=std::string::npos);
  cfg.maze_braid_probability=0.25;
  CHECK(learningNote(23,cfg,0).explanation.find("25.0%")!=std::string::npos);
  cfg.per_is_beta=0; cfg.sequence_train_interval=0; cfg.episodic_action_bonus=0;
  CHECK(learningNote(1,cfg,0.1).explanation.find("equally weighted")!=std::string::npos);
  CHECK(learningNote(8,cfg,0).explanation.find("disabled")!=std::string::npos);
  CHECK(learningNote(11,cfg,0).explanation.find("disabled")!=std::string::npos);
  CHECK(learningNote(14,cfg,0).explanation.find("disabled")!=std::string::npos);
  cfg.bptt_chunk_len=2; cfg.sequence_train_interval=17; cfg.replay_burn_in=7;
  CHECK(learningNote(11,cfg,0).explanation.find("every 17 steps")!=std::string::npos);
  CHECK(learningNote(12,cfg,0).explanation.find("up to 7 preceding")!=std::string::npos);
  // A displayed live value is a snapshot, not text that shifts while reading.
  const auto snapshot=learningNote(1,cfg,0.1);
  learningNote(1,cfg,0.9);
  CHECK(snapshot.explanation.find("10.0%")!=std::string::npos);
}

TEST(notes_all_copy_and_formulas_fit_the_minimum_width) {
  Config cfg;
  for (size_t i=0; i<learningNoteCount(); ++i) {
    for (double epsilon : {0.0,0.000001,0.1,0.999999,1.0}) {
      const auto note=learningNote(i,cfg,epsilon);
      CHECK(!note.title.empty() && !note.explanation.empty() && !note.notation.empty());
      CHECK(!note.source.empty());
      CHECK(noteReadingTime(note)>=24000 && noteReadingTime(note)<=60000);
      for (const auto& text : {note.title,note.category,note.explanation,note.notation}) {
        for (unsigned char c:text) CHECK_MSG(noteGlyph(c)!=nullptr,text);
        const auto lines=wrapNoteText(text,46);
        for (const auto& line:lines) CHECK(line.size()<=46);
        std::string rebuilt;
        for (const auto& line:lines) { if (!rebuilt.empty()) rebuilt+=' '; rebuilt+=line; }
        CHECK(rebuilt==text); // no dropped words, symbols or punctuation
      }
      const auto math=typesetNoteMath(note.latex);
      CHECK(math.width<=560);
      CHECK(math.baseline>=0 && math.baseline<=math.height);
      for (const auto& mark:math.marks) {
        CHECK(mark.x>=0 && mark.y>=0);
        if (mark.glyph) {
          CHECK(noteGlyph(mark.glyph)!=nullptr);
          CHECK(mark.x+5*mark.scale<=math.width);
          CHECK(mark.y+9*mark.scale<=math.height);
        } else {
          CHECK(mark.x2>=0 && mark.x2<math.width);
          CHECK(mark.y2>=0 && mark.y2<math.height);
        }
      }
    }
  }
  const auto hard=wrapNoteText("abcdefgh ij",3);
  CHECK(hard==std::vector<std::string>({"abc","def","gh","ij"}));
  CHECK(wrapNoteText("",0).empty());
}

TEST(notes_typesetting_preserves_math_structure) {
  const auto power=typesetNoteMath("x^{2}");
  CHECK(power.marks[1].y<power.marks[0].y);
  const auto subscript=typesetNoteMath("x_{2}");
  CHECK(subscript.marks[1].y>subscript.marks[0].y);
  const auto fraction=typesetNoteMath(R"(\frac{1}{2})");
  CHECK(fraction.marks.size()==3);
  CHECK(fraction.marks[0].y<fraction.marks[2].y);
  CHECK(fraction.marks[2].y<fraction.marks[1].y);
  const auto letters=typesetNoteMath(R"(\epsilon\gamma\delta\alpha\beta\sigma\theta\tau\eta\Phi\sum\le)");
  const std::u32string expected=U"εγδαβσθτηΦΣ≤";
  CHECK(letters.marks.size()==expected.size());
  for (size_t i=0;i<expected.size();++i) CHECK(letters.marks[i].glyph==expected[i]);
  for (const std::string bad : {R"(\unknown)",R"(\frac{1})","x^","x}","{x","x^{2}^3"}) {
    bool rejected=false;
    try { typesetNoteMath(bad); } catch (const std::invalid_argument&) { rejected=true; }
    CHECK_MSG(rejected,bad);
  }
}
