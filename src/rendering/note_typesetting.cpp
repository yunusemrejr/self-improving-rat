#include "rendering/note_typesetting.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_map>

namespace sir {

const NoteGlyph* noteGlyph(char32_t c) {
  static const std::unordered_map<char32_t, NoteGlyph> glyphs = {
    {' ',{0}}, {'A',{14,17,17,31,17,17,17}}, {'B',{30,17,17,30,17,17,30}},
    {'C',{14,17,16,16,16,17,14}}, {'D',{30,17,17,17,17,17,30}},
    {'E',{31,16,16,30,16,16,31}}, {'F',{31,16,16,30,16,16,16}},
    {'G',{14,17,16,23,17,17,15}}, {'H',{17,17,17,31,17,17,17}},
    {'I',{14,4,4,4,4,4,14}}, {'J',{7,2,2,2,18,18,12}},
    {'K',{17,18,20,24,20,18,17}}, {'L',{16,16,16,16,16,16,31}},
    {'M',{17,27,21,21,17,17,17}}, {'N',{17,25,21,19,17,17,17}},
    {'O',{14,17,17,17,17,17,14}}, {'P',{30,17,17,30,16,16,16}},
    {'Q',{14,17,17,17,21,18,13}}, {'R',{30,17,17,30,20,18,17}},
    {'S',{15,16,16,14,1,1,30}}, {'T',{31,4,4,4,4,4,4}},
    {'U',{17,17,17,17,17,17,14}}, {'V',{17,17,17,17,17,10,4}},
    {'W',{17,17,17,21,21,21,10}}, {'X',{17,17,10,4,10,17,17}},
    {'Y',{17,17,10,4,4,4,4}}, {'Z',{31,1,2,4,8,16,31}},
    {'a',{0,0,14,1,15,17,15}}, {'b',{16,16,30,17,17,17,30}},
    {'c',{0,0,14,17,16,17,14}}, {'d',{1,1,15,17,17,17,15}},
    {'e',{0,0,14,17,31,16,14}}, {'f',{6,9,8,28,8,8,8}},
    {'g',{0,0,15,17,17,17,15,1,14}}, {'h',{16,16,30,17,17,17,17}},
    {'i',{4,0,12,4,4,4,14}}, {'j',{2,0,6,2,2,2,2,18,12}},
    {'k',{16,16,18,20,24,20,18}}, {'l',{12,4,4,4,4,4,14}},
    {'m',{0,0,26,21,21,21,21}}, {'n',{0,0,30,17,17,17,17}},
    {'o',{0,0,14,17,17,17,14}}, {'p',{0,0,30,17,17,17,30,16,16}},
    {'q',{0,0,15,17,17,17,15,1,1}}, {'r',{0,0,22,25,16,16,16}},
    {'s',{0,0,15,16,14,1,30}}, {'t',{8,8,28,8,8,9,6}},
    {'u',{0,0,17,17,17,19,13}}, {'v',{0,0,17,17,17,10,4}},
    {'w',{0,0,17,17,21,21,10}}, {'x',{0,0,17,10,4,10,17}},
    {'y',{0,0,17,17,17,17,15,1,14}}, {'z',{0,0,31,2,4,8,31}},
    {'0',{14,17,19,21,25,17,14}}, {'1',{4,12,4,4,4,4,14}},
    {'2',{14,17,1,2,4,8,31}}, {'3',{30,1,1,14,1,1,30}},
    {'4',{2,6,10,18,31,2,2}}, {'5',{31,16,16,30,1,1,30}},
    {'6',{14,16,16,30,17,17,14}}, {'7',{31,1,2,4,8,8,8}},
    {'8',{14,17,17,14,17,17,14}}, {'9',{14,17,17,15,1,1,14}},
    {'.',{0,0,0,0,0,12,12}}, {',',{0,0,0,0,0,12,4,8}},
    {':',{0,12,12,0,12,12,0}}, {';',{0,12,12,0,12,4,8}},
    {'!',{4,4,4,4,4,0,4}}, {'?',{14,17,1,2,4,0,4}},
    {'\'',{4,4,8}}, {'"',{10,10,10}}, {'-',{0,0,0,31}},
    {'+',{0,4,4,31,4,4}}, {'=',{0,0,31,0,31}},
    {'/',{1,2,2,4,8,8,16}}, {'%',{17,2,4,8,17}},
    {'(',{2,4,8,8,8,4,2}}, {')',{8,4,2,2,2,4,8}},
    {'[',{14,8,8,8,8,8,14}}, {']',{14,2,2,2,2,2,14}},
    {'|',{4,4,4,4,4,4,4}}, {'_',{0,0,0,0,0,0,31}},
    {'<',{0,2,4,8,4,2}}, {'>',{0,8,4,2,4,8}},
    {U'α',{0,0,13,18,18,18,13}}, {U'β',{6,9,9,14,9,9,14,8,8}},
    {U'γ',{0,0,17,17,10,4,4,4}}, {U'δ',{6,8,4,14,17,17,14}},
    {U'ε',{0,0,14,16,28,16,14}}, {U'η',{0,0,22,25,17,17,17,1,1}},
    {U'θ',{14,17,17,31,17,17,14}}, {U'τ',{0,0,31,4,4,4,2}},
    {U'σ',{0,0,15,18,17,17,14}}, {U'Φ',{4,14,21,21,21,14,4}},
    {U'Σ',{31,16,8,4,8,16,31}}, {U'≤',{2,4,8,4,2,0,31}},
  };
  const auto it = glyphs.find(c);
  return it == glyphs.end() ? nullptr : &it->second;
}

namespace {
void place(MathLayout& into, const MathLayout& part, int x, int y) {
  for (auto mark : part.marks) {
    mark.x += x; mark.y += y; mark.x2 += x; mark.y2 += y;
    into.marks.push_back(mark);
  }
}
MathLayout glyph(char32_t c, int scale) {
  if (!noteGlyph(c)) throw std::invalid_argument("Unsupported math glyph");
  return {6*scale, 9*scale, 7*scale, {{c,0,0,0,0,scale}}};
}
MathLayout join(const MathLayout& a, const MathLayout& b) {
  MathLayout result;
  result.width = a.width + b.width;
  result.baseline = std::max(a.baseline,b.baseline);
  result.height = result.baseline + std::max(a.height-a.baseline,b.height-b.baseline);
  place(result,a,0,result.baseline-a.baseline);
  place(result,b,a.width,result.baseline-b.baseline);
  return result;
}
class Parser {
 public:
  explicit Parser(const std::string& input) : input_(input) {}
  MathLayout parse(int scale) {
    auto result = expression(scale,false);
    if (pos_ != input_.size()) fail();
    return result;
  }
 private:
  const std::string& input_;
  size_t pos_ = 0;
  int depth_ = 0;
  [[noreturn]] void fail() const { throw std::invalid_argument("Malformed or unsupported note LaTeX"); }
  MathLayout expression(int scale, bool group) {
    if (++depth_ > 20) fail();
    MathLayout result;
    while (pos_ < input_.size() && input_[pos_] != '}') {
      auto base = atom(scale);
      MathLayout sub, sup;
      bool has_sub = false, has_sup = false;
      while (pos_<input_.size() && (input_[pos_]=='_' || input_[pos_]=='^')) {
        const bool up = input_[pos_++] == '^';
        if ((up && has_sup) || (!up && has_sub)) fail();
        auto script = atom(std::max(1,scale-1));
        if (up) { sup=script; has_sup=true; } else { sub=script; has_sub=true; }
      }
      if (has_sub || has_sup) {
        const int lift = has_sup ? sup.height-2*scale : 0;
        MathLayout decorated;
        decorated.width = base.width + std::max(sub.width,sup.width);
        decorated.baseline = lift+base.baseline;
        decorated.height = std::max(lift+base.height,has_sub ? lift+base.baseline+sub.height-scale : 0);
        place(decorated,base,0,lift);
        if (has_sup) place(decorated,sup,base.width,0);
        if (has_sub) place(decorated,sub,base.width,lift+base.baseline-scale);
        base = decorated;
      }
      result = join(result,base);
    }
    if (group) {
      if (pos_ == input_.size() || input_[pos_] != '}') fail();
      ++pos_;
    }
    --depth_;
    return result;
  }
  MathLayout requiredGroup(int scale) {
    if (pos_==input_.size() || input_[pos_]!='{') fail();
    ++pos_;
    auto result=expression(scale,true);
    if (result.width==0) fail();
    return result;
  }
  MathLayout atom(int scale) {
    if (pos_ == input_.size()) fail();
    char c = input_[pos_++];
    if (c=='{') return expression(scale,true);
    if (c=='}' || c=='_' || c=='^') fail();
    if (c!='\\') return glyph(static_cast<unsigned char>(c),scale);
    const size_t start=pos_;
    while (pos_<input_.size() && std::isalpha(static_cast<unsigned char>(input_[pos_]))) ++pos_;
    const std::string command=input_.substr(start,pos_-start);
    if (command=="mathrm") return requiredGroup(scale);
    if (command=="frac") {
      const auto numerator=requiredGroup(scale), denominator=requiredGroup(scale);
      MathLayout result;
      result.width=std::max(numerator.width,denominator.width)+4*scale;
      const int bar=numerator.height+scale;
      result.baseline=bar+3*scale;
      result.height=bar+2*scale+denominator.height;
      place(result,numerator,(result.width-numerator.width)/2,0);
      place(result,denominator,(result.width-denominator.width)/2,bar+2*scale);
      result.marks.push_back({0,scale,bar,result.width-scale,bar,scale});
      return result;
    }
    if (command=="sqrt") {
      const auto inside=requiredGroup(scale);
      MathLayout result{inside.width+6*scale,inside.height+2*scale,inside.baseline+2*scale,{}};
      place(result,inside,5*scale,2*scale);
      result.marks.push_back({0,0,result.height/2,scale,result.height/2,scale});
      result.marks.push_back({0,scale,result.height/2,2*scale,result.height-scale,scale});
      result.marks.push_back({0,2*scale,result.height-scale,4*scale,0,scale});
      result.marks.push_back({0,4*scale,0,result.width-scale,0,scale});
      return result;
    }
    if (command=="log" || command=="max") {
      MathLayout result;
      for (char letter:command) result=join(result,glyph(letter,scale));
      return result;
    }
    static const std::unordered_map<std::string,char32_t> symbols={
      {"alpha",U'α'},{"beta",U'β'},{"gamma",U'γ'},{"delta",U'δ'},
      {"epsilon",U'ε'},{"eta",U'η'},{"theta",U'θ'},{"tau",U'τ'},
      {"sigma",U'σ'},{"Phi",U'Φ'},{"sum",U'Σ'},{"le",U'≤'}
    };
    const auto it=symbols.find(command);
    if (it==symbols.end()) fail();
    return glyph(it->second,scale);
  }
};
} // namespace

MathLayout typesetNoteMath(const std::string& latex, int scale) {
  if (scale<1 || scale>4 || latex.size()>2048) throw std::invalid_argument("Invalid note math size");
  return Parser(latex).parse(scale);
}
} // namespace sir
