// Real SDL software-renderer checks, isolated from the user's checkpoint.
#include "rendering/renderer.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
struct Snapshot { int width, height; std::vector<uint32_t> pixels; };
Snapshot snapshot(const sir::Renderer& renderer, const std::filesystem::path& path) {
  require(renderer.saveScreenshot(path.string()), "screenshot failed");
  auto* bmp=SDL_LoadBMP(path.c_str());
  require(bmp!=nullptr,"screenshot could not be read");
  auto* surface=SDL_ConvertSurfaceFormat(bmp,SDL_PIXELFORMAT_ARGB8888,0);
  SDL_FreeSurface(bmp);
  require(surface!=nullptr,"screenshot conversion failed");
  Snapshot s{surface->w,surface->h,{}};
  for (int y=0;y<s.height;++y) {
    const auto* row=reinterpret_cast<const uint32_t*>(
        static_cast<const char*>(surface->pixels)+y*surface->pitch);
    s.pixels.insert(s.pixels.end(),row,row+s.width);
  }
  SDL_FreeSurface(surface);
  return s;
}
SDL_Event key(SDL_Keycode code, bool repeat=false) {
  SDL_Event event{}; event.type=SDL_KEYDOWN;
  event.key.keysym.sym=code; event.key.repeat=repeat;
  return event;
}
SDL_Event click(int x,int y) {
  SDL_Event event{}; event.type=SDL_MOUSEBUTTONUP;
  event.button.button=SDL_BUTTON_LEFT; event.button.x=x; event.button.y=y;
  return event;
}
} // namespace

int main(int argc, char** argv) {
  const bool retain=argc>1;
  const auto output=retain ? std::filesystem::path(argv[1]) :
      std::filesystem::temp_directory_path()/(
          "sir-render-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    std::filesystem::create_directories(output);
    for (int width : {1060,640,320}) {
      sir::Config cfg; cfg.window_width=width;
      if (width<900) cfg.window_height=480;
      sir::Rng rng(42); sir::Simulation sim(cfg,rng);
      const auto rng_before=rng.saveState();
      sir::Renderer renderer; std::string error;
      require(renderer.init(cfg,&error),error.c_str());
      sir::PanelData panel; panel.paused=true; panel.epsilon=0.1;
      panel.avg_energy=sim.homeostasis().energy();
      panel.avg_hunger=sim.homeostasis().hunger();
      renderer.render(sim,panel);
      const auto file=output/(std::to_string(width)+"-current.bmp");
      const auto first=snapshot(renderer,file);
      require(first.width>=640,"minimum readable width not enforced");
      int divider=first.height-63;
      while (divider>0 && first.pixels[divider*first.width+28]!=0xff2a3537) --divider;
      require(divider>118,"missing learning-note divider");
      auto stable_habitat=[&](const Snapshot& s) {
        require(std::equal(first.pixels.begin(),first.pixels.begin()+divider*first.width,
                           s.pixels.begin()),"reading controls changed the habitat");
      };
      require(renderer.handleEvent(key(SDLK_h)),"hold key not handled");
      renderer.render(sim,panel);
      const auto held=snapshot(renderer,file); stable_habitat(held);
      require(first.pixels!=held.pixels,"hold state not visible");
      SDL_Delay(35);
      require(renderer.handleEvent(key(SDLK_h,true)),"repeat key not consumed");
      renderer.render(sim,panel);
      require(snapshot(renderer,file).pixels==held.pixels,"hold timer or repeat changed held note");
      auto previous=held;
      for (size_t i=0;i<sir::learningNoteCount();++i) {
        if (i) require(renderer.handleEvent(key(SDLK_n)),"next key not handled");
        renderer.render(sim,panel);
        const auto next_file=retain ? output/(std::to_string(width)+"-"+std::to_string(i)+".bmp") : file;
        auto next=snapshot(renderer,next_file); stable_habitat(next);
        if (i) require(next.pixels!=previous.pixels,"next did not change displayed explanation");
        previous=std::move(next);
      }
      require(renderer.handleEvent(click(first.width-210,divider+20)),"hold click not handled");
      renderer.render(sim,panel);
      require(snapshot(renderer,file).pixels!=previous.pixels,"mouse did not resume rotation");
      require(renderer.handleEvent(click(first.width-96,divider+20)),"next click not handled");
      renderer.render(sim,panel); stable_habitat(snapshot(renderer,file));
      require(!renderer.handleEvent(key(SDLK_SPACE)),"reader swallowed simulation pause");
      require(rng.saveState()==rng_before,"reader changed simulation RNG");
      renderer.shutdown();
    }
    if (!retain) std::filesystem::remove_all(output);
    std::cout<<"SDL: all notes, compact layouts, keyboard, mouse and paused reading passed\n";
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<"\nScreenshots: "<<output<<'\n';
    return 1;
  }
}
