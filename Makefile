CFLAGS = -Iinclude/ -g -O3 -Wall -DIMGUI_IMPL_OPENGL_LOADER_GL3W
LDFLAGS = -lglfw -lGL -ldl -g -O3

all: gl3w.o imgui.o imgui_demo.o imgui_draw.o imgui_impl_glfw.o imgui_impl_opengl3.o imgui_widgets.o main.o
	clang++ $^ -o raytracer $(LDFLAGS)

%.o: src/%.c
	clang -c $< -o $@ $(CFLAGS)

%.o: src/%.cpp
	clang++ -c $< -o $@ $(CFLAGS)
