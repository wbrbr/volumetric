all:
	clang++ src/*.cpp src/*.c -o raytracer -Iinclude/ -lglfw -lGL -ldl -g -O3 -Wall
