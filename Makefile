test:
	g++ test.cpp src/input.cpp src/elements.cpp src/quads.cpp src/points.cpp src/polynomials.cpp -I include -std=c++11 -o test
clean:
	rm test
