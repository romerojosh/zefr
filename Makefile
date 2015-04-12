libs = -L /usr/local/Cellar/openblas/0.2.13/lib -lblas
inc = -I /usr/local/Cellar/openblas/0.2.13/include 
flags = -O3

zefr:
	g++ src/zefr.cpp src/input.cpp src/geometry.cpp src/faces.cpp src/elements.cpp src/quads.cpp src/points.cpp src/polynomials.cpp src/solver.cpp src/shape.cpp -I include $(inc) $(libs) -std=c++11 $(flags) -o bin/zefr
clean:
	rm bin/zefr 
