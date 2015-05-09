#libs = -L /usr/local/Cellar/openblas/0.2.13/lib -lblas
libs = -L /opt/OpenBLAS/lib -lopenblas
#inc = -I /usr/local/Cellar/openblas/0.2.13/include 
inc = -I/opt/OpenBLAS/include
flags = -O3 -fopenmp

zefr:
	g++ src/zefr.cpp src/input.cpp src/geometry.cpp src/faces.cpp src/elements.cpp src/funcs.cpp src/quads.cpp src/points.cpp src/polynomials.cpp src/solver.cpp src/shape.cpp -I include $(inc) $(libs) -std=c++11 $(flags) -o bin/zefr
clean:
	rm bin/zefr 
