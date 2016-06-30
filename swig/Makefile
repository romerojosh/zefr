CXX = g++
AR = ar -rvs
CU = nvcc
SWIG = swig -c++ -python
PYTHON_INC = /usr/include/python2.7
TARGET = _zefr.so

SWIG_INC = -I$(PYTHON_INC)
SWIG_LIB = libzefr.so

$(TARGET): zefr_wrap.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INC) -fPIC -shared -o $@ $< $(LIBS) -Wl,-rpath=/home/jcrabill/zefr/swig -L. -lzefr 

%_wrap.cpp: %.i
	$(SWIG) $(INCS) -o $@ $<

%.o: %.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INC) -fPIC -o $@ $< $(LIBS)
