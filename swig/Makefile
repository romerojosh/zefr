CXX = g++
CU = nvcc
SWIG = swig -c++ -python
TARGET = _zefr.so

SWIG_OBJS = zefr_wrap.o
SWIG_INCS = -I/usr/include/python2.7
SWIG_LIBS = 

$(TARGET): $(OBJS) $(SWIG_OBJS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INCS) -shared -o $@ $(OBJS) $(SWIG_OBJS) $(LIBS) $(SWIG_LIBS)

%_wrap.cpp: %.i
	$(SWIG) $(INCS) -o $@ $<

%.o: %.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INCS) -c -o $@ $< 
