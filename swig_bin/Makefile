ifeq ($(strip $(SWIG_BIN)),)
SWIG = swig -c++ -python
else
SWIG = $(SWIG_BIN)/swig -c++ -python
endif
TARGET = _zefr.so

SWIG_OBJS = zefr_wrap.o
SWIG_INCS = -I$(strip $(PYTHON_INC_DIR))/ -I$(strip $(MPI4PY_INC_DIR))/
SWIG_LIBS = 

$(TARGET): $(SOBJS) $(SWIG_OBJS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INCS) -shared -o $@ $(SOBJS) $(SWIG_OBJS) $(LIBS) $(SWIG_LIBS)

%_wrap.cpp: %.i
	$(SWIG) $(FLAGS) $(INCS) $(SWIG_INCS) -o $@ $<

%.o: %.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INCS) -c -o $@ $< 

clean:
	rm -f *.o *.so
