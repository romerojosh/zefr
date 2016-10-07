SWIG = /home/jcrabill/local/bin/swig -c++ -python
TARGET = _zefr.so

SWIG_OBJS = zefr_wrap.o
#SWIG_INCS = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/mpi4py/include/
SWIG_INCS = -I/home/jcrabill/Workspace/pyfr-venv/lib/python3.5/site-packages/mpi4py/include -I/home/jcrabill/local/include/python3.5m
SWIG_LIBS = 

$(TARGET): $(SOBJS) $(SWIG_OBJS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INCS) -shared -o $@ $(SOBJS) $(SWIG_OBJS) $(LIBS) $(SWIG_LIBS)

%_wrap.cpp: %.i
	$(SWIG) $(FLAGS) $(INCS) $(SWIG_INCS) -o $@ $<

%.o: %.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INCS) -c -o $@ $< 

clean:
	rm -f *.o *.so
