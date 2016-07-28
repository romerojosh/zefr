TEMPLATE = app
CONFIG += console
CONFIG -= qt

DEFINES = _MPI _GPU
#DEFINES += _GPU

QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += $$PWD/include \
    /usr/lib/openmpi/include

SOURCES += \
    src/elements.cpp \
    src/faces.cpp \
    src/funcs.cpp \
    src/geometry.cpp \
    src/hexas.cpp \
    src/input.cpp \
    src/multigrid.cpp \
    src/points.cpp \
    src/polynomials.cpp \
    src/quads.cpp \
    src/solver.cpp \
    src/zefr.cpp \
    src/elements_kernels.cu \
    src/faces_kernels.cu \
    src/funcs_kernels.cu \
    src/solver_kernels.cu \
    src/zefr_interface.cpp \
    swig_bin/testZefr.cpp

HEADERS += \
    include/elements_kernels.h \
    include/elements.hpp \
    include/faces_kernels.h \
    include/faces.hpp \
    include/funcs.hpp \
    include/geometry.hpp \
    include/hexas.hpp \
    include/input.hpp \
    include/macros.hpp \
    include/mdvector_gpu.h \
    include/mdvector.hpp \
    include/multigrid.hpp \
    include/points.hpp \
    include/polynomials.hpp \
    include/quads.hpp \
    include/solver_kernels.h \
    include/solver.hpp \
    include/zefr.hpp \
    include/inputstruct.hpp \
    include/zefr_interface.hpp

DISTFILES += \
    swig_bin/zefr.i \
    swig_bin/test.py \
    swig_bin/Makefile \
    Makefile

