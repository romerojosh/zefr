TEMPLATE = app
CONFIG += console
CONFIG -= qt

DEFINES = _MPI
DEFINES += _CPU
DEFINES += _GPU
DEFINES += _BUILD_LIB

QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += $$PWD/include \
    /usr/lib/openmpi/include \
    /opt/OpenMPI-3.1.0/include \
    /usr/include/hdf5/serial

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
    src/filter_kernels.cu \
    src/aux_kernels.cu \
    src/gimmik_gpu.cu \
    src/gimmik_cpu.c \
    src/zefr_interface.cpp \
    src/zefrWrap.cpp \
    src/filter.cpp \
    src/tris.cpp \
    src/tets.cpp \
    src/prisms.cpp \
    include/global.i

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
    include/aux_kernels.h \
    include/solver.hpp \
    include/zefr.hpp \
    include/inputstruct.hpp \
    include/zefr_interface.hpp \
    include/inputstruct.hpp \
    include/filter_kernels.h \
    include/flux.hpp \
    include/tris.hpp \
    include/tets.hpp \
    include/prisms.hpp \
    include/gimmik.h \
    include/zefrPyGlobals.h \
    external/convert.i

DISTFILES += \
    include/zefr.i \
    run/zefrWrap.py \
    Makefile \
    run/runfile.dict \
    run/run_zefr.py \
    run/zefrInterface.py \
    run/run_zefr_samcart.py

