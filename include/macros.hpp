#ifndef macros_hpp
#define macros_hpp

#include <execinfo.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

/* Nice NVTX macro from Parallel Forall blog */
#if defined(_NVTX) && defined(_GPU)
#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_NVTX_RANGE(name,cid) { \
  /*cudaDeviceSynchronize();*/ \
      int color_id = cid; \
      color_id = color_id%num_colors;\
      nvtxEventAttributes_t eventAttrib = {0}; \
      eventAttrib.version = NVTX_VERSION; \
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
      eventAttrib.colorType = NVTX_COLOR_ARGB; \
      eventAttrib.color = colors[color_id]; \
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
      eventAttrib.message.ascii = name; \
      nvtxRangePushEx(&eventAttrib); \
}
#define POP_NVTX_RANGE {cudaDeviceSynchronize(); nvtxRangePop();}
#else
#define PUSH_NVTX_RANGE(name,cid)
#define POP_NVTX_RANGE
#endif

#ifdef _GPU
#define check_error() \
{ \
  cudaError_t err = cudaGetLastError(); \
  if (err != cudaSuccess) \
  { \
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << std::endl; \
    ThrowException(cudaGetErrorString(err)); \
  } \
}
#endif

#ifndef _GPU
#define check_error() {}
#define event_record_wait_pair(event, stream_r, stream_w) {}
#endif

#define ThrowException(msg) \
{ std::stringstream s; s << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << msg; \
  throw std::runtime_error(s.str());}\

#define error_backtrace \
{ \
  std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << std::endl; \
  void* array[10]; size_t size; \
  size = backtrace(array,10); \
  backtrace_symbols_fd(array, size, STDERR_FILENO); \
}

#define _(x) std::cout << #x << ": " << x << std::endl;

#endif /* macros_hpp */
