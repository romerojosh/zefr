/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

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
