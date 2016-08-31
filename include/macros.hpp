#ifndef macros_hpp
#define macros_hpp

#include <execinfo.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

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
