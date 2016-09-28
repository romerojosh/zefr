#ifndef macros_hpp
#define macros_hpp

#include <sstream>
#include <stdexcept>

#define ThrowException(msg) \
{ std::stringstream s; s << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << msg; \
  throw std::runtime_error(s.str());}\

#define _(x) std::cout << #x << ": " << x << std::endl;

#endif /* macros_hpp */
