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
