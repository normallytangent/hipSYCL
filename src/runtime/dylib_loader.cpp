/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/runtime/dylib_loader.hpp"

#include "hipSYCL/common/debug.hpp"

#include <cassert>
#include <string_view>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

namespace hipsycl {
namespace rt {
namespace detail {

void close_library(void *handle, std::string_view loader) {
#ifndef _WIN32
  if (dlclose(handle)) {
    HIPSYCL_DEBUG_ERROR << loader << ": dlclose() failed" << std::endl;
  }
#else
  if (!FreeLibrary(static_cast<HMODULE>(handle))) {
    HIPSYCL_DEBUG_ERROR << loader << ": FreeLibrary() failed" << std::endl;
  }
#endif
}

void *load_library(const std::string &filename, std::string_view loader) {
#ifndef _WIN32
  if (void *handle = dlopen(filename.c_str(), RTLD_NOW)) {
    assert(handle != nullptr);
    return handle;
  } else {
    HIPSYCL_DEBUG_WARNING << loader << ": Could not load library: "
                          << filename << std::endl;
    if (char *err = dlerror()) {
      HIPSYCL_DEBUG_WARNING << err << std::endl;
    }
  }
#else
  if (HMODULE handle = LoadLibraryA(filename.c_str())) {
    return static_cast<void *>(handle);
  } else {
    // too lazy to use FormatMessage bs right now, so look up the error at
    // https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes
    HIPSYCL_DEBUG_WARNING << loader << ": Could not load library: "
                          << filename << " with: " << GetLastError()
                          << std::endl;
  }
#endif
  return nullptr;
}

void *get_symbol_from_library(void *handle, const std::string &symbolName, std::string_view loader) {
#ifndef _WIN32
  void *symbol = dlsym(handle, symbolName.c_str());
  if (char *err = dlerror()) {
    HIPSYCL_DEBUG_WARNING << loader << ": Could not find symbol name: "
                          << symbolName << std::endl;
    HIPSYCL_DEBUG_WARNING << err << std::endl;
  } else {
    return symbol;
  }
#else
  if (FARPROC symbol =
          GetProcAddress(static_cast<HMODULE>(handle), symbolName.c_str())) {
    return reinterpret_cast<void *>(symbol);
  } else {
    // too lazy to use FormatMessage bs right now, so look up the error at
    // https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes
    HIPSYCL_DEBUG_WARNING << loader << ": Could not find symbol name: "
                          << symbolName << " with: " << GetLastError()
                          << std::endl;
  }
#endif
  return nullptr;
}
} // namespace detail
} // namespace rt
} // namespace hipsycl
