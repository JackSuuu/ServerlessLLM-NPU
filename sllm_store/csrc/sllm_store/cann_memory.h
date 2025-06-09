// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#pragma once

#include <string>
// CANN headers - these will be available when building with CANN toolkit
#ifdef USE_CANN
#include "acl/acl.h"
#include "hccl/hccl.h"
#endif

// #include "cann_memory_pool.h"

class CannMemory {
 public:
  CannMemory();
  ~CannMemory();

  // Disable copying and moving
  CannMemory(const CannMemory&) = delete;
  CannMemory& operator=(const CannMemory&) = delete;
  CannMemory(CannMemory&&) = delete;
  CannMemory& operator=(CannMemory&&) = delete;

  int Allocate(size_t size, int device);
  void* get() const;
  std::string getHandle() const;  // CANN uses string-based handles

 private:
  void* data_;
  std::string handle_;  // CANN IPC handle as string
  size_t size_;
  int device_;
}; 