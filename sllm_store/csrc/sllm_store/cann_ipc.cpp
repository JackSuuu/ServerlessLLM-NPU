#ifdef USE_CANN
#include "cann_ipc.h"
#include <sstream>
#include <iomanip>
#include <glog/logging.h>

void CannIpcManager::registerMemory(void* ptr, size_t size, int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_registry_[ptr] = {size, device_id};
}

std::string CannIpcManager::createIpcHandle(void* device_ptr, size_t size, int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Register the memory
    memory_registry_[device_ptr] = {size, device_id};
    
    // Create a simple handle (for testing - in production use proper IPC)
    std::ostringstream oss;
    oss << std::hex << reinterpret_cast<uintptr_t>(device_ptr) 
        << "_" << size << "_" << device_id;
    std::string handle = oss.str();
    
    handle_to_ptr_[handle] = device_ptr;
    return handle;
}

void* CannIpcManager::openIpcHandle(const std::string& handle_str, int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = handle_to_ptr_.find(handle_str);
    if (it != handle_to_ptr_.end()) {
        return it->second;
    }
    
    LOG(ERROR) << "IPC handle not found: " << handle_str;
    return nullptr;
}

void CannIpcManager::closeIpcHandle(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_registry_.erase(ptr);
}

#endif // USE_CANN