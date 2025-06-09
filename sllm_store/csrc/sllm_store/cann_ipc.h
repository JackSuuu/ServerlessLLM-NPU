// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
// ----------------------------------------------------------------------------
#pragma once

#ifdef USE_CANN
#include <string>
#include <unordered_map>
#include <mutex>

// Simple IPC using memory mapping - simplified for initial testing
class CannIpcManager {
public:
    static CannIpcManager& getInstance() {
        static CannIpcManager instance;
        return instance;
    }

    // For testing, use a simple mapping of device pointers to sizes
    void registerMemory(void* ptr, size_t size, int device_id);
    std::string createIpcHandle(void* device_ptr, size_t size, int device_id);
    void* openIpcHandle(const std::string& handle_str, int device_id);
    void closeIpcHandle(void* ptr);

private:
    struct MemoryInfo {
        size_t size;
        int device_id;
    };
    
    std::mutex mutex_;
    std::unordered_map<void*, MemoryInfo> memory_registry_;
    std::unordered_map<std::string, void*> handle_to_ptr_;
};

#endif // USE_CANN