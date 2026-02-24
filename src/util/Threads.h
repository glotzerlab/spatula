// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <functional>
#include <vector>

#include "BS_thread_pool.hpp"

namespace spatula { namespace util {
/**
 * @brief Helper class for handle the parallelization logic for the spatula module.
 *
 * ThreadPool is a singleton which stores the state of parallelization and thread pools for the
 * entire application.
 *
 * The singleton defaults to using every available thread.
 */
class ThreadPool {
    private:
    // Helper to pin a thread to a specific CPU core for better cache locality
    static void pin_thread(std::size_t thread_idx)
    {
        auto process_affinity = BS::get_os_process_affinity();
        if (process_affinity.has_value() && thread_idx < process_affinity->size()) {
            std::vector<bool> thread_affinity(process_affinity->size(), false);
            thread_affinity[thread_idx] = true;
            BS::this_thread::set_os_thread_affinity(thread_affinity);
        }
    }

    public:
    /// Get the ThreadPool singletom
    static ThreadPool& get()
    {
        static ThreadPool threads;
        return threads;
    }

    /// Get the current thread pool
    BS::light_thread_pool& get_pool()
    {
        return m_pool;
    }

    /// Get the synced std::out like stream for output
    BS::synced_stream& get_synced_out()
    {
        return m_out;
    }

    /// Set the number of threads to run spatula on.
    void set_threads(unsigned int num_threads)
    {
        m_pool.reset(num_threads, &ThreadPool::pin_thread);
    }

    /// Get the current number of threads in the thread pool.
    size_t get_num_threads()
    {
        return m_pool.get_thread_count();
    }
    /**
     * @brief enable the serial execution of a given loop.
     *
     * This exists to help with profiling, as this enables profilers like py-spy to determine the
     * slow elements of the loop for optimization purposes.
     *
     * @param start the first index of the loop
     * @param end the last exclusive index of the loop
     * @param loop A function that takes the two indices start and stop and optionally returns
     * something.
     */
    template<typename return_type, typename index_type>
    return_type serial_compute(index_type start,
                               index_type end,
                               std::function<return_type(index_type a, index_type b)> loop)
    {
        return loop(start, end);
    }

    ThreadPool(ThreadPool const&) = delete;
    void operator=(ThreadPool const&) = delete;

    private:
    ThreadPool() : m_out(), m_pool(0, &ThreadPool::pin_thread) { }
    // Must be before m_pool to ensure construction before the thread pool to
    // avoid crashes.
    BS::synced_stream m_out;
    BS::light_thread_pool m_pool;
};
}} // namespace spatula::util
