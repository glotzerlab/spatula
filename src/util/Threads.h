#pragma once

#include <functional>

#include "BS_thread_pool.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pgop { namespace util {
/**
 * @brief Helper class for handle the parallelization logic for the pgop module.
 *
 * ThreadPool is a singleton which stores the state of parallelization and thread pools for the
 * entire application.
 *
 * The singleton defaults to using every available thread.
 */
class ThreadPool {
    public:
    /// Get the ThreadPool singletom
    static ThreadPool& get()
    {
        static ThreadPool threads;
        return threads;
    }

    /// Get the current thread pool
    BS::thread_pool& get_pool()
    {
        return m_pool;
    }

    /// Get the synced std::out like stream for output
    BS::synced_stream& get_synced_out()
    {
        return m_out;
    }

    /// Set the number of threads to run PGOP on.
    void set_threads(unsigned int num_threads)
    {
        m_pool.reset(num_threads);
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
    ThreadPool() : m_out(), m_pool() { }
    // Must be before m_pool to ensure construction before the thread pool to
    // avoid crashes.
    BS::synced_stream m_out;
    BS::thread_pool m_pool;
};

void export_threads(nb::module_& m);
}} // namespace pgop::util
