#pragma once

#include <functional>

#include "BS_thread_pool.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pgop { namespace util {
class ThreadPool {
    public:
    static ThreadPool& get()
    {
        static ThreadPool threads;
        return threads;
    }

    BS::thread_pool& get_pool()
    {
        return m_pool;
    }

    BS::synced_stream& get_synced_out()
    {
        return m_out;
    }

    void set_threads(unsigned int num_threads)
    {
        m_pool.reset(num_threads);
    }

    size_t get_num_threads()
    {
        return m_pool.get_thread_count();
    }
    /**
     * @brief enable the serial execution of a given loop. This exists to help with profiling, as
     * this enables profilers like py-spy to determine the slow elements of the loop for
     * optimization purposes.
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
    // Must be before m_pool to ensure construction before the thread pull to
    // avoid crashes.
    BS::synced_stream m_out;
    BS::thread_pool m_pool;
};

void export_threads(py::module& m);
}} // namespace pgop::util
