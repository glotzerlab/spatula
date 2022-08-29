#pragma once

#include "BS_thread_pool.hpp"

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

    ThreadPool(ThreadPool const&) = delete;
    void operator=(ThreadPool const&) = delete;

    private:
    ThreadPool() : m_out(), m_pool() { }
    // Must be before m_pool to ensure construction before the thread pull to
    // avoid crashes.
    BS::synced_stream m_out;
    BS::thread_pool m_pool;
};
