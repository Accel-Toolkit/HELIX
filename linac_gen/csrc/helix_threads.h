// Thread count for a kernel module's OpenMP loops, set from Python.
//
// linac_gen.parallel.threads hands every kernel module the one HELIX thread
// setting (GUI "CPU threads", CLI --threads, linac_gen.set_threads, env
// HELIX_THREADS) through the module's set_num_threads(n).  Each parallel
// region names its team with num_threads(helix::num_threads()), so the team
// does not depend on the thread that calls the kernel.  An OpenMP default
// set by OMP_NUM_THREADS or omp_set_num_threads belongs to the thread that
// set it, and a GUI run is a worker thread of a process that caps
// OMP_NUM_THREADS at 1 for third-party code.
//
// 0, the value before Python sets anything, keeps the calling thread's
// OpenMP default: the behaviour of a binary without this header.
#pragma once

#include <atomic>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helix {
// Internal linkage: every module keeps its own setting, whatever the
// symbol-visibility flags it was built with.
namespace {

inline std::atomic<int>& thread_setting() {
    static std::atomic<int> n{0};
    return n;
}

inline int num_threads() {
#ifdef _OPENMP
    const int n = thread_setting().load(std::memory_order_relaxed);
    return n > 0 ? n : omp_get_max_threads();
#else
    return 1;
#endif
}

inline void set_num_threads(int n) {
    if (n < 0)
        throw std::invalid_argument(
            "thread count must be >= 0 (0 = the OpenMP default)");
    thread_setting().store(n, std::memory_order_relaxed);
}

// The team a parallel region of this module gets from the calling thread,
// for the tests: the setting decides it, not the calling thread's OpenMP
// default (1 in a GUI worker).
inline int team_size() {
    int n = 1;
#ifdef _OPENMP
    #pragma omp parallel num_threads(num_threads())
    {
        if (omp_get_thread_num() == 0)
            n = omp_get_num_threads();
    }
#endif
    return n;
}

}  // namespace
}  // namespace helix
