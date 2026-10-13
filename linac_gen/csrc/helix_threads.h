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

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
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

// The team for a region of ``work`` units (particles, grid points): about
// sqrt(work / per_thread) threads, at most the setting, at least one.  A
// thread costs a fixed time to start and stop in every region (here about
// 6 us each, measured on a 14-core Mac), so the best team grows as the
// square root of the work; past it, threads cost more than they save.
// per_thread <= 0: the whole setting.  The team never changes a result.
inline int team_for(double work, double per_thread) {
    const int n = num_threads();
    if (!(per_thread > 0.0)) return n;
    const double t = std::sqrt(std::max(work, 0.0) / per_thread);
    const long r = std::lround(std::min(t, 1.0e6));
    return static_cast<int>(std::max(1L, std::min(static_cast<long>(n), r)));
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

// The calling thread's floating-point environment (rounding mode, flush-to-
// zero) for the threads of a parallel region whose results must equal that
// thread's own arithmetic: OpenMP does not hand it over on every platform
// (libomp copies it on x86 only).  Taken in the calling thread before the
// region; each thread sets it on entry and restores its own on exit, so the
// pool's threads keep theirs for other regions (torch's among them).  In
// the default environment, the only one HELIX uses, this changes nothing.
//     const helix::FpEnv env;
//     #pragma omp parallel ...
//     { const helix::FpEnv::Use use(env); ... }
class FpEnv {
public:
    FpEnv() { std::fegetenv(&env_); }
    class Use {
    public:
        explicit Use(const FpEnv& e) {
            std::fegetenv(&own_);
            std::fesetenv(&e.env_);
        }
        ~Use() { std::fesetenv(&own_); }
        Use(const Use&) = delete;
        Use& operator=(const Use&) = delete;
    private:
        std::fenv_t own_;
    };
private:
    std::fenv_t env_;
};

}  // namespace
}  // namespace helix
