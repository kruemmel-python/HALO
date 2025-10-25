/*
 * halo_driver.cpp
 *
 * Unified HALO native driver that exposes both the legacy CPU fast-path
 * routines and the experimental OpenCL GPU backend from a single
 * translation unit.  The two original implementations lived in
 * `fastpath.cpp` (C++/SIMD) and `CipherCore_OpenCl.c` (plain C).  By
 * compiling this single source file users obtain one shared library that
 * offers the full API surface required by `halo.py` as well as the GPU
 * management functions consumed by `halo_demo_app.py`.
 *
 * The GPU portion is written in C and expects a C ABI.  We therefore wrap
 * its inclusion in an `extern "C"` block before pulling in the modern C++
 * fast-path.  Nothing else is defined in this file – the heavy lifting
 * continues to live in the two implementation units which keeps the
 * maintenance story unchanged while simplifying the build.
 */

extern "C" {
#include "CipherCore_OpenCl.c"
}

#include "fastpath.cpp"
