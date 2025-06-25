// Headers for libqwutil (used by both the frontend and MLIR)

// You need to #include this before anything else on Windows.
// See https://stackoverflow.com/a/6563891/321301

#ifndef UTIL_H
#define UTIL_H

// We use M_PI, but per [1]:
// > The math constants aren't defined in Standard C/C++. To use them, you must
// > first define _USE_MATH_DEFINES, and then include <cmath> or <math.h>.
// [1]: https://learn.microsoft.com/en-us/cpp/c-runtime-library/math-constants?view=msvc-170
#define _USE_MATH_DEFINES
#include <cmath>

// Tolerance for real parameters (e.g., global phases or the theta in
// Rz(theta))
#define ATOL 1e-12

#endif
