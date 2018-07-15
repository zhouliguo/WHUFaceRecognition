#ifndef MATH_FUNCTIONS_H_
#define MATH_FUNCTIONS_H_

#ifdef _BLAS
#ifdef _WIN64
#pragma comment( lib, "blas_win64_MT" )
#pragma comment( lib, "lapack_win64_MT" )
#else
#pragma comment( lib, "libblas" )
#pragma comment( lib, "liblapack" )
#endif
#include <armadillo>
#endif

float simd_dot(const float* x, const float* y, const long& len);

// matrix product:
// MA = ta ? A^T : A;
// MB = tb ? B^T : B;
// return C(n, m) = MA(n, k) * MB(k, m);
void matrix_procuct(const float* A, const float* B, float* C, const int n,
    const int m, const int k, bool ta = false, bool tb = false);

#endif // MATH_FUNCTIONS_H_
