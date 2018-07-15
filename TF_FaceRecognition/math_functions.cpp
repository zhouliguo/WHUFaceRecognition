#include "math_functions.h"
#include <xmmintrin.h>
#include <cstdint>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

float simd_dot(const float* x, const float* y, const long& len) {
  float inner_prod = 0.0f;
  __m128 X, Y; // 128-bit values
  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
  float temp[4];

  long i;
  for (i = 0; i + 4 < len; i += 4) {
      X = _mm_loadu_ps(x + i); // load chunk of 4 floats
      Y = _mm_loadu_ps(y + i);
      acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
  }
  _mm_storeu_ps(&temp[0], acc); // store acc into an array
  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

  // add the remaining values
  for (; i < len; ++i) {
      inner_prod += x[i] * y[i];
  }
  return inner_prod;
}

void matrix_procuct(const float* A, const float* B, float* C, const int n,
    const int m, const int k, bool ta, bool tb) {
#ifdef _BLAS
  arma::fmat mA = ta ? arma::fmat(A, k, n).t() : arma::fmat(A, n, k);
  arma::fmat mB = tb ? arma::fmat(B, m, k).t() : arma::fmat(B, k, m);
  arma::fmat mC(C, n, m, false);
  mC = mA * mB;
#else
  const float* x = B;
  for (int i = 0, idx = 0; i < m; ++i) {
    const float* y = A;
    for (int j = 0; j < n; ++j, ++idx) {
      C[idx] = simd_dot(x, y, k);
      y += k;
    }
    x += k;
  }
#endif
}
