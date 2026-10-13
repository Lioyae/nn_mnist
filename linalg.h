#ifndef __LINALG_H__
#define __LINALG_H__

#include <stdlib.h>
#include <string.h>

static inline double *allpc_matrix(int m, int n) {
  return (double *)malloc(sizeof(double) * (size_t)m * n);
}

static inline double *zeros(int n) {
  return (double *)calloc(n, sizeof(double));
}

static inline int idx2(int i, int j, int cols) {
  return i * cols + j;
}

static inline double dot(const double *a, const double *b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; i++) {
    s += a[i] * b[i];
  }
  return s;
}

static inline void matvec(const double *W, const double *x, int m, int n, double *y) {
  for (int i = 0; i < m; i++) {
    const double *row = W + (size_t)i * n;
    y[i] = dot(row, x, n);
  }
}

static inline void matT_vec(const double *W, const double *x, int m, int n, double *y) {
  for (int j = 0; j < n; j++) {
    double s = 0.0;
    for (int i = 0; i < m; i++) {
      s += W[(size_t)i * n + j] * x[i];
    }
    y[j] = s;
  }
}

static inline void vec_add(double *y, const double *b, int n) {
  for (int i = 0; i < n; i ++) {
    y[i] += b[i];
  }
}

static inline void outer(const double *a, const double *b, int m, int n, double *C){
  for (int i = 0; i < m; i++) {
    double ai = a[i];
    double *row = C + (size_t)i * n;
    for (int j = 0; j < n; j++) {
      row[j] = ai * b[j];
    }
  }
}

static inline void hadamard(double *y, const double *a, const double *b, int n) {
  for (int i = 0; i < n; i++) {
    y[i] = a[i] * b[i];
  }
}

#endif
