#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__

#include <math.h>

static inline void relu_inplace(double *z, int n){
  for (int i = 0; i < n; i++) {
    if (z[i] < 0) z[i] = 0;
  }
}

static inline void relu_backward(const double *dA, const double *Z, int n, double *dZ) {
  for (int i = 0; i < n; i++) {
    dZ[i] = (Z[i] > 0) ? dA[i] : 0.0;
  }
}

static inline void softmax(double *z, int n, double *y) {
  double m = z[0];
  for (int i = 0; i < n; i++){
    if (z[i] > m) m = z[i];
  }

  double s = 0.0;
  for (int i = 0; i < n; i++) {
    y[i] = exp(z[i] -m);
    s += y[i];
  }

  for (int i = 0; i < n; i++) {
    y[i] /= s;
  }
}

static inline void sigmoid(const double *z, int n, double *y) {
  for (int i = 0; i < n; i++) {
    y[i] = 1.0 / (1.0 + exp(-z[i]));
  }
}

static inline void sigmoid_backward(const double *dA, const double *A, int n, double *dZ) {
  for (int i = 0; i < n; i++) {
    dZ[i] = dA[i] * A[i] * (1.0 - A[i]);
  }
}

static inline void leaky_relu_inplace(double *z, int n) {
  for (int i = 0; i < n; i++) {
    if (z[i] < 0) z[i] *= 0.01;
  }
}

static inline void leaky_relu_backward(const double *dA, const double *Z, int n, double *dZ) {
    for (int i = 0; i < n; i++) {
        dZ[i] = (Z[i] > 0) ? dA[i] : dA[i] * 0.01;
    }
}

#endif
