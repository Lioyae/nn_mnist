#include <stdio.h>
#include "linalg.h"

int main(void) {

  double p[3] = {1, 2, 3};
  double q[3] = {4, 5, 6};
  double d = dot(p, q, 3);
  printf("dot: %g (expect 32)\n", d);

  double W[6] = {1,2,3, 4,5,6};
  double x[3] = {1,1,1};
  double y[2];
  matvec(W, x, 2, 3, y);
  printf("matvec: y[0]=%g y[1]=%g (expect 6, 15)\n", y[0], y[1]);

  double a[2] = {1, 2};
  double b[3] = {10, 20, 30};
  double C[6];
  outer(a, b, 2, 3, C);
  printf("outer: C = [%g %g %g; %g %g %g] (expect 10 20 30; 20 40 60)\n", C[0], C[1], C[2], C[3], C[4], C[5]);

  double W2[6] = {1,2,3, 4,5,6};
  double x2[2] = {1, 1};
  double y2[3];
  matT_vec(W2, x2, 2, 3, y2);
  printf("matT_vec: y = [%g %g %g] expect 5 7 9", y2[0], y2[1], y2[2]);
  return 0;
}
