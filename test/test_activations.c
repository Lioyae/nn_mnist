#include <math.h>
#include <stdio.h>
#include "../activations.h"

int main(void) {
  // 测试 ReLu
  double z[5] = {-2, -1, 0, 1, 2};
  relu_inplace(z, 5);
  printf("relu: %g %g %g %g %g (expect 0 0 0 1 2)\n", z[0], z[1], z[2], z[3], z[4]);

  // 测试 RelU 反向
  double dA1[5] = {1, 1, 1, 1, 1};
  double Z1[5] = {-2, -1, 0, 1, 2};
  double dZ1[5];
  leaky_relu_backward(dA1, Z1, 5, dZ1);
  printf("relu_backward: %g %g %g %g %g (expect 0.01 0.01 0.01 1 1)\n", dZ1[0], dZ1[1], dZ1[2], dZ1[3], dZ1[4]);

  // 测试 Softmax
  double logits[3] = {1.0, 2.0, 3.0};
  double p[3];
  softmax(logits, 3, p);
  printf("softmax: %g %g %g, sum=%g expect 0.090 0.245 0.665 \n", p[0], p[1], p[2], p[0] + p[1] + p[2]);

  // softmax 溢出测试
  double big[3] = {1000.0, 1001.0, 1002.0};
  double p2[3];
  softmax(big, 3, p2);
  printf("softmax_big: %g %g %g, sum=%g \n", p2[0], p2[1], p2[2], p2[0] + p2[1] + p2[2]);

  // sigmoid 测试
  double zs[3] = {-2.0, 0.0, 2.0};
  double as[3];
  sigmoid(zs, 3, as);
  printf("sigmoid: %g %g %g (expect 0.119 0.5 0.881)\n", as[0], as[1], as[2]);

  // 测试 sigmoid_backward
  double dA2[3] = {1.0, 1.0, 1.0};
  double dZ2[3];
  sigmoid_backward(dA2, as, 3, dZ2);
  printf("sigmoid_backward: %g %g %g\n", dZ2[0], dZ2[1], dZ2[2]);

  // 测试 Leaky ReLU
  double zl[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  leaky_relu_inplace(zl, 5);
  printf("leaky_relu: %g %g %g %g %g (expect -0.02 -0.01 0 1 2)\n",
         zl[0], zl[1], zl[2], zl[3], zl[4]);

  // 测试 Leaky ReLU 反向
  double dA3[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
  double Z3[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  double dZ3[5];
  leaky_relu_backward(dA3, Z3, 5, dZ3);
  printf("leaky_relu_backward: %g %g %g %g %g (expect 0.01 0.01 0.01 1 1)\n", dZ3[0], dZ3[1], dZ3[2], dZ3[3], dZ3[4]);

  return 0;
}
