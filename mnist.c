#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

/**
 * @brief  对数据进行大端转换
 * @param  f 文件地址
 * @return 返回大端数对应的小端数
 */
int read_int_be(FILE *f) {
  unsigned char buf[4];
  if (fread(buf, 1, 4, f) != 4){
    fprintf(stderr, "read_int_be: read failed\n");
    exit(1);
  }
  return ((buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]);
}

/**
 * @brief  加载图像并进行归一化处理
 * @param  path 文件地址
 * @param  out_N 图片数量地址
 * @return 返回data保存有所有图像数据的数组
 */
double *load_images(const char *path, int *out_N){
  // 打开文件
  FILE *f = fopen(path, "rb");
  if (!f){
    perror(path);
    exit(1);
  }

  // 读文件头
  int magic = read_int_be(f);  // 魔数，表示固定为0x00000803, 表示这是图像文件
  int N     = read_int_be(f);  // 图像数量 N
  int rows  = read_int_be(f);  // 行数 rows
  int cols  = read_int_be(f);  // 列数 cols
  printf("images: magic=%x, N = %d, rows=%d, cols=%d\n", magic, N, rows, cols);

  // 分配内存
  if (magic != 0x00000803 || rows != 28 || cols != 28){
    fprintf(stderr, "unexpected image file format\n");
    exit(1);
  }

  int dim = rows * cols;
  double *data = malloc(sizeof(double) * (size_t)N * dim);
  if (!data) {
    fprintf(stderr, "OOM\n");
    exit(1);
  }

  unsigned char *buf = malloc(dim);
  if (!buf){
    fprintf(stderr, "OOM\n");
    exit(1);
  }

  // 逐张读取并归一化
  for (int i = 0; i < N; i++) {
    if (fread(buf, 1, dim, f) != (size_t)dim){
      fprintf(stderr, "short read at image %d\n", i);
      exit(1);
    }

    for (int j = 0; j < dim; j++) {
      data[(size_t)i * dim + j] = buf[j] / 255.0;
    }
  }

  // 清理并返回
  free(buf);
  fclose(f);
  *out_N = N;
  return data;
}

/**
 * @brief  加载图像标签
 * @param  path 文件地址
 * @param  out_N 图片标签数量地址
 * @return 返回labels保存有所有图像标签数据的数组
 */
unsigned char *load_labels(const char *path, int *out_N){
  // 打开文件
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(1);
  }

  // 读取文件头
  int magic = read_int_be(f);
  int N     = read_int_be(f);
  printf("label: magic=%x, N=%d\n", magic, N);
  if (magic != 0x00000801){
    fprintf(stderr, "unexpected label file format\n");
    exit(1);
  }

  unsigned char *labels = malloc(N);
  if (!labels) {
    fprintf(stderr, "OOM\n");
    exit(1);
  }
  if (fread(labels, 1, N, f) != (size_t)N){
    fprintf(stderr, "short read labels\n");
    exit(1);
  }
  fclose(f);
  *out_N = N;
  return labels;
}

/**
 * @brief  使用ASCII打印图像
 * @param  img 图像地址代表第几张图像
 * @param  rows 行号
 * @param  cols 列号
 * @return 无返回值
 */
void print_image(const double *img, int rows, int cols){
  // 设计十级灰度
  const char *ramp = " .:-=+*#%@";

  // 为每个像素分配灰度
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      double v = img[r * cols + c];
      int idx = (int)(v * 9.99);
      putchar(ramp[idx]);
    }
    putchar('\n');
  }
}

/**
 * @brief  输出pam格式图像
 * @param  path 图像地址
 * @param  img  元数据
 * @return 无返回值
 */
void save_image_as_pgm(const char *path, const double *img){
  FILE *f = fopen(path, "wb");
  if (!f){
    perror(path);
    exit(1);
  }
  fprintf(f, "P5\n28 28\n255\n");

  unsigned char buf[28 * 28];

  for (int i = 0; i < 28 * 28; i++) {
    double v = img[i] * 255.0 +0.5;
    if (v < 0.0) v =   0.0;
    if (v > 255.0) v = 255.0;
    buf[i] = (unsigned char)v;
  }
  fwrite(buf, 1, sizeof(buf), f);
  fclose(f);
}

int main(void){
  int n_train = 0, n_lab = 0;
  double *train_x =  load_images("./.data/train-images.idx3-ubyte", &n_train);
  unsigned char *train_y = load_labels("./.data/train-labels.idx1-ubyte", &n_lab);

  if (n_train != n_lab){
    fprintf(stderr, "count mismatch\n");
    return 1;
  }

  for (int i = 0; i < 10; i++) {
    printf("\n--- image %d, label = %d ---\n", i, train_y[i]);
    print_image(train_x + (size_t)i * 784, 28, 28);
  }

  for (int i = 0; i < 3; i++) {
    char name[64];
    snprintf(name, sizeof(name), "digit_%d_label_%d.pam", i, train_y[i]);
    save_image_as_pgm(name, train_x + (size_t)i * 784);
  }

  free(train_x);
  free(train_y);

  return 0;
}
