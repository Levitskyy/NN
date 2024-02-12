#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdlib.h>

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Mat;

#define MAT_AT(mat, i, j) (mat).data[(i) * (mat).cols + (j)]

float rand_float();
float sigf(float value);

Mat mat_alloc(size_t rows, size_t cols);
void print_mat(Mat mat, const char *name);
void mat_rand(Mat mat, float min, float max);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_fill(Mat a, float value);
void mat_mul(Mat dst, float a);
void mat_add (Mat dst, float a);
void mat_copy(Mat dst, Mat origin);
void mat_sigf(Mat mat);


#endif // MATRIX_H_