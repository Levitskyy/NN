#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

float sigf(float value) {
    float res = (1 / (1 + expf(-value)));
    return res;
}

Mat mat_alloc(size_t rows, size_t cols) {
    Mat mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = malloc(sizeof(*mat.data) * rows * cols);
    
    return mat;
}

void print_mat(Mat mat, const char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < mat.rows; ++i) {
        for (size_t j = 0; j < mat.cols; ++j) {
            printf("%f    ", (MAT_AT(mat, i, j)));
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_rand(Mat mat, float low, float high) {
    for (size_t i = 0; i < mat.rows; ++i) {
        for (size_t j = 0; j < mat.cols; ++j) {
            MAT_AT(mat, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_sum(Mat dst, Mat a) {
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b) {
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < a.cols; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_fill(Mat a, float value) {
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            MAT_AT(a, i, j) = value;
        }
    }
}

void mat_add(Mat dst, float a) {
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += a;
        }
    }
}

void mat_mul(Mat dst, float a) {
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) *= a;
        }
    }
}

void mat_copy(Mat dst, Mat origin) {
    assert(dst.cols == origin.cols);
    assert(dst.rows == origin.rows);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(origin, i, j);
        }
    }
}

Mat mat_row(Mat mat, size_t row) {
    Mat ret = {
        .rows = 1,
        .cols = mat.cols,
        .data = &MAT_AT(mat, row, 0)
    };

    return ret;
}

void mat_sigf(Mat mat) {
    for (size_t i = 0; i < mat.rows; ++i) {
        for (size_t j = 0; j < mat.cols; ++j) {
            MAT_AT(mat, i, j) = sigf(MAT_AT(mat, i, j));
        }
    }
}
