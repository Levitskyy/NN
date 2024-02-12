#ifndef NN_H_
#define NN_H_

#include "matrix.h"

typedef struct {
    size_t size;
    Mat values;
} InputLayer;

typedef struct {
    size_t size;
    Mat input_weights;
    Mat input_biases;
    Mat input_values;
    Mat values;
} Layer;

typedef struct {
    size_t size;
    InputLayer inputLayer;
    Layer* layers;
} Model;

InputLayer inputLayer_alloc(size_t size);
Layer layer_alloc(size_t size, size_t prev_layer_size);
Model model_alloc(size_t size, const size_t* sizes);
void model_random(Model model);
void model_compute(Model model);
void print_model(Model model);
void model_init_values(Model model, size_t size, const float *init_values);


#endif // NN_H_