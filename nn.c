#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

InputLayer inputLayer_alloc(size_t size) {
    InputLayer layer;
    layer.size = size;
    layer.values = mat_alloc(1, size);
    return layer;
}
Layer layer_alloc(size_t size, size_t prev_layer_size) {
    Layer layer;
    layer.size = size;
    layer.input_weights = mat_alloc(prev_layer_size, size);
    layer.input_biases = mat_alloc(1, size);
    layer.input_values = mat_alloc(1, prev_layer_size);
    layer.values = mat_alloc(1, size);
    return layer;
}
Model model_alloc(size_t size, const size_t* sizes) {
    assert(size > 1);
    assert(sizes != NULL);
    
    Model model;
    model.size = size;
    model.inputLayer = inputLayer_alloc(sizes[0]);
    model.layers = (Layer*)malloc(sizeof(Layer) * (size - 1));

    for(size_t i = 1; i < size; ++i) {
        model.layers[i - 1] = layer_alloc(sizes[i], sizes[i - 1]);
    }

    return model;
}

void model_random(Model model) {
    for (size_t i = 0; i < model.size - 1; ++i) {
        mat_rand(model.layers[i].input_weights, -1, 1);
        mat_rand(model.layers[i].input_biases, -1, 1);
    }
}

void model_compute(Model model) {
    mat_copy(model.layers[0].input_values, model.inputLayer.values);
    
    for (size_t i = 0; i < model.size - 1; ++i) {
        if (i != 0) {
            mat_copy(model.layers[i].input_values, model.layers[i - 1].values);
        }
        Layer layer = model.layers[i];
        mat_dot(layer.values, layer.input_values, layer.input_weights);
        mat_sum(layer.values, layer.input_biases);
        mat_sigf(layer.values);
    }
}

void print_model(Model model) {
    for (size_t i = 0; i < model.size - 1; ++i) {
        Layer layer = model.layers[i];
        print_mat(layer.input_values, "values");
        print_mat(layer.input_weights, "weights");
        print_mat(layer.input_biases, "biases");
        print_mat(layer.values, "out");
    }
}

void model_init_values(Model model, size_t size, const float *init_values) {
    assert(size == model.inputLayer.size);
    for (size_t i = 0; i < size; ++i) {
        MAT_AT(model.inputLayer.values, 0, i) = init_values[i];
    }
}
