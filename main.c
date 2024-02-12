#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "nn.h"

int main() {
    srand(time(0));
    size_t sizes[3] = {3, 2, 3}; // Количество нейронов в каждом слое (слева - входные данные, справа - результат)
    float start_values[3] = {0.5f, 0.37f, 0.02f}; // Стартовые значения
    
    Model model = model_alloc(3, sizes);
    model_init_values(model, 3, start_values);
    model_random(model);
    model_compute(model);

    print_mat(model.inputLayer.values, "start values");
    printf("\n----------------\n");

    print_model(model);
    printf("\n----------------\n");

    printf("res = %f\n", MAT_AT(model.layers[model.size - 2].values, 0, 0));

    return 0;
}