#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "nn.h"

int main() {
    srand(time(0));
    size_t sizes[3] = {2, 2, 1}; // Количество нейронов в каждом слое (слева - входные данные, справа - результат)
   // float start_values[3] = {0.5f, 0.37f, 0.02f}; // Стартовые значения

    float training_input[] = 
    {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };
    float training_output[] = 
    {
        0,
        1,
        1,
        0
    };
    TrainingData data;
    data.input = mat_alloc(4, 2);
    data.output = mat_alloc(4, 1);
    data.input.data = training_input;
    data.output.data = training_output;

    Model model = model_alloc(3, sizes);
    model_random(model);
    
    float cost = model_cost(model, data);
    
    printf("\n----------------\n");
    printf("cost = %f\n", cost);

    return 0;
}