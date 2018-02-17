#include "session/session.h"
#include "../test/lib/testlib.h"

// Library
#include <iostream>
#include <fstream>
#include <sstream>

#include <ctime>

#define ITERATIONS 10
#define TRAINING_COUNT 100
#define CLASSES 10

int main() {
    unsigned int image_size = 4;
    Tensor x_data = readMnistImages(TRAINING_COUNT, &image_size);
    Tensor y_data = readMnistLabels(TRAINING_COUNT);

    // Setup placeholders and variables
    Placeholder *x = new Placeholder("x");
    Placeholder *y_ = new Placeholder("y");
    Tensor w_init = Tensor({image_size, CLASSES});
    w_init.setAllData(0.001);
    Variable *w = new Variable(w_init, "w");
    Variable *b = new Variable({CLASSES}, "b");

    // Create the linear model
    SoftMax *y = new SoftMax(*(new MatMult(x, w)) + *b);

    // Create the loss function
    ReduceMean *loss = new ReduceMean(
        *(new ReduceSum(
            *new DotProduct({ y_, new TensorLog(y) }) * 
            *(new Constant(-1))
            )
        ) +
        *(*(*(new ReduceSum(new ReduceSum(w))) * 
            *(new Constant(0.01))
            ) +
          *(*(new ReduceSum(b)) *
            *(new Constant(0.01))
            ))
        );

    std::map<std::string, Tensor> placeholders;
    placeholders["x"] = x_data;
    placeholders["y"] = y_data;

    Session session = Session();
    // Create the optimizer and training node
    TensorNode *minimizer = new Train(loss, 0.0001);
    // std::cout << "Initial model: " << session.run(y, placeholders) << std::endl;
    std::cout << "Initial Loss : " << session.run(loss, placeholders) << std::endl;
    std::clock_t start = std::clock();
    for (int i = 0; i < ITERATIONS; i++) {
        if (i % (ITERATIONS / 10) == 0) {
            std::cout << "Loss after training iteration #" << i << " : " << session.run(minimizer, placeholders) << std::endl;
            // std::cout << "matMult x: " << session.run(new MatMult(x, w), placeholders) << std::endl;
            // std::cout << "Model: " << session.run(y, placeholders) << std::endl;
        } else {
            // session.run(minimizer, placeholders);
        }
    }
    std::cout << "Time Elapsed: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    // std::cout << "Final model: " << session.run(y, placeholders) << std::endl;
    std::ofstream file;
    file.open("classifier.txt");
    file << "Classifier : " << session.run(w, placeholders) << std::endl;
    file.close();
    std::cout << "Final Loss : " << session.run(loss, placeholders) << std::endl;
    return 0;
}
