#include "session/session.h"
#include "../test/lib/testlib.h"

// Library
#include <iostream>
#include <sstream>

#include <ctime>

#define ITERATIONS 1000

int main() {
    unsigned int item_count = 3;
    // unsigned int item_count = 5;
    unsigned int image_size = 4;
    unsigned int classes = 3;
    Tensor x_data = Tensor({
        1, 2, 3, 10,
        10, 3, 2, 1,
        1, 2, 10, 3,
        // 15, 5, 2, 1,
        // 1, 2, 15, 5,
    }, {item_count, image_size});
    Tensor y_data = Tensor({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        // 0, 1, 0,
        // 0, 0, 1,
    }, {item_count, classes});

    // Setup placeholders and variables
    Placeholder *x = new Placeholder("x");
    Placeholder *y_ = new Placeholder("y");
    Variable *w = new Variable(
        {
            .10, .20, .30,
            .11, .21, .31,
            .12, .22, .32,
            .13, .23, .33,
        },
        {image_size, classes}, "w");
    Variable *b = new Variable({classes}, "b");

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
    TensorNode *minimizer = new Train(loss, 0.01);
    std::cout << "Initial model: " << session.run(y, placeholders) << std::endl;
    std::cout << "Initial Loss : " << session.run(loss, placeholders) << std::endl;
    std::clock_t start = std::clock();
    for (int i = 0; i < ITERATIONS; i++) {
        if (i % (ITERATIONS / 10) == 0) {
            std::cout << "Loss after training iteration #" << i << " : " << session.run(minimizer, placeholders) << std::endl;
        } else {
            session.run(minimizer, placeholders);
        }
    }
    std::cout << "Time Elapsed: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "Final model: " << session.run(y, placeholders) << std::endl;
    std::cout << "Classifier : " << session.run(w, placeholders) << std::endl;
    std::cout << "Final Loss : " << session.run(loss, placeholders) << std::endl;
    return 0;
}
