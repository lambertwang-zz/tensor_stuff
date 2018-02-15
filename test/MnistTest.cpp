#include "session/session.h"
#include "optimizer/gradientDescentOptimizer.h"
#include "lib/testlib.h"

// Library
#include <iostream>
#include <sstream>

#include <ctime>

#define TRAINING_COUNT 3

int main() {
    unsigned int img_size;

    Tensor x_data = readMnistImages(TRAINING_COUNT, &img_size);
    Tensor y_data = readMnistLabels(TRAINING_COUNT);

    Placeholder *x = new Placeholder("x");
    Placeholder *y_ = new Placeholder("y");
    Variable *w = new Variable({img_size, 10});
    Variable *b = new Variable({10});
    SoftMax *y = new SoftMax(*(new MatMult(x, w)) + *b);

    // ReduceSum *loss = new ReduceSum(new Square(*y_ - *y));
    ReduceMean *loss = new ReduceMean(new ReduceSum(*(*y_ * *(new TensorLog(y))) * *(new Constant(-1))));

    std::map<std::string, Tensor> placeholders;
    placeholders["x"] = x_data;
    placeholders["y"] = y_data;

    Session session = Session();
    // expect(session.run(y, placeholders), "105");
    // std::cout << session.run(y, placeholders) << std::endl;
    // std::cout << session.run(loss, placeholders) << std::endl;
    /*
    sess1.initialize(linear_model);
    expect(sess1.run(linear_model, placeholder3), "0, 0.3, 0.6, 0.9");
    // Add *linear_model = *(*new Placeholder("x") * *new Variable(-1, "w")) + *new Variable(1, "b");
    Placeholder *y = new Placeholder("y");
    // expect(sess1.run(*linear_model - *y, placeholder3), "0, 0.3, 0.6, 0.9");
    ReduceSum *loss = new ReduceSum(new Square(*linear_model - *y));

    sess1.initialize(loss);
    expect(sess1.run(loss, placeholder3), "23.66");

    std::cout << "\e[35mTesting Gradient Descent Optimizer\e[0m" << std::endl;
    */

    return 0;
}
