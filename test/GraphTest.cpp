#include "session/session.h"
#include "optimizer/gradientDescentOptimizer.h"
#include "lib/testlib.h"

// Library
#include <iostream>
#include <sstream>

#include <ctime>

int main() {
    unsigned int item_count = 5;
    unsigned int image_size = 4;
    unsigned int classes = 3;
    Tensor x_data = Tensor({
        3, 2, 0, 0,
        0, 4, 9, 0,
        0, 10, 0, 3,
        0, 3, 6, 0,
        0, 5, 0, 2,
    }, {item_count, image_size});
    Tensor y_data = Tensor({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 1, 0,
        0, 0, 1,
    }, {item_count, classes});

    Placeholder *x = new Placeholder("x");
    Placeholder *y_ = new Placeholder("y");
    Variable *w = new Variable({image_size, classes});
    Variable *b = new Variable({classes});
    SoftMax *y = new SoftMax(*(new MatMult(x, w)) + *b);

    // ReduceSum *loss = new ReduceSum(new Square(*y_ - *y));
    // ReduceMean *loss = new ReduceMean(new ReduceSum(*(*y_ * *(new TensorLog(y))) * *(new Constant(-1))));
    Mult *loss = *(*y_ * *(new TensorLog(y))) * *(new Constant(-1));

    std::map<std::string, Tensor> placeholders;
    placeholders["x"] = x_data;
    placeholders["y"] = y_data;

    Session session = Session();
    std::cout << "Linear Model: " << session.run(y, placeholders) << std::endl;
    std::cout << "Loss function: " << session.run(loss, placeholders) << std::endl;
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
