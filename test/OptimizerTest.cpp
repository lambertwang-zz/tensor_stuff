#include "session/session.h"
#include "optimizer/gradientDescentOptimizer.h"
#include "lib/testlib.h"

// Library
#include <iostream>
#include <fstream>
#include <sstream>

#include <ctime>

int main() {
    std::cout << "\e[35mTesting sessions\e[0m" << std::endl;

    Session sess1 = Session();
    expect(sess1.run(new Add({new Variable(5), new Constant(100)})), "105");

    std::map<std::string, Tensor> placeholder;
    placeholder["test"] = Tensor(35);
    expect(sess1.run(new Add({new Placeholder("test"), new Constant(100)}), placeholder), "135");

    expect(sess1.run(new Add({new Constant({10, 20, 30, 40}, {4}), new Constant({4, 3, 2, 1}, {4})})), "14, 23, 32, 41");

    std::map<std::string, Tensor> placeholder2;
    placeholder2["test"] = Tensor({1000, 2000, 3000, 4000}, {4});
    expect(sess1.run(new Add({new Placeholder("test"), new Constant({100, 2, 4, 5}, {4})}), placeholder2), "1100, 2002, 3004, 4005");

    std::cout << "\e[35mTesting error function\e[0m" << std::endl;

    std::map<std::string, Tensor> placeholder3;
    placeholder3["x"] = Tensor({1, 2, 3, 4}, {4});
    placeholder3["y"] = Tensor({0, -1, -2, -3}, {4});

    Add *linear_model = *(*new Placeholder("x") * *new Variable(0.3, "w")) + *new Variable(-0.3, "b");
    sess1.initialize(linear_model);
    expect(sess1.run(linear_model, placeholder3), "0, 0.3, 0.6, 0.9");
    // Add *linear_model = *(*new Placeholder("x") * *new Variable(-1, "w")) + *new Variable(1, "b");
    Placeholder *y = new Placeholder("y");
    // expect(sess1.run(*linear_model - *y, placeholder3), "0, 0.3, 0.6, 0.9");
    ReduceSum *loss = new ReduceSum(new Square(*linear_model - *y));

    sess1.initialize(loss);
    expect(sess1.run(loss, placeholder3), "23.66");

    std::cout << "\e[35mTesting Gradient Descent Optimizer\e[0m" << std::endl;

    double learning_rate = 0.01;
    unsigned int iterations = 1000;
    // std::cout << "Training Gradient descent optimizer on y = w * x + b." << std::endl;
    // std::cout << "y = -1 * x + 1." << std::endl;
    // std::cout << "Initial values: w = 0.3, b = -0.3" << std::endl;
    // std::cout << "Learning rate: " << learning_rate << ", Iterations: " << iterations << std::endl;
    GradientDescentOptimizer optimizer = GradientDescentOptimizer(Tensor(learning_rate));
    TensorNode *train = optimizer.minimize(loss);
    sess1.initialize(train);
    std::clock_t start;
    start = std::clock();
    /*
    std::ofstream file;
    file.open("loss_function.csv");
    file << "loss,w,b," << std::endl;
    */
    for (unsigned int i = 0; i < iterations; i++) {
        Tensor loss = sess1.run(train, placeholder3);
        // std::cout << "w = " << sess1.getVar(new Variable(0, "w")).getAllData() << ", ";
        // std::cout << "b = " << sess1.getVar(new Variable(0, "b")).getAllData() << std::endl;
        /*
        file << loss.getAllData() << ",";
        file << sess1.getVar(new Variable(0, "w")).getAllData() << ",";
        file << sess1.getVar(new Variable(0, "b")).getAllData() << ",";
        file <<std::endl;
        */
    }
    // file.close();
    std::cout << "Time Elapsed: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    // std::cout << "Final loss = " << sess1.run(train, placeholder3).getAllData() << std::endl;
    // std::cout << "Final w = " << sess1.getVar(new Variable(0, "w")).getAllData() << std::endl;
    // std::cout << "Final b = " << sess1.getVar(new Variable(0, "b")).getAllData() << std::endl;
    expectLt(sess1.run(train, placeholder3), 1e-4);

    GradientDescentOptimizer optimizer2 = GradientDescentOptimizer(Tensor(0.01));
    Session sess2 = Session();
    TensorNode *train2 = optimizer2.minimize(new Add({new Mult({new Constant(2), new Variable(5, "x")}), new Variable(-5, "b")}));
    start = std::clock();
    for (unsigned int i = 0; i < 100; i++) {
        sess2.run(train2);
    }
    std::cout << "Time Elapsed: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    expectLt(sess2.run(train2), 1e-4);

    /*
    double learning_rate_3 = 0.0015;
    unsigned int iterations_3 = 10000;
    std::cout << "Training Gradient descent optimizer on y = a * x^2 + b * x + c." << std::endl;
    std::cout << "Learning rate: " << learning_rate_3 << ", Iterations: " << iterations_3 << std::endl;

    Placeholder *x_3 = new Placeholder("x");
    Add *model_3= *(*(*new Variable(3, "a") * *new Square(x_3)) + *(*new Variable(5, "b") * *x_3)) + *new Variable(-90, "c");

    // Add *linear_model = *(*new Placeholder("x") * *new Variable(-1, "w")) + *new Variable(1, "b");
    Placeholder *y_3 = new Placeholder("y");
    // expect(sess1.run(*linear_model - *y, placeholder3), "0, 0.3, 0.6, 0.9");
    Session sess3 = Session();
    ReduceSum *loss_3 = new ReduceSum(new Square(*model_3 - *y_3));

    GradientDescentOptimizer optimizer_3 = GradientDescentOptimizer(Tensor(learning_rate_3));
    TensorNode *train_3 = optimizer_3.minimize(loss_3);

    std::map<std::string, Tensor> placeholder_3;
    placeholder_3["x"] = Tensor({1, 2, 3, 4, 5}, {5});
    placeholder_3["y"] = Tensor({190, 676, 1436, 2470, 3778}, {5});

    sess3.initialize(train_3);
    start = std::clock();
    std::ofstream file_3;
    file_3.open("loss_function.csv");
    file_3 << "loss,w,b," << std::endl;
    for (unsigned int i = 0; i < iterations_3; i++) {
        Tensor loss = sess3.run(train_3, placeholder_3);
        if (i % 50 == 0) {
            file_3 << loss.getAllData() << ",";
            file_3 << sess3.getVar(new Variable(0, "a")).getAllData() << ",";
            file_3 << sess3.getVar(new Variable(0, "b")).getAllData() << ",";
            file_3 << sess3.getVar(new Variable(0, "c")).getAllData() << ",";
            file_3 <<std::endl;
        }
        if (i % 1000 == 0) {
            std::cout << "Iteration " << i << std::endl;
            std::cout << "loss = " << loss.getAllData() << std::endl;
            std::cout << "a    = " << sess3.getVar(new Variable(0, "a")).getAllData() << std::endl;
            std::cout << "b    = " << sess3.getVar(new Variable(0, "b")).getAllData() << std::endl;
            std::cout << "c    = " << sess3.getVar(new Variable(0, "c")).getAllData() << std::endl;
        }
        */
    /*
    }
    file_3.close();
    std::cout << "Time Elapsed: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "Final loss = " << sess3.run(train_3, placeholder_3).getAllData() << std::endl;
    std::cout << "Final a = " << sess3.getVar(new Variable(0, "a")).getAllData() << std::endl;
    std::cout << "Final b = " << sess3.getVar(new Variable(0, "b")).getAllData() << std::endl;
    std::cout << "Final c = " << sess3.getVar(new Variable(0, "c")).getAllData() << std::endl;
    */

    std::cout << "\e[35mTesting Calculator Abilities\e[0m" << std::endl;

    GradientDescentOptimizer optimizer_4 = GradientDescentOptimizer(Tensor(.5));
    Session sess_4 = Session();
    TensorNode *train_4 = optimizer_4.minimize(
        new Square(
            new Subtract({
                new Mult({
                        new Constant(3), 
                        new Constant(3)
                        }), 
                new Variable(0, "x")
                })
            )
        );
    sess_4.run(train_4);
    expect(sess_4.getVarTag("x"), "9");

    printTestResults();

    return 0;
}
