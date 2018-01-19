#include "node/node.h"
#include "session/session.h"
#include "optimizer/gradientDescentOptimizer.h"

// Library
#include <iostream>
#include <fstream>
#include <sstream>

#include <ctime>

std::ostream& operator<<(std::ostream& out, const std::vector<double>& data) {
    if (data.size() > 0) {
        out << data[0];
    } for (unsigned int i = 1; i < data.size(); i++) {
           out << ", " << data[i];
    }

    return out;
}

static unsigned int error_count = 0;

void expect(Tensor t, std::string e) {
    std::ostringstream sstream;
    sstream << t.getAllData();
    std::string tmp = sstream.str();
    if (tmp.compare(e)) {
        error_count++;
        std::cout << "\e[1m\e[31mERROR, MISMATCH!\e[0m" << std::endl;
        std::cout << "Expecting \"\e[1m\e[34m" << t.getAllData() << "\e[0m\"" << std::endl;
        std::cout << "To Equal  \"\e[1m\e[36m" << e << "\e[0m\"" << std::endl;
    } else {
        std::cout << "Received expected output: \"\e[1m\e[36m" << e << "\e[0m\"" << std::endl;
    }
}

void printTestResults() {
    if (error_count) {
        std::cout << "\e[31mTests resulted in " << error_count << " Error(s)!\e[0m" << std::endl;

    } else {
        std::cout << "\e[32m" << "All tests sucessful!" << "\e[0m" << std::endl;
    }
}

int main(int argc, char **argv) {
    std::cout << "\e[35mTesting tensor operations\e[0m" << std::endl;
    expect(Tensor(1) + Tensor(1), "2");
    expect(Tensor(2) + Tensor(3), "5");
    expect(Tensor(.02) + Tensor(40), "40.02");
    expect(Tensor(2) + Tensor(40), "42");

    expect(Tensor(.02) * Tensor(40), "0.8");
    expect(Tensor(2) - Tensor(40), "-38");

    std::cout << "\e[35mTesting constant tensors\e[0m" << std::endl;


    Constant node1 = Constant(3);
    expect(node1.evaluate(), "3");

    Constant node2 = Constant(5.5);
    expect(node2.evaluate(), "5.5");

    Constant node3 = Constant(5);
    expect(node3.evaluate(), "5");

    std::vector<double> node4_data = {1.5, 2.5, 3.5, 4.5};
    std::vector<unsigned int> node4_shape = {4};
    Constant node4 = Constant(node4_data, node4_shape);
    expect(node4.evaluate(), "1.5, 2.5, 3.5, 4.5");

    std::vector<double> node5_data = {1, 2, 3, 4, 5, 6};
    std::vector<unsigned int> node5_shape = {2, 1, 3};
    Constant node5 = Constant(node5_data, node5_shape);
    expect(node5.evaluate(), "1, 2, 3, 4, 5, 6");

    Constant node6 = Constant({11, 22, 33, 44}, {4});
    expect(node6.evaluate(), "11, 22, 33, 44");

    std::cout << "\e[35mTesting mathematics tensors\e[0m" << std::endl;

    Add math1 = Add({&node1, &node2});
    expect(math1.evaluate(), "8.5");
    // Using operator overloads
    expect((node1 + node2)->evaluate(), "8.5");
    expect((node1 - node2)->evaluate(), "-2.5");

    Add math2 = Add({&node4, &node6});
    expect(math2.evaluate(), "12.5, 24.5, 36.5, 48.5");

    Mult math3 = Mult({&node1, &node2});
    expect(math3.evaluate(), "16.5");
    expect((node1 * node2)->evaluate(), "16.5");

    Mult math4 = Mult({&node4, &node6});
    expect(math4.evaluate(), "16.5, 55, 115.5, 198");

    Square math5 = Square(&node2);
    expect(math5.evaluate(), "30.25");

    Square math6 = Square(&node4);
    expect(math6.evaluate(), "2.25, 6.25, 12.25, 20.25");

    expect(ReduceSum(&node4).evaluate(), "12");

    std::cout << "\e[35mTesting matrix multiplications\e[0m" << std::endl;
    expect(Tensor({1, 3, 5, 7}, {2, 2}) * Tensor({2, 4, 6, 8}, {2, 2}), "22, 34, 46, 74");
    expect(Tensor({4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 3}) * Tensor({1, 2, 3}, {1, 3}), "48, 54, 60");

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

    Add *linear_model = *(*new Variable(0.3, "w") * *new Placeholder("x")) + *new Variable(-0.3, "b");
    sess1.initialize(linear_model);
    expect(sess1.run(linear_model, placeholder3), "0, 0.3, 0.6, 0.9");
    // Add *linear_model = *(*new Placeholder("x") * *new Variable(-1, "w")) + *new Variable(1, "b");
    Placeholder *y = new Placeholder("y");
    // expect(sess1.run(*linear_model - *y, placeholder3), "0, 0.3, 0.6, 0.9");
    ReduceSum *loss = new ReduceSum(new Square(*linear_model - *y));

    sess1.initialize(loss);
    expect(sess1.run(loss, placeholder3), "23.66");

    std::cout << "\e[35mTesting Gradient Descent Optimizer\e[0m" << std::endl;
    /*

    double learning_rate = 0.01;
    unsigned int iterations = 1000;
    std::cout << "Training Gradient descent optimizer on y = w * x + b." << std::endl;
    std::cout << "y = -1 * x + 1." << std::endl;
    std::cout << "Initial values: w = 0.3, b = -0.3" << std::endl;
    std::cout << "Learning rate: " << learning_rate << ", Iterations: " << iterations << std::endl;
    GradientDescentOptimizer optimizer = GradientDescentOptimizer(Tensor(learning_rate));
    TensorNode *train = optimizer.minimize(loss);
    sess1.initialize(train);
    std::clock_t start;
    start = std::clock();
    std::ofstream file;
    file.open("loss_function.csv");
    file << "loss,w,b," << std::endl;
    for (unsigned int i = 0; i < iterations; i++) {
        Tensor loss = sess1.run(train, placeholder3);
        file << loss.getAllData() << ",";
        file << sess1.getVar(new Variable(0, "w")).getAllData() << ",";
        file << sess1.getVar(new Variable(0, "b")).getAllData() << ",";
        file <<std::endl;
    }
    file.close();
    std::cout << "Time Elapsed: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "Final loss = " << sess1.run(train, placeholder3).getAllData() << std::endl;
    std::cout << "Final w = " << sess1.getVar(new Variable(0, "w")).getAllData() << std::endl;
    std::cout << "Final b = " << sess1.getVar(new Variable(0, "b")).getAllData() << std::endl;

    GradientDescentOptimizer optimizer2 = GradientDescentOptimizer(Tensor(0.01));
    Session sess2 = Session();
    TensorNode *train2 = optimizer2.minimize(new Add({new Mult({new Constant(2), new Variable(5, "x")}), new Variable(-5, "b")}));
    start = std::clock();
    for (unsigned int i = 0; i < 100; i++) {
        sess2.run(train2);
    }
    std::cout << "Time Elapsed: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    expect(sess2.run(train2), "7.13624e-005");

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
        /*
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

    GradientDescentOptimizer optimizer_4 = GradientDescentOptimizer(Tensor(1));
    Session sess_4 = Session();
    TensorNode *train_4 = optimizer_4.minimize(
            // new ReduceSum(
                new Square(
                    new Subtract({
                        new Mult({
                                new Constant(3), 
                                new Constant(3)
                                }), 
                        new Variable(0, "x")
                        })
                    )
              //   )
            );
    for (unsigned int i = 0; i < 1; i++) {
        sess_4.run(train_4);
    }
    expect(sess_4.getVarTag("x"), "9");

    printTestResults();

    return 0;
}
