#include "session/session.h"
#include "lib/testlib.h"

// Library
#include <iostream>
#include <fstream>
#include <sstream>

#include <ctime>

#define LEARNING_RATE_1 0.01
#define ITERATIONS_1 1000

void sessionTest() {
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
}

TensorNode *erfTest() {
    std::cout << "\e[35mTesting error function\e[0m" << std::endl;

    Session session = Session();
    std::map<std::string, Tensor> placeholder;
    placeholder["x"] = Tensor({1, 2, 3, 4}, {4});
    placeholder["y"] = Tensor({0, -1, -2, -3}, {4});

    Add *linear_model = *(*new Placeholder("x") * *new Variable(0.3, "w")) + *new Variable(-0.3, "b");
    session.initialize(linear_model);
    expect(session.run(linear_model, placeholder), "0, 0.3, 0.6, 0.9");

    Placeholder *y = new Placeholder("y");
    ReduceSum *loss = new ReduceSum(new Square(*linear_model - *y));

    session.initialize(loss);
    expect(session.run(loss, placeholder), "23.66");

    return loss;
}

void gradientDescent(
    TensorNode *loss,
    double learning_rate,
    unsigned int iterations,
    std::map<std::string, Tensor> placeholder) {

    std::cout << "\e[35mTesting Gradient Descent Optimizer\e[0m" << std::endl;

    Session session = Session();
    TensorNode *train = new Train(loss, learning_rate);
    session.initialize(train);
    std::clock_t start = std::clock();
    /*
    std::ofstream file;
    file.open("loss_function.csv");
    file << "loss,w,b," << std::endl;
    */
    for (unsigned int i = 0; i < iterations; i++) {
        Tensor loss = session.run(train, placeholder);
        /*
        file << loss.getAllData() << ",";
        file << sess1.getVar(new Variable(0, "w")).getAllData() << ",";
        file << sess1.getVar(new Variable(0, "b")).getAllData() << ",";
        file <<std::endl;
        */
    }
    // file.close();
    std::cout << "Time Elapsed: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    expectLt(session.run(loss, placeholder), 1e-4);
}

int main() {
    sessionTest();
    TensorNode *loss = erfTest();

    std::map<std::string, Tensor> placeholder;
    placeholder["x"] = Tensor({1, 2, 3, 4}, {4});
    placeholder["y"] = Tensor({0, -1, -2, -3}, {4});

    gradientDescent(loss, LEARNING_RATE_1, ITERATIONS_1, placeholder);
    TensorNode *loss2 = new Add({new Mult({new Constant(2), new Variable(5, "x")}), new Variable(-5, "b")});
    gradientDescent(loss2, LEARNING_RATE_1, 100, placeholder);

    std::cout << "\e[35mTesting Calculator Abilities\e[0m" << std::endl;

    Session sess_4 = Session();
    TensorNode *train_4 = new Train(
        new Square(
            new Subtract({
                new Mult({
                        new Constant(3), 
                        new Constant(3)
                        }), 
                new Variable(0, "x")
                })
            )
        , 0.5);
    sess_4.run(train_4);
    expect(sess_4.getVarTag("x"), "9");

    printTestResults();

    return 0;
}
