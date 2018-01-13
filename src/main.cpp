#include "node/node.h"
#include "session/session.h"
#include "optimizer/gradientDescentOptimizer.h"

// Library
#include <iostream>
#include <sstream>

std::ostream& operator<<(std::ostream& out, const std::vector<float>& data) {
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
        std::cout << "Expecting \"\e[1m\e[33m" << t.getAllData() << "\e[0m\"" << std::endl;
        std::cout << "To Equal  \"\e[1m\e[32m" << e << "\e[0m\"" << std::endl;
    } else {
        std::cout << "Received expected output: \"\e[1m\e[32m" << e << "\e[0m\"" << std::endl;
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

    std::vector<float> node4_data = {1.5, 2.5, 3.5, 4.5};
    std::vector<int> node4_shape = {4};
    Constant node4 = Constant(node4_data, node4_shape);
    expect(node4.evaluate(), "1.5, 2.5, 3.5, 4.5");

    std::vector<float> node5_data = {1, 2, 3, 4, 5, 6};
    std::vector<int> node5_shape = {2, 1, 3};
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
    // Add *linear_model = *(*new Placeholder("x") * *new Variable(-1, "w")) + *new Variable(1, "b");
    Placeholder *y = new Placeholder("y");
    ReduceSum *loss = new ReduceSum(new Square(*linear_model - *y));

    expect(sess1.run(loss, placeholder3), "23.66");

    std::cout << "\e[35mTesting Gradient Descent Optimizer\e[0m" << std::endl;

    GradientDescentOptimizer optimizer = GradientDescentOptimizer(Tensor(0.01));
    TensorNode *train = optimizer.minimize(loss);
    expect(sess1.run(train, placeholder3), "23.66");
    expect(sess1.run(train, placeholder3), "23.66");

    printTestResults();

    return 0;
}
