#include "lib/testlib.h"

// Library
#include <iostream>
#include <fstream>
#include <sstream>

#include <ctime>

int main() {
    TensorNode *x = new Variable(5, "x");
    TensorNode *y = new Variable({ 1, 2, 3, 5 }, { 2, 2 }, "y");
    Session *session = new Session();

    std::cout << "\e[35mTesting tensor derivatives\e[0m" << std::endl;
    TensorNode *add = new Add({ y, x });
    session->initialize(add);
    expect(add->derivative(y, session), "1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1");
    expect(add->derivative(x, session), "1, 1, 1, 1");

    std::cout << "\e[35mTesting Softmax derivative\e[0m" << std::endl;
    TensorNode *z = new Variable({ 1, 2, 3, 5, 7, 9, 10, 10, 10 }, { 3, 3 }, "z");
    TensorNode *softmax= new SoftMax(z);
    session->initialize(softmax);
    expect(session->run(softmax), "0.0900306, 0.244728, 0.665241, 0.0158762, 0.11731, 0.866813, 0.333333, 0.333333, 0.333333");
    expect(softmax->derivative(z, session),
        "0.0819251, -0.022033, -0.059892, 0, 0, 0, 0, 0, 0, "
        "-0.022033, 0.184836, -0.162803, 0, 0, 0, 0, 0, 0, "
        "-0.059892, -0.162803, 0.222695, 0, 0, 0, 0, 0, 0, "
        "0, 0, 0, 0.0156242, -0.00186245, -0.0137617, 0, 0, 0, "
        "0, 0, 0, -0.00186245, 0.103549, -0.101686, 0, 0, 0, "
        "0, 0, 0, -0.0137617, -0.101686, 0.115448, 0, 0, 0, "
        "0, 0, 0, 0, 0, 0, 0.222222, -0.111111, -0.111111, "
        "0, 0, 0, 0, 0, 0, -0.111111, 0.222222, -0.111111, "
        "0, 0, 0, 0, 0, 0, -0.111111, -0.111111, 0.222222");


    printTestResults();

    return 0;
}
