#include "lib/testlib.h"

// Library
#include <iostream>
#include <fstream>
#include <sstream>

#include <ctime>

int main() {
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
    expect(Tensor({1, 3, 5, 7}, {2, 2}) * Tensor(-1), "-1, -3, -5, -7");
    expect(Tensor({1, 3, 5, 7}, {2, 2}) * Tensor({2, 4, 6, 8}, {2, 2}), "20, 28, 52, 76");
    expect(
        Tensor({
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
            }, {3, 3}) *
        Tensor({
            1,
            2,
            3
        }, {3, 1}), "32, 50, 68");
    expectShape(Tensor({4, 5, 6, 7, 8, 9}, {2, 3}) * Tensor({1, 2, 3}, {3, 1}), "2, 1");
    expect(
        Tensor({
            1, 2, 3, 10,
            10, 3, 2, 1,
            1, 2, 10, 3,
            }, {3, 4}) *
        Tensor({
            0, 1, 0,
            0, 0, 0,
            0, 0, 1,
            1, 0, 0,
            }, {4, 3}), "10, 1, 3, 1, 10, 2, 3, 1, 10");
    expectShape(
        Tensor({
            1, 2, 3, 10,
            10, 3, 2, 1,
            1, 2, 10, 3,
        }, {3, 4}) *
        Tensor({
            0, 1, 0,
            0, 0, 0,
            0, 0, 1,
            1, 0, 0,
        }, {4, 3}), "3, 3");

    std::cout << "\e[35mTesting submatrix addition\e[0m" << std::endl;
    expect(Tensor({
        10, 20, 30, 40, 50, 60,
        10, 20, 30, 40, 50, 60
    }, {2, 3, 2}) + Tensor({1, 2, 3, 4, 5, 6}, {3, 2}),
    "11, 22, 33, 44, 55, 66, 11, 22, 33, 44, 55, 66");

    std::cout << "\e[35mTesting reduction operations\e[0m" << std::endl;
    expect(Tensor({1, 3, 5, 7}, {2, 2}).reduceSum(), "4, 12");
    expect(Tensor({1, 3, 5, 7}, {4}).reduceSum(), "16");
    expectShape(Tensor({1, 3, 5, 7, 9, 11}, {2, 3}).reduceSum(), "2");

    expect(Tensor({1, 3, 5, 7}, {2, 2}).reduceMean(), "2, 6");
    expect(Tensor({1, 3, 5, 7}, {4}).reduceMean(), "4");
    expectShape(Tensor({1, 3, 5, 7, 9, 11}, {2, 3}).reduceMean(), "2");

    expect(Tensor({3, 4, 5, 7}, {2, 2}).vectorNorm(), "5, 8.60233");
    expect(Tensor({1, 3, 5, 7}, {4}).vectorNorm(1), "16");

    std::cout << "\e[35mTesting softmax\e[0m" << std::endl;
    expect(Tensor({1, 1, 2, 2}, {2, 2}).softMax(), "0.5, 0.5, 0.5, 0.5");
    expect(Tensor({1, 2, 3, 4, 5, 7}, {3, 2}).softMax(), "0.268941, 0.731059, 0.268941, 0.731059, 0.119203, 0.880797");
    expect(Tensor({1, 1, 1, 1}, {4}).softMax(), "0.25, 0.25, 0.25, 0.25");

    std::cout << "\e[35mTesting log\e[0m" << std::endl;
    expect(Tensor({-1, 0, .5, 1, 2, 3}, {6}).tensor_log(), "nan, -inf, -0.693147, 0, 0.693147, 1.09861");

    std::cout << "\e[35mTesting dotproduct\e[0m" << std::endl;
    expect(Tensor::dotProduct(Tensor({1, 2, 3, 4}, {2, 2}), Tensor({5, 6, 7, 8}, {2, 2})),
        "5, 12, 21, 32");

    printTestResults();

    return 0;
}
