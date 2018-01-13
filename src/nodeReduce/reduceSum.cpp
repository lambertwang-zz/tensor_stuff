/**
 * ReduceSum class
 */

#include "reduceSum.h"

std::string ReduceSum::getDefaultTag() {
    return "reducesum_";
}

ReduceSum::ReduceSum(const TensorNode *val) {
    createTag();
    input.push_back(val);
}

Tensor ReduceSum::evaluate() const {
    for (const TensorNode *n: input) {
        return n->evaluate().reduceSum();
    }

    return Tensor();
}

Tensor ReduceSum::evaluate(Session *session) const {
    for (const TensorNode *n: input) {
        return session->getEval(n).reduceSum();
    }
    return Tensor();
}

Tensor ReduceSum::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            return Tensor(1);
        }
    }
    return Tensor(0);
}

