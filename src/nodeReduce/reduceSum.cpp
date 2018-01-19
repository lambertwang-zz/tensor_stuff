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
            std::vector<unsigned int> input_shape = session->getEval(n).getShape(),
                d_shape = {1};
            d_shape.insert(d_shape.end(), 
                input_shape.begin(), 
                input_shape.end());

            Tensor derivative = Tensor(d_shape);
            derivative.setAllData(1);

            return derivative;
        }
    }
    return Tensor(0);
}

