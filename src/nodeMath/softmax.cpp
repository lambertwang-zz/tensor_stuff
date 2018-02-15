/**
 * SoftMax class
 */

#include "softmax.h"

std::string SoftMax::getDefaultTag() {
    return "softmax_";
}

SoftMax::SoftMax(const TensorNode *val) {
    createTag();
    input.push_back(val);
}

Tensor SoftMax::evaluate() const {
    for (const TensorNode *n: input) {
        return n->evaluate().softMax();
    }

    return Tensor();
}

Tensor SoftMax::evaluate(Session *session) const {
    for (const TensorNode *n: input) {
        return session->getEval(n).softMax();
    }
    return Tensor();
}

Tensor SoftMax::derivative(const TensorNode *dx, Session *session) const {
    (void)dx;
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            Tensor input_eval = session->getEval(n);
            std::vector<unsigned int> d_shape = input_eval.getShape();
            d_shape.insert(d_shape.end(), 
                d_shape.begin(), 
                d_shape.end());

            Tensor derivative = Tensor(d_shape);
            derivative.setAllData(0);
            unsigned int count = input_eval.getDataCount();
            for (unsigned int i = 0; i < count; i++) {
                // TODO: Implement this
                derivative.setData(i + i * count, input_eval.getData(i) * 2);
            }
            return derivative;
        }
    }
    return Tensor(0);
}
