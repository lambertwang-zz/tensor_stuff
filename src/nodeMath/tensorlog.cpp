
/**
 * TensorTensorLog class
 */

#include "tensorlog.h"

std::string TensorLog::getDefaultTag() {
    return "log_";
}

TensorLog::TensorLog(const TensorNode *value) {
    createTag();
    input.push_back(value);
}

Tensor TensorLog::evaluate() const {
    for (const TensorNode *n: input) {
        return n->evaluate().tensor_log();
    }

    return Tensor();
}

Tensor TensorLog::evaluate(Session *session) const {
    for (const TensorNode *n: input) {
        return session->getEval(n).tensor_log();
    }
    return Tensor();
}

/**
 * TODO: See Add::derivative():
 * d(ln(x))/dx = 1/x
 */
Tensor TensorLog::derivative(const TensorNode *dx, Session *session) const {
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
                derivative.setData(i + i * count, 1.0 / input_eval.getData(i));
            }
            return derivative;
        }
    }
    return Tensor(0);
}
