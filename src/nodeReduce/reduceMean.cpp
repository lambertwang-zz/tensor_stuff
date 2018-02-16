/**
 * ReduceMean class
 */

#include "reduceMean.h"

std::string ReduceMean::getDefaultTag() {
    return "reducemean_";
}

ReduceMean::ReduceMean(const TensorNode *val) {
    createTag();
    input.push_back(val);
}

Tensor ReduceMean::evaluate() const {
    for (const TensorNode *n: input) {
        return n->evaluate().reduceMean();
    }

    return Tensor();
}

Tensor ReduceMean::evaluate(Session *session) const {
    for (const TensorNode *n: input) {
        return session->getEval(n).reduceMean();
    }
    return Tensor();
}

Tensor ReduceMean::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            Tensor input_eval = session->getEval(n);
            std::vector<unsigned int> input_shape = input_eval.getShape(),
                d_shape = std::vector<unsigned int>(input_shape.begin(), input_shape.end() - 1);
            d_shape.insert(d_shape.end(), 
                input_shape.begin(), 
                input_shape.end());

            Tensor derivative = Tensor(d_shape);
            derivative.setAllData(1.0 / input_shape[input_eval.getRank() - 1]);

            return derivative;
        }
    }
    return Tensor(0);
}

