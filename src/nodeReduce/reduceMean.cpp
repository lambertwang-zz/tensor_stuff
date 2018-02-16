/**
 * ReduceMean class
 */

#include "reduceMean.h"

std::string ReduceMean::getDefaultTag() {
    return "reducemean_";
}

ReduceMean::ReduceMean(const TensorNode *n_val) {
    createTag();
    val = n_val;
    input.push_back(n_val);
}

Tensor ReduceMean::evaluate() const {
    return val->evaluate().reduceMean();
}

Tensor ReduceMean::evaluate(Session *session) const {
    return session->getEval(val).reduceMean();
}

Tensor ReduceMean::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    if (val->getTag().compare(dx->getTag()) == 0) {
        Tensor input_eval = session->getEval(val);
        std::vector<unsigned int> input_shape = input_eval.getShape(),
            d_shape = std::vector<unsigned int>(input_shape.begin(), input_shape.end() - 1);
        d_shape.insert(d_shape.end(), 
            input_shape.begin(), 
            input_shape.end());

        Tensor derivative = Tensor(d_shape);
        derivative.setAllData(1.0 / input_shape[input_eval.getRank() - 1]);

        return derivative;
    }
    throw std::invalid_argument("TensorNode '" + dx->getTag() + "' is not a valid input for Node '" + getTag() + "'");
}

