/**
 * Square class
 */

#include "square.h"

std::string Square::getDefaultTag() {
    return "square_";
}

Square::Square(const TensorNode *n_val) {
    createTag();
    val = n_val;
    input.push_back(n_val);
}

Tensor Square::evaluate() const {
    return val->evaluate().square();
}

Tensor Square::evaluate(Session *session) const {
    return session->getEval(val).square();
}

/**
 * TODO: See Add::derivative():
 * d(x^2)/dx = 2x
 */
Tensor Square::derivative(const TensorNode *dx, Session *session) const {
    if (val->getTag().compare(dx->getTag()) == 0) {
        Tensor input_eval = session->getEval(val);
        std::vector<unsigned int> d_shape = input_eval.getShape();
        d_shape.insert(d_shape.end(), 
            d_shape.begin(), 
            d_shape.end());

        // Ensure constant derivatives always have shape [1]
        if (d_shape.size() < 1) {
            d_shape.push_back(1);
        }

        Tensor derivative = Tensor(d_shape);
        derivative.setAllData(0);
        unsigned int count = input_eval.getDataCount();
        for (unsigned int i = 0; i < count; i++) {
            derivative.setData(i + i * count, input_eval.getData(i) * 2);
        }
        return derivative;
    }
    throw std::invalid_argument("TensorNode '" + dx->getTag() + "' is not a valid input for Node '" + getTag() + "'");
}
