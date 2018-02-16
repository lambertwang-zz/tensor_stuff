/**
 * Square class
 */

#include "square.h"

std::string Square::getDefaultTag() {
    return "square_";
}

Square::Square(const TensorNode *val) {
    createTag();
    input.push_back(val);
}

Tensor Square::evaluate() const {
    for (const TensorNode *n: input) {
        return n->evaluate().square();
    }

    return Tensor();
}

Tensor Square::evaluate(Session *session) const {
    for (const TensorNode *n: input) {
        return session->getEval(n).square();
    }
    return Tensor();
}

/**
 * TODO: See Add::derivative():
 * d(x^2)/dx = 2x
 */
Tensor Square::derivative(const TensorNode *dx, Session *session) const {
    (void)dx;
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            Tensor input_eval = session->getEval(n);
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
    }
    return Tensor(0);
}
