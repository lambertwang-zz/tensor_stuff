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

Tensor Square::derivative(const TensorNode *dx, Session *session) const {
    (void)dx;
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            return session->getEval(n);
        }
    }
    return Tensor(0);
}
