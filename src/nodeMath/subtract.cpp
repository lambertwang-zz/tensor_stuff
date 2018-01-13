/**
 * Subtract class
 */

#include "subtract.h"

std::string Subtract::getDefaultTag() {
    return "subtract_";
}

Subtract::Subtract(std::initializer_list<const TensorNode *> values) {
    createTag();
    for (const TensorNode *n: values) {
        input.push_back(n);
    }
}

Tensor Subtract::evaluate() const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(n->evaluate());
        } else {
            output -= n->evaluate();
        }
    }
    return output;
}

Tensor Subtract::evaluate(Session *session) const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(session->getEval(n));
        } else {
            output -= session->getEval(n);
        }
    }
    return output;
}

Tensor Subtract::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    if (input.size() == 0) {
        return Tensor(0);
    }
    // If input is the target, return 1
    if (input[0]->getTag().compare(dx->getTag()) == 0) {
        return Tensor(1);
    }
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            return Tensor(-1);
        }
    }
    return Tensor(0);
}

Subtract *operator-(const TensorNode& lhs, const TensorNode& rhs) {
    return new Subtract({&lhs, &rhs});
}
