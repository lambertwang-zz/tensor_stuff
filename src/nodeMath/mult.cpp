/**
 * Mult class
 */

#include "mult.h"

std::string Mult::getDefaultTag() {
    return "mult_";
}

Mult::Mult(std::initializer_list<const TensorNode *> values) {
    createTag();
    for (const TensorNode *n: values) {
        input.push_back(n);
    }
}

Tensor Mult::evaluate() const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(n->evaluate());
        } else {
            output *= n->evaluate();
        }
    }
    return output;
}

Tensor Mult::evaluate(Session *session) const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(session->getEval(n));
        } else {
            output *= session->getEval(n);
        }
    }
    return output;
}

Tensor Mult::derivative(const TensorNode *dx, Session *session) const {
    Tensor output;
    bool is_initialized = false;
    bool is_input = false;
    for (const TensorNode *n: input) {
        if (dx->getTag().compare(n->getTag())) {
            if (!is_initialized) {
                is_initialized = true;
                output = Tensor(session->getEval(n));
            } else {
                output *= session->getEval(n);
            }
        } else {
            is_input = true;
        }
    }

    if (is_input) {
        return output;
    } else {
        return Tensor(0);
    }
}

Mult *operator*(const TensorNode &lhs, const TensorNode& rhs) {
    return new Mult({&lhs, &rhs});
}
