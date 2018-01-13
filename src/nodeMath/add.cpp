/**
 * Add class
 */

#include "add.h"

std::string Add::getDefaultTag() {
    return "add_";
}

Add::Add(const std::initializer_list<const TensorNode *> values) {
    createTag();
    for (const TensorNode *n: values) {
        input.push_back(n);
    }
}

Tensor Add::evaluate() const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(n->evaluate());
        } else {
            output += n->evaluate();
        }
    }
    return output;
}

Tensor Add::evaluate(Session *session) const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(session->getEval(n));
        } else {
            output += session->getEval(n);
        }
    }
    return output;
}

Tensor Add::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            return Tensor(1);
        }
    }
    return Tensor(0);
}

Add *operator+(const TensorNode& lhs, const TensorNode& rhs) {
    return new Add({&lhs, &rhs});
}
