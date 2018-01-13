/**
 * Clamp class
 */

#include "clamp.h"

Clamp::Clamp(const TensorNode *val, float n_min, float n_max) {
    createTag();

    min = n_min;
    max = n_max;
    input.push_back(val);
}

Tensor Clamp::evaluate() const {
    for (const TensorNode *n: input) {
        return n->evaluate().clamp(min, max);
    }
    return Tensor();
}

Tensor Clamp::evaluate(Session *session) const {
    for (const TensorNode *n: input) {
        return session->getEval(n).clamp(min, max);
    }
    return Tensor();
}
