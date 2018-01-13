/**
 * Optimizer Class
 * 
 */

#include "optimizer.h"

Optimizer::Optimizer() {
}

/*
Optimizer::Optimizer(const float n_learning_rate) {
    learning_rate = Tensor(n_learning_rate);
}

Optimizer::Optimizer(const Tensor n_learning_rate) {
    learning_rate = Tensor(n_learning_rate);
}
*/

Tensor Optimizer::evaluate() const {
    return Tensor();
}

Tensor Optimizer::evaluate(Session *session) const {
    (void)session;
    return Tensor();
}
