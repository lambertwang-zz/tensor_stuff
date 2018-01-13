/**
 * GradientDescentOptimizer Class
 * 
 */

#include "gradientDescentOptimizer.h"
#include "trainNode/train.h"

GradientDescentOptimizer::GradientDescentOptimizer(const float n_learning_rate) {
    learning_rate = Tensor(n_learning_rate);
}

GradientDescentOptimizer::GradientDescentOptimizer(const Tensor n_learning_rate) {
    learning_rate = Tensor(n_learning_rate);
}

Tensor GradientDescentOptimizer::evaluate() const {
    return Tensor();
}

Tensor GradientDescentOptimizer::evaluate(Session *session) const {
    (void)session;
    return Tensor();
}

TensorNode *GradientDescentOptimizer::minimize(TensorNode *target) {
    return new Train(this, target);
}
