/**
 * GradientDescentOptimizer Class Header
 * 
 * We can use a technique that evaluates and updates the coefficients every
 * iteration called stochastic gradient descent to minimize the error of a
 * model on our training data.
 * 
 * 
 * This procedure can be used to find the set of coefficients in a model
 * that result in the smallest error for the model on the training data.
 * Each iteration, the coefficients (b) in machine learning language are
 * updated using the equation:
 *  b = b - learning_rate * error * x
 */

#ifndef __GRADIENTDESCENTOPTIMIZER_H__
#define __GRADIENTDESCENTOPTIMIZER_H__

#include "optimizer.h"

class GradientDescentOptimizer: public Optimizer {
    private:
    Tensor learning_rate;
    public:
    /**
     * Constructor for a GradientDescentOptimizer
     */
    GradientDescentOptimizer(const float n_learning_rate);
    GradientDescentOptimizer(const Tensor n_learning_rate);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    virtual void initialize(Session *session) const {(void)session;};

    TensorNode *minimize(TensorNode *target);
};

#endif