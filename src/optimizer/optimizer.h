/**
 * Optimizer Class Header
 * 
 */

#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include "node/tensorNode.h"

class Optimizer {
    private:
    Tensor learning_rate;
    public:
    /**
     * Constructor for an Optimizer
     */
    Optimizer();
    // Optimizer(const Tensor n_learning_rate);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    virtual void initialize(Session *session) const {(void)session;};
};

#endif
