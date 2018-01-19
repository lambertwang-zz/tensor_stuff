/**
 * Train Class Header
 * 
 */

#ifndef __TRAIN_H__
#define __TRAIN_H__

#include "node/tensorNode.h"
#include "computationalGraph/graphNode.h"
#include "optimizer/optimizer.h"

class Train: public TensorNode {
    private:
    const Optimizer *optimizer;
    GraphNode *graph;
    Tensor learning_rate;
    public:
    /**
     * Constructor for a Train
     */
    Train(const Optimizer *n_opt, const TensorNode *val, Tensor n_learning_rate);

    Tensor evaluate() const {
        throw std::invalid_argument("Train::evalute(): Training node cannot be evaluated without a session.");
    }

    Tensor evaluate(Session *session) const;
};

#endif
