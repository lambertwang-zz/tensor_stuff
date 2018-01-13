/**
 * ReduceSum Class Header
 * 
 */

#ifndef __REDUCESUM_H__
#define __REDUCESUM_H__

#include "node/tensorNode.h"

class ReduceSum: public TensorNode {
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a ReduceSum
     */
    ReduceSum(const TensorNode *val);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

#endif
