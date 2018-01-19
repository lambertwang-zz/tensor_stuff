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
     * Sums all of the tensors in the first rank of the input.
     * [1, 2, 3, 4] = 10
     * [[1, 2], [3, 4], [5, 6]] = [1, 2] + [3, 4] + [5, 6]
     *  = [1 + 3 + 5, 2 + 4 + 6] = [9, 12]
     */
    ReduceSum(const TensorNode *val);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

#endif
