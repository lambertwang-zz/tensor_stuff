/**
 * ReduceMean Class Header
 * 
 */

#ifndef __REDUCEMEAN_H__
#define __REDUCEMEAN_H__

#include "node/tensorNode.h"

class ReduceMean: public TensorNode {
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a ReduceMean
     * Sums all of the tensors in the first rank of the input.
     * [1, 2, 3, 4] = 2.5
     */
    ReduceMean(const TensorNode *val);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

#endif
