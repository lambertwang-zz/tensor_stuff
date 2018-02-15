/**
 * Matmult Class Header
 * 
 */

#ifndef __MATMULT_H__
#define __MATMULT_H__

#include "node/tensorNode.h"

class MatMult: public TensorNode {
    protected:
    const TensorNode *lhs;
    const TensorNode *rhs;
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a MatMult
     */
    MatMult(const TensorNode *n_lhs, const TensorNode *n_rhs);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    /**
     * Returns the product of all other inputs
     */
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

#endif
