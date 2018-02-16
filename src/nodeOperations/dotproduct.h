/**
 * DotProduct Class Header
 * 
 */

#ifndef __DOTPRODUCT_H__
#define __DOTPRODUCT_H__

#include "node/tensorNode.h"

class DotProduct: public TensorNode {
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a DotProduct
     */
    DotProduct(std::initializer_list<const TensorNode *> values);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    /**
     * Returns the product of all other inputs
     */
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

#endif
