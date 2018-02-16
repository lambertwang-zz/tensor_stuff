/**
 * Mult Class Header
 * 
 */

#ifndef __MULT_H__
#define __MULT_H__

#include "node/tensorNode.h"

class Mult: public TensorNode {
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a Mult
     */
    Mult(std::initializer_list<const TensorNode *> values);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    /**
     * Returns the product of all other inputs
     */
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

Mult *operator*(const TensorNode &lhs, const TensorNode& rhs);

#endif
