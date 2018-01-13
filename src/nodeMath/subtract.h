/**
 * Subtract Class Header
 * 
 */

#ifndef __SUBTRACT_H__
#define __SUBTRACT_H__

#include "node/tensorNode.h"

class Subtract: public TensorNode {
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a Subtract
     * The first value is positive
     * { i1, i2, i3, ... , in } = + i1 - i2 - i3 - ... - in
     */
    Subtract(std::initializer_list<const TensorNode *> values);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;

    /**
     * Returns the 1 if the node it the first input, -1 otherwise
     */
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

Subtract *operator-(const TensorNode& lhs, const TensorNode& rhs);

#endif
