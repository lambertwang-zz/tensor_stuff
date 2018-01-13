/**
 * Add Class Header
 * 
 */

#ifndef __ADD_H__
#define __ADD_H__

#include "node/tensorNode.h"

class Add: public TensorNode {
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for an Add
     */
    Add(const std::initializer_list<const TensorNode *> values);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    /**
     * Always returns 1
     */
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

Add *operator+(const TensorNode& lhs, const TensorNode& rhs);

#endif
