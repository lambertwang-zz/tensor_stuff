/**
 * Vector Norm Class Header
 * 
 */

#ifndef __VECTORNORM_H__
#define __VECTORNORM_H__

#include "node/tensorNode.h"

class VectorNorm: public TensorNode {
    private:
    const TensorNode *val;
    unsigned int l;
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a Vector Norm
     * The L exponent defaults to 2 for the magnitude of the vector.
     * 
     */
    VectorNorm(const TensorNode *n_val, unsigned int n_l = 2);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

#endif
