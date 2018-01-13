/**
 * Clamp Class Header
 * 
 */

#ifndef __CLAMP_H__
#define __CLAMP_H__

#include "node/tensorNode.h"

class Clamp: public TensorNode {
    private:
    float min;
    float max;
    public:
    /**
     * Constructor for a Clamp
     */
    Clamp(const TensorNode *val, float n_min = 0.0, float n_max = 1.0);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
};

#endif
