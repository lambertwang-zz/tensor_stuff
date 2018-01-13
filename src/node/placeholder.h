/**
 * Placeholder Class Header
 * 
 */

#ifndef __PLACEHODLER_H__
#define __PLACEHOLDER_H__

#include "tensorNode.h"

class Placeholder: public TensorNode {
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a Placeholder
     */
    Placeholder(std::string tag);

    /**
     * Virtual evaluate function
     */
    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
};

#endif
