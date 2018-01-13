/**
 * Constant Class Header
 * 
 */

#ifndef __CONSTANT_H__
#define __CONSTANT_H__

#include "tensorNode.h"

class Constant: public TensorNode {
    private:
    Tensor constant;
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a Constant
     */
    Constant();
    Constant(const float val);
    Constant(const std::vector<float> data, const std::vector<int> shape);
    Constant(const std::initializer_list<float>& data, const std::initializer_list<int>& shape);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
};

#endif
