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
    Constant(const double val);
    Constant(const std::vector<double> data, const std::vector<unsigned int> shape);
    Constant(const std::initializer_list<double>& data, const std::initializer_list<unsigned int>& shape);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
};

#endif
