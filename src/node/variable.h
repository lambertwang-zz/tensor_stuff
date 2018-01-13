/**
 * Variable Class Header
 * 
 */

#ifndef __VARIABLE_H__
#define __VARIABLE_H__

#include "tensorNode.h"

class Variable: public TensorNode {
    private:
    Tensor initial_value;
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a Variable
     */
    Variable(const float init = 0, std::string n_tag = "");
    Variable(const std::vector<float> data, const std::vector<int> shape, std::string n_tag = "");
    Variable(const std::initializer_list<float>& data, const std::initializer_list<int>& shape, std::string n_tag = "");

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    /**
     * Virtual inizialize function
     */
    void initialize(Session *session) const;
};

#endif
