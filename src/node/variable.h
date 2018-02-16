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
    Variable(std::string n_tag = "");
    Variable(const double init = 0, std::string n_tag = "");
    Variable(const Tensor init, std::string n_tag);
    Variable(const std::vector<double> data, const std::vector<unsigned int> shape, std::string n_tag = "");
    Variable(const std::initializer_list<unsigned int>& shape, std::string n_tag = "");
    Variable(const std::initializer_list<double>& data, const std::initializer_list<unsigned int>& shape, std::string n_tag = "");

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    /**
     * Virtual inizialize function
     */
    void initialize(Session *session) const;
};

#endif
