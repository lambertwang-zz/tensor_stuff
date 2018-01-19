/**
 * Variable class
 */

#include "variable.h"

std::string Variable::getDefaultTag() {
    return "variable_";
}

Variable::Variable(const double init, std::string n_tag) {
    createTag(n_tag);
    initial_value = Tensor(init);
}

Variable::Variable(const std::vector<double> data, const std::vector<unsigned int> shape, std::string n_tag) {
    createTag(n_tag);
    initial_value = Tensor(data, shape);
}

Variable::Variable(const std::initializer_list<double>& data, const std::initializer_list<unsigned int>& shape, std::string n_tag) {
    createTag(n_tag);
    initial_value = Tensor(data, shape);
}

Tensor Variable::evaluate() const {
    return initial_value;
}

Tensor Variable::evaluate(Session *session) const {
    return session->getVar(this);
}

void Variable::initialize(Session *session) const {
    session->setVar(tag, initial_value);
}

