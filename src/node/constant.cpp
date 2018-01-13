/**
 * Constant class
 */

#include "constant.h"

std::string Constant::getDefaultTag() {
    return "constant_";
}

Constant::Constant() {
    createTag();
    constant = Tensor();
}

Constant::Constant(const float val) {
    createTag();
    constant = Tensor(val);
}

Constant::Constant(const std::vector<float> data, const std::vector<int> shape) {
    createTag();
    constant = Tensor(data, shape);
}

Constant::Constant(const std::initializer_list<float>& data, const std::initializer_list<int>& shape) {
    createTag();
    constant = Tensor(data, shape);
}

Tensor Constant::evaluate() const {
    return constant;
}

Tensor Constant::evaluate(Session *session) const {
    session = session;
    return constant;
}
