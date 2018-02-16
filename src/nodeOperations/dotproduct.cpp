/**
 * DotProduct class
 */

#include "dotproduct.h"

std::string DotProduct::getDefaultTag() {
    return "dotproduct_";
}

DotProduct::DotProduct(std::initializer_list<const TensorNode *> values) {
    createTag();
    for (const TensorNode *n: values) {
        input.push_back(n);
    }
}

Tensor DotProduct::evaluate() const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(n->evaluate());
        } else {
            output = Tensor::dotProduct(output, n->evaluate());
        }
    }
    return output;
}

Tensor DotProduct::evaluate(Session *session) const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(session->getEval(n));
        } else {
            output = Tensor::dotProduct(output, session->getEval(n));
        }
    }
    return output;
}

/**
 * TODO: See Add::derivative():
 * Tensors must have identical shapes
 */
Tensor DotProduct::derivative(const TensorNode *dx, Session *session) const {
    Tensor output;
    bool is_initialized = false;
    bool is_input = false;
    for (const TensorNode *n: input) {
        if (dx->getTag().compare(n->getTag())) {
            if (!is_initialized) {
                is_initialized = true;
                output = Tensor(session->getEval(n));
            } else {
                output = Tensor::dotProduct(output, session->getEval(n));
            }
        } else {
            is_input = true;
        }
    }

    if (is_input) {
        std::vector<unsigned int> d_shape = output.getShape();
        d_shape.insert(d_shape.end(), 
            d_shape.begin(), 
            d_shape.end());

        Tensor derivative = Tensor(d_shape);
        derivative.setAllData(0);
        unsigned int count = output.getDataCount();
        for (unsigned int i = 0; i < count; i++) {
            derivative.setData(i * (1 + count), output.getData(i));
        }
        return derivative;
    } else {
        return Tensor(0);
    }
}
