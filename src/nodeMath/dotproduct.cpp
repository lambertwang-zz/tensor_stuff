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
        Tensor input_eval = session->getEval(dx);
        std::vector<unsigned int> d_shape = input_eval.getShape(),
            output_shape = output.getShape();
        d_shape.insert(d_shape.end(), 
            d_shape.begin(), 
            d_shape.end());
        d_shape.insert(d_shape.end(), 
            output_shape.begin(), 
            output_shape.end());

        Tensor derivative = Tensor(d_shape);
        derivative.setAllData(0);
        unsigned int count = input_eval.getDataCount();
        unsigned int out_count = output.getDataCount();
        for (unsigned int i = 0; i < count; i++) {
            for (unsigned int j = 0; j < out_count; j++) {
                derivative.setData(j + out_count * (i + i * count), output.getData(j));
            }
        }
        return derivative;
    } else {
        return Tensor(0);
    }
}
