/**
 * MatMult class
 */

#include "matmult.h"

std::string MatMult::getDefaultTag() {
    return "matmult_";
}

MatMult::MatMult(const TensorNode *n_lhs, const TensorNode *n_rhs) {
    createTag();
    lhs = n_lhs;
    rhs = n_rhs;
    input.push_back(lhs);
    input.push_back(rhs);
}

Tensor MatMult::evaluate() const {
    return Tensor(lhs->evaluate() * rhs->evaluate());
}

Tensor MatMult::evaluate(Session *session) const {
    return Tensor(session->getEval(lhs) * session->getEval(rhs));
}

/**
 * TODO: See Add::derivative():
 */
Tensor MatMult::derivative(const TensorNode *dx, Session *session) const {
    Tensor output;
    bool is_initialized = false;
    bool is_input = false;
    for (const TensorNode *n: input) {
        if (dx->getTag().compare(n->getTag())) {
            if (!is_initialized) {
                is_initialized = true;
                output = Tensor(session->getEval(n));
            } else {
                output *= session->getEval(n);
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
