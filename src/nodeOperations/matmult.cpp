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
    Tensor output = session->getEval(this);
    bool is_input = false;
    // True if is lhs, false if is rhs
    bool is_lhs = false;
    if (!dx->getTag().compare(lhs->getTag())) {
        is_input = true;
        is_lhs = true;
    } else if (!dx->getTag().compare(rhs->getTag())) {
        is_input = true;
    }

    if (is_input) {
        Tensor t_lhs = session->getEval(lhs),
            t_rhs = session->getEval(rhs);
        std::vector<unsigned int> lhs_shape = t_lhs.getShape(),
            rhs_shape = t_rhs.getShape(),
            d_shape = output.getShape();

        unsigned int out_x = output.getRank() > 1 ? d_shape[1] : 1,
            out_y = output.getRank() > 0 ? d_shape[0] : 1,
            lhs_x = t_lhs.getRank() > 1 ? t_lhs.getShape()[1] : 1; // == rhs_y

        /**
         * For each row and column in the output matrix,
         * calculate the coefficients to produce the matrix from the other factor
         */
        if (is_lhs) {
            d_shape.insert(d_shape.end(),
                lhs_shape.begin(), 
                lhs_shape.end());
        } else {
            d_shape.insert(d_shape.end(),
                rhs_shape.begin(), 
                rhs_shape.end());
        }
        Tensor derivative = Tensor(d_shape);
        derivative.setAllData(0);

        for (unsigned int i = 0; i < out_y; i++) {
            for (unsigned int j = 0; j < out_x; j++) {
                for (unsigned int k = 0; k < lhs_x; k++) {
                    if (is_lhs) {
                        derivative.setData(
                            (j + i * out_x) * t_lhs.getDataCount()
                            + i * lhs_x + k, t_rhs.getData(j + k * out_x));
                    } else {
                        derivative.setData(
                            (j + i * out_x) * t_rhs.getDataCount()
                            + j + k * out_x, t_lhs.getData(k + i * lhs_x));
                    }
                }
            }
        }
        return derivative;
    }
    throw std::invalid_argument("TensorNode '" + dx->getTag() + "' is not a valid input for Node '" + getTag() + "'");
}
