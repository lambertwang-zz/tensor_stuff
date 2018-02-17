/**
 * VectorNorm class
 */

#include "vectornorm.h"

std::string VectorNorm::getDefaultTag() {
    return "vectornorm_";
}

VectorNorm::VectorNorm(const TensorNode *n_val, unsigned int n_l) {
    createTag();
    val = n_val;
    l = n_l;
    input.push_back(n_val);
}

Tensor VectorNorm::evaluate() const {
    return val->evaluate().vectorNorm(l);
}

Tensor VectorNorm::evaluate(Session *session) const {
    return session->getEval(val).vectorNorm(l);
}

Tensor VectorNorm::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    if (val->getTag().compare(dx->getTag()) == 0) {
        Tensor input = session->getEval(val);
        std::vector<unsigned int> input_shape = input.getShape(),
            d_shape = std::vector<unsigned int>(input_shape.begin(), input_shape.end() - 1);
        d_shape.insert(d_shape.end(), 
            input_shape.begin(), 
            input_shape.end());

        /**
         * TODO: Rename this. Refers to the cardinality of the
         * subtensor shape of rank 1 less than the input
         */
        unsigned int subtensor_count = 1;
        for (unsigned int i = 0; i < input.getRank() - 1; i++) {
            subtensor_count *= input_shape[i];
        }

        // Ensure constant derivatives always have shape [1]
        if (d_shape.size() < 1) {
            d_shape.push_back(1);
        }
        Tensor derivative = Tensor(d_shape);

        for (unsigned int i = 0; i < subtensor_count; i++) {
            unsigned int deriv_index = i * (input_shape[input.getRank() - 1] + input.getDataCount());
            for (unsigned int j = 0; j < input_shape[input.getRank() - 1]; j++) {
                // TODO: Implement this
                // Currently implemented for l1 norm
                if (input.getData(i * input_shape[input.getRank() - 1] + j) >= 1.0) {
                    derivative.setData(j + deriv_index, 1);
                } else {
                    derivative.setData(j + deriv_index, -1);
                }
            }
        }

        return derivative;
    }
    return Tensor(0);
}


