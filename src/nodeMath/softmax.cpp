/**
 * SoftMax class
 */
#include "softmax.h"

std::string SoftMax::getDefaultTag() {
    return "softmax_";
}

SoftMax::SoftMax(const TensorNode *n_val) {
    createTag();
    val = n_val;
    input.push_back(n_val);
}

Tensor SoftMax::evaluate() const {
    return val->evaluate().softMax();
}

Tensor SoftMax::evaluate(Session *session) const {
    return session->getEval(val).softMax();
}

Tensor SoftMax::derivative(const TensorNode *dx, Session *session) const {
    (void)dx;
    if (val->getTag().compare(dx->getTag()) == 0) {
        Tensor softmax_eval = session->getEval(this);
        std::vector<unsigned int> d_shape = softmax_eval.getShape();
        d_shape.insert(d_shape.end(), 
            d_shape.begin(), 
            d_shape.end());

        Tensor derivative = Tensor(d_shape);
        derivative.setAllData(0);
        unsigned int subtensor_count = 1;
        unsigned int reduce_count = d_shape[softmax_eval.getRank() - 1];
        for (unsigned int i = 0; i < softmax_eval.getRank() - 1; i++) {
            subtensor_count *= d_shape[i];
        }

        for (unsigned int i = 0; i < subtensor_count; i++) {
            for (unsigned int j = 0; j < reduce_count; j++) {
                /**
                 * D_k S_j = 
                 *      S_j (1 - S_k) for j == k
                 *      - S_k S_j     for j != k
                 */
                unsigned int subtensor_index = softmax_eval.getDataCount() * (j + i * reduce_count) + i * reduce_count;
                double s_j = softmax_eval.getData(i * reduce_count + j);
                for (unsigned int k = 0; k < reduce_count; k++) {
                    double d_val = 0, s_k = softmax_eval.getData(i * reduce_count + k);
                    if (j == k) {
                        d_val = s_j * (1 - s_k);
                    } else {
                        d_val = - s_j * s_k;
                    }
                    derivative.setData(subtensor_index + k, d_val);
                }
            }
        }
        return derivative;
    }
    throw std::invalid_argument("TensorNode '" + dx->getTag() + "' is not a valid input for Node '" + getTag() + "'");
}
