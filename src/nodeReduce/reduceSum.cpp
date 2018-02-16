/**
 * ReduceSum class
 */

#include "reduceSum.h"

std::string ReduceSum::getDefaultTag() {
    return "reducesum_";
}

ReduceSum::ReduceSum(const TensorNode *val) {
    createTag();
    input.push_back(val);
}

Tensor ReduceSum::evaluate() const {
    for (const TensorNode *n: input) {
        return n->evaluate().reduceSum();
    }

    return Tensor();
}

Tensor ReduceSum::evaluate(Session *session) const {
    for (const TensorNode *n: input) {
        return session->getEval(n).reduceSum();
    }
    return Tensor();
}

Tensor ReduceSum::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            Tensor input = session->getEval(n);
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
                    derivative.setData(j + deriv_index, 1);
                }
            }

            return derivative;
        }
    }
    return Tensor(0);
}

