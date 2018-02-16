/**
 * Add class
 */

#include "add.h"

std::string Add::getDefaultTag() {
    return "add_";
}

Add::Add(const std::initializer_list<const TensorNode *> values) {
    createTag();
    for (const TensorNode *n: values) {
        input.push_back(n);
    }
}

Tensor Add::evaluate() const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(n->evaluate());
        } else {
            output += n->evaluate();
        }
    }
    return output;
}

Tensor Add::evaluate(Session *session) const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(session->getEval(n));
        } else {
            output += session->getEval(n);
        }
    }
    return output;
}

/**
 * TODO: Test and fix this function. Create general function
 * for re-use between other mathematics nodes.
 */
Tensor Add::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            Tensor input_eval = session->getEval(this),
                dx_eval = session->getEval(n);
            std::vector<unsigned int> d_shape = input_eval.getShape(),
                dx_shape = dx_eval.getShape();
            d_shape.insert(d_shape.end(), 
                dx_shape.begin(), 
                dx_shape.end());
            
            // Ensure constant derivatives always have shape [1]
            if (d_shape.size() < 1) {
                d_shape.push_back(1);
            }

            Tensor derivative = Tensor(d_shape);
            derivative.setAllData(0);
            unsigned int in_count = input_eval.getDataCount();
            unsigned int dx_count = dx_eval.getDataCount();
            for (unsigned int i = 0; i < in_count; i++) {
                derivative.setData((i % dx_count) + i * dx_count, 1);
            }
            return derivative;
        }
    }
    throw std::invalid_argument("TensorNode '" + dx->getTag() + "' is not a valid input for Node '" + getTag() + "'");
}

Add *operator+(const TensorNode& lhs, const TensorNode& rhs) {
    return new Add({&lhs, &rhs});
}
