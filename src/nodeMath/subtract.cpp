/**
 * Subtract class
 */

#include "subtract.h"

std::string Subtract::getDefaultTag() {
    return "subtract_";
}

Subtract::Subtract(std::initializer_list<const TensorNode *> values) {
    createTag();
    for (const TensorNode *n: values) {
        input.push_back(n);
    }
}

Tensor Subtract::evaluate() const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(n->evaluate());
        } else {
            output -= n->evaluate();
        }
    }
    return output;
}

Tensor Subtract::evaluate(Session *session) const {
    Tensor output;
    bool is_initialized = false;
    for (const TensorNode *n: input) {
        if (!is_initialized) {
            is_initialized = true;
            output = Tensor(session->getEval(n));
        } else {
            output -= session->getEval(n);
        }
    }
    return output;
}

/**
 * TODO: See Add::derivative():
 */
Tensor Subtract::derivative(const TensorNode *dx, Session *session) const {
    (void)session;
    if (input.size() == 0) {
        return Tensor(0);
    }
    // If input is the target, return 1
    if (input[0]->getTag().compare(dx->getTag()) == 0) {
        Tensor input_eval = session->getEval(input[0]);
        std::vector<unsigned int> d_shape = input_eval.getShape();
        d_shape.insert(d_shape.end(), 
            d_shape.begin(), 
            d_shape.end());

        Tensor derivative = Tensor(d_shape);
        derivative.setAllData(0);
        unsigned int count = input_eval.getDataCount();
        for (unsigned int i = 0; i < count; i++) {
            derivative.setData(i + i * count, 1);
        }
        return derivative;
    }
    for (const TensorNode *n: input) {
        if (n->getTag().compare(dx->getTag()) == 0) {
            Tensor input_eval = session->getEval(n);
            std::vector<unsigned int> d_shape = input_eval.getShape();
            d_shape.insert(d_shape.end(), 
                d_shape.begin(), 
                d_shape.end());

            Tensor derivative = Tensor(d_shape);
            derivative.setAllData(0);
            unsigned int count = input_eval.getDataCount();
            for (unsigned int i = 0; i < count; i++) {
                derivative.setData(i + i * count, -1);
            }
            return derivative;
        }
    }
    return Tensor(0);
}

Subtract *operator-(const TensorNode& lhs, const TensorNode& rhs) {
    return new Subtract({&lhs, &rhs});
}
