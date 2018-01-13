/**
 * TensorNode class
 */

#include "tensorNode.h"

std::string TensorNode::createTag(std::string n_tag) {
    static unsigned int node_tag_iterator = 0;
    if (n_tag.empty()) {
        tag = getDefaultTag() + std::to_string(node_tag_iterator++);
        return tag;
    }
    tag = n_tag;
    return tag;
}

std::string TensorNode::setTag(std::string n_tag) {
    tag = n_tag;
    return tag;
}

std::string TensorNode::getDefaultTag() {
    return DEFAULT_TAG_PRE;
}

TensorNode::TensorNode() {
    createTag();
}

void TensorNode::initialize(Session *session) const {
    (void)session;
}

Tensor TensorNode::derivative(const TensorNode *dx, Session *session) const {
    (void)dx; (void)session;
    throw std::invalid_argument("TensorNode::derivative(): Cannot compute derivative of node tag: " + tag);
}

std::string TensorNode::getTag() const {
    return tag;
}

std::vector<const TensorNode *> TensorNode::getInput() const {
    return input;
}

std::ostream& operator<<(std::ostream& out, const TensorNode *data) {
    out << data->evaluate();
    return out;
}