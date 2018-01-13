/**
 * Placeholder class
 */

#include "placeholder.h"

std::string Placeholder::getDefaultTag() {
    return "placeholder_";
}

Placeholder::Placeholder(std::string n_tag) {
    if (n_tag.empty()) {
        throw std::invalid_argument("Placeholder::Placeholder(): Tag cannot be empty.");
    }
    setTag(n_tag);
}

Tensor Placeholder::evaluate() const {
    throw std::invalid_argument("Placeholder::evauluate(): Session cannot be null.");
}

Tensor Placeholder::evaluate(Session *session) const {
    return session->getPlaceholder(this);
}
