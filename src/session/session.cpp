#include "session.h"

#include <algorithm>

Session::Session() {
    is_initialized = false;
    placeholder_store = NULL;
}

void Session::initialize(TensorNode *node) {
    is_initialized = true;
    variable_store.clear();

    /**
     * Initializes the session by adding all of the variables to the variable store
     * Initializes the nodes using a depth-first search on the node inputs
     */
    std::list<std::string> initialized;
    std::list<const TensorNode *> queue;

    // Add initial element to queue
    queue.push_back(node);
    initialized.push_back(node->getTag());

    while (!queue.empty()) {
        // get and delete last element (depth-first)
        const TensorNode *current = queue.back();
        queue.pop_back();

        current->initialize(this);

        for (const TensorNode *n: current->getInput()) {
            // Check if element was not already added to queue
            if (std::find(initialized.begin(), initialized.end(), n->getTag()) == initialized.end()) {
                queue.push_back(n);
                initialized.push_back(n->getTag());
            }
        }
    }

    run_count = 0;
}

Tensor Session::run(TensorNode *node, std::map<std::string, Tensor> placeholder) {
    if (!is_initialized) {
        initialize(node);
    }

    // Evaluate root node
    Tensor result;
    // Set placeholder store
    placeholder_store = &placeholder;
    result = node->evaluate(this);
    prev_result = result;
    run_count++;
    // Reset placeholder store
    placeholder_store = NULL;

    eval_store.clear();

    return result;
}

Tensor Session::getPrev() {
    return prev_result;
}

Tensor Session::getVar(const TensorNode *node) {
    try {
        Tensor output = variable_store.at(node->getTag());
        return output;
    } catch (const std::out_of_range&) {
        throw std::out_of_range("Session::getVar(): Variable with tag " + node->getTag() + " not initialized.");
    }
}

Tensor Session::getVarTag(std::string tag) {
    try {
        Tensor output = variable_store.at(tag);
        return output;
    } catch (const std::out_of_range&) {
        throw std::out_of_range("Session::getVar(): Variable with tag " + tag + " not initialized.");
    }
}

void Session::setVar(std::string tag, Tensor val) {
    try {
        variable_store.at(tag);
        variable_store[tag].copyFrom(val);
    } catch (const std::out_of_range&) {
        variable_store[tag] = Tensor(val);
    }
}

Tensor Session::getPlaceholder(const TensorNode *node) {
    if (placeholder_store == NULL) {
        throw std::invalid_argument("Session::getPlaceholder(): No placeholder store found");
    }
    try {
        Tensor output = placeholder_store->at(node->getTag());
        return output;
    } catch (const std::out_of_range&) {
        throw std::out_of_range("Session::getPlaceholder(): Placeholder with tag " + node->getTag() + " not initialized.");
    }
}

Tensor Session::getEval(const TensorNode *node) {
    try {
        Tensor output = eval_store.at(node->getTag());
        return output;
    } catch (const std::out_of_range&) {
        eval_store[node->getTag()] = Tensor(node->evaluate(this));
        return eval_store[node->getTag()];
    }
}

std::map<std::string, Tensor> Session::getVarStore() {
    return variable_store;
}