/**
 * GraphNode class
 */

#include "graphNode.h"

#include <algorithm>

GraphNode::GraphNode(const TensorNode *n_node, GraphNode *n_root) {
    node = n_node;
    // derivative = Tensor(0);

    if (n_root == NULL) {
        root = this;
    } else {
        root = n_root;
    }

    // Construct the rest of the nodes in the graph 
    for (const TensorNode *n: node->getInput()) {
        if (findNode(n->getTag()) == NULL) {
            children.push_back(new GraphNode(n, root));
        }
    }
}

std::string GraphNode::getTag() const {
    return node->getTag();
}

GraphNode *GraphNode::findNode(std::string tag) {
    // Use a dfs
    std::list<GraphNode *> queue;
    std::list<std::string> visited;
    queue.push_back(root);
    visited.push_back(root->getTag());

    while (!queue.empty()) {
        GraphNode *current = queue.back();
        queue.pop_back();

        if (current->getTag().compare(tag) == 0) {
            return current;
        }

        for (GraphNode *n: current->children) {
            // Check if element was not already added to queue
            if (std::find(visited.begin(), visited.end(), n->getTag()) == visited.end()) {
                queue.push_back(n);
                visited.push_back(n->getTag());
            }
        }
    }

    return NULL;
}

void GraphNode::computeEdges(
    Session *session,
    std::map<std::string, std::map<std::string, Tensor>> *edgeMap) {

    for (GraphNode *n: children) {
        (*edgeMap)[n->getTag()][getTag()] = node->derivative(n->node, session);
        n->computeEdges(session, edgeMap);
    }
}

void GraphNode::computeDerivatives(
    std::map<std::string, std::map<std::string, Tensor>> *edgeMap,
    std::map<std::string, Tensor> *derivatives) {

    // Check if derivative already exists
    try {
        (*derivatives).at(getTag());
        std::cout << "Derivative already exists" << std::endl;
        return;
    } catch (const std::out_of_range&) {
        // continue
    }

    if (this == root) {
        std::cout << "Assigning root derivative" << std::endl;
        (*derivatives)[getTag()] = Tensor(1);
    } else {
        Tensor derivative;
        bool is_initialized = false;
        std::cout << "Computing derivative of " << getTag() << std::endl;
        for (auto const& x: (*edgeMap)[getTag()]) {
            std::cout << "Derivative with respect to " << x.first << std::endl;
            // Check if target parent has derivative calculated
            try {
                (*derivatives).at(x.first);
            } catch (const std::out_of_range&) {
                std::cout << "Parent Derivative not computed. Computing now" << std::endl;
                findNode(x.first)->computeDerivatives(edgeMap, derivatives);
            }

            std::cout << x.second << std::endl;
            std::cout << (*derivatives)[x.first] << std::endl;
            if (!is_initialized) {
                is_initialized = true;
                derivative = Tensor(x.second * (*derivatives)[x.first]);
            } else {
                derivative += x.second * (*derivatives)[x.first];
            }
        }
        (*derivatives)[getTag()] = derivative;
        std::cout << "Derivative: " << derivative << std::endl;
    }

    for (GraphNode *n: children) {
        n->computeDerivatives(edgeMap, derivatives);
    }
}

Tensor GraphNode::computeDerivative(std::string tag, Session *session, Tensor val) {
    return Tensor();
}
