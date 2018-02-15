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
        GraphNode *graph_node = findNode(n->getTag());
        if (graph_node == NULL) {
            children.push_back(new GraphNode(n, root));
        } else {
            children.push_back(graph_node);
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
        Tensor derivative =node->derivative(n->node, session);
        (*edgeMap)[n->getTag()][getTag()] = derivative;
#ifdef DEBUG
        std::cout << "Computing derivative of '" << getTag() << "'" << "with respect to node '" << n->getTag() << "'" << std::endl;
        std::cout << "Derivative: " << derivative << std::endl;
#endif
        n->computeEdges(session, edgeMap);
    }
}

void GraphNode::computeDerivatives(
    std::map<std::string, std::map<std::string, Tensor>> *edgeMap,
    std::map<std::string, Tensor> *derivatives) {

    // Check if derivative already exists
    try {
        (*derivatives).at(getTag());
        return;
    } catch (const std::out_of_range&) {
        // continue
    }


    if (this == root) {
        // droot/droot = 1
        // TODO: If root shape is not [], then this is false
        (*derivatives)[getTag()] = Tensor(1);
    } else {
        Tensor derivative;
        bool is_initialized = false;
        for (auto const& x: (*edgeMap)[getTag()]) {
            // x.first is parent tag
            // x.second is dparent/dthis
            // Check if target parent has derivative calculated
            try {
                (*derivatives).at(x.first);
            } catch (const std::out_of_range&) {
                findNode(x.first)->computeDerivatives(edgeMap, derivatives);
            }

            // Sum all paths to parent from this node
            if (!is_initialized) {
                is_initialized = true;
                derivative = Tensor((*derivatives)[x.first] * x.second);
            } else {
                derivative += (*derivatives)[x.first] * x.second;
            }
        }
        (*derivatives)[getTag()] = derivative;
#ifdef DEBUG
        std::cout << "Computing derivative of root with respect to node '" << getTag() << "'" << std::endl;
        std::cout << "Derivative: " << derivative << std::endl;
#endif
    }

    for (GraphNode *n: children) {
        n->computeDerivatives(edgeMap, derivatives);
    }
}
