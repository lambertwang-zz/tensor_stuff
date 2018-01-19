/**
 * Train class
 */

#include "train.h"

Train::Train(const Optimizer *n_opt, const TensorNode *val, Tensor n_learning_rate) {
    input.push_back(val);

    optimizer = n_opt;
    graph = new GraphNode(val);
    learning_rate = n_learning_rate;
}

Tensor Train::evaluate(Session *session) const {
    Tensor result;
    for (const TensorNode *n: input) {
        result = session->getEval(n);
    }

    // A map of node tags to edges on the graph. The value of an 
    // edge is the derivative of its parent with respect to itself.
    // (*edgeMap)[a][b] = db/da
    std::map<std::string, std::map<std::string, Tensor>> edgeMap;
    graph->computeEdges(session, &edgeMap);

    // A map of nodes to their derivatives with respect to the root.
    std::map<std::string, Tensor> derivatives;
    graph->computeDerivatives(&edgeMap, &derivatives);

    // Adjust each variable in the session according to its derivative
    std::map<std::string, Tensor> variable_store = session->getVarStore();

    for (auto const& v: variable_store) {
        // Tensor adj = (result * learning_rate) * derivatives[v.first];
        Tensor adj = learning_rate * derivatives[v.first];
        session->setVar(v.first, v.second - adj);
    }

    return result;
}