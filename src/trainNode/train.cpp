/**
 * Train class
 */

#include "train.h"

#include "tensoralgebra.h"

Train::Train(const TensorNode *val, Tensor n_learning_rate) {
    input.push_back(val);

    graph = new GraphNode(val);
    learning_rate = n_learning_rate;
}

Tensor Train::evaluate(Session *session) const {
    Tensor result;
    for (const TensorNode *n: input) {
        result = session->getEval(n);
    }

#ifdef DEBUG
    std::cout << "DEBUG: Beginning Training Step" << std::endl;
#endif

    // A map of node tags to edges on the graph. The value of an 
    // edge is the derivative of its parent with respect to itself.
    // (*edgeMap)[a][b] = db/da
    std::map<std::string, std::map<std::string, Tensor>> edgeMap;
    graph->computeEdges(session, &edgeMap);
#ifdef DEBUG
    std::cout << "DEBUG: Computed Edges" << std::endl;
#endif

    // A map of nodes to their derivatives with respect to the root.
    std::map<std::string, Tensor> derivatives;
    graph->computeDerivatives(&edgeMap, &derivatives);
#ifdef DEBUG
    std::cout << "DEBUG: Computed Derivatives" << std::endl;
#endif

    // Adjust each variable in the session according to its derivative
    std::map<std::string, Tensor> variable_store = session->getVarStore();

    for (auto const& v: variable_store) {
        // Tensor adj = derivatives[v.first] * (learning_rate * result);
        Tensor adj = derivatives[v.first] * learning_rate;
#ifdef DEBUG
        std::cout << "DEBUG: variable '" << v.first << " ' initial val " << v.second<< std::endl;
        std::cout << "DEBUG: Adjusting variable '" << v.first << "' by " << adj << std::endl;
#endif
        session->setVar(v.first, v.second - adj);
    }

    return result;
}