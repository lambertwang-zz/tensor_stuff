/**
 * Train class
 */

#include "train.h"

Train::Train(const Optimizer *n_opt, const TensorNode *val) {
    input.push_back(val);

    optimizer = n_opt;
    graph = new GraphNode(val);
}

Tensor Train::evaluate(Session *session) const {
    Tensor result;
    for (const TensorNode *n: input) {
        result = session->getEval(n);
    }

    // A map of node tags to edges on the graph. The value of an 
    // edge is the derivative of its parent with respect to itself.
    // (*edgeMap)[a][b] = db/da

    std::cout << "Computing Edges" << std::endl;

    std::map<std::string, std::map<std::string, Tensor>> edgeMap;

    graph->computeEdges(session, &edgeMap);

    std::cout << "Computing Derivatives" << std::endl;

    // A map of nodes to their derivatives with respect to the root.
    std::map<std::string, Tensor> derivatives;

    graph->computeDerivatives(&edgeMap, &derivatives);

    std::cout << "Adjusting variables" << std::endl;

    std::map<std::string, Tensor> *variable_store = session->getVarStore();

    for (auto const& v: (*variable_store)) {
        std::cout << "Var : " << v.second << std::endl;
        std::cout << "Rate: " << Tensor(0.01) << std::endl;
        std::cout << "erf : " << result << std::endl;
        std::cout << "dv/de:" << derivatives[v.first] << std::endl;
        std::cout << "Computing t1" << std::endl;
        Tensor t1 = Tensor(Tensor(0.01) * result);
        std::cout << t1 << std::endl;
        std::cout << "Computing t2" << std::endl;
        Tensor t2 = Tensor(t1 * derivatives[v.first]);
        std::cout << t2 << std::endl;
        std::cout << "Computing t3" << std::endl;
        Tensor t3 = Tensor(v.second - t2);
        std::cout << t3 << std::endl;
        (*variable_store)[v.first] = v.second - Tensor(0.1) * result * derivatives[v.first];
    }

    return result;
}