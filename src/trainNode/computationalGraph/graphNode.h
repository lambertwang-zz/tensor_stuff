/**
 * GraphNode Class Header
 * This is a rooted graph
 * The root is the output node
 */

#ifndef __GRAPHNODE_H__
#define __GRAPHNODE_H__

#include "node/tensorNode.h"

class GraphNode {
    private:
    /**
     * The TensorNode associated with this GraphNode
     */
    const TensorNode *node;
    /**
     * Then node's children
     */
    std::vector<GraphNode *> children;
    /**
     * The root of the graph.
     */
    GraphNode *root;
    // The derivative of this node in respect to the root (output) node
    // Tensor derivative;

    // A map of node tags to the derivative values of its children
    // std::map<std::string, Tensor> edges;

    /**
     * A list of lists of GraphNodes
     * [[a, b, c], [d, e, f]]
     * The derivative of the root node with respect to this one is calculated:
     * (a + b + c) * (d * e * f)
     */
    // std::vector<std::vector<GraphNode *>> factors;

    /**
     * A list of paths feeding into the node.
     */
    // std::vector<GraphNode *> paths;
    public:
    /**
     * Constructor for a GraphNode
     */
    // GraphNode(const TensorNode *n_node, Session *n_session, GraphNode *n_root = NULL);
    GraphNode(const TensorNode *n_node, GraphNode *n_root = NULL);

    /**
     * Returns the tag of the node
     */
    std::string getTag() const;

    /**
     * Finds a node in the graph by tag.
     */
    GraphNode *findNode(std::string tag);

    /**
     * Computes derivatives of each node with respect to its inputs using
     * each TensorNode's defined derivative functions.
     */
    void computeEdges(
        Session *session,
        std::map<std::string, std::map<std::string, Tensor>> *edgeMap);

    void computeDerivatives(
        std::map<std::string, std::map<std::string, Tensor>> *edgeMap,
        std::map<std::string, Tensor> *derivatives);

    /**
     * Computes the derivative of this node with respect to a target
     * Uses reverse mode differentiation
     */
    Tensor computeDerivative(std::string tag, Session *session, Tensor val = Tensor(0));
};

#endif
