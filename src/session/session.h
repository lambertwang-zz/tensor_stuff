/**
 * Session Class Header
 * 
 */

#ifndef __SESSION_H__
#define __SESSION_H__

// Libraries
#include <list>
#include <map>

#include <node/tensorNode.h>

class TensorNode;

class Session {
    private:
    bool is_initialized;
    /**
     * Map of node id to the variable values of the session.
     */
    std::map<std::string, Tensor> variable_store;

    /**
     * Pointer to map of node id to current running input for the placeholder node.
     */
    std::map<std::string, Tensor> *placeholder_store;

    /**
     * Map of node id to evaluation results of the current run.
     */
    std::map<std::string, Tensor> eval_store;

    void initializeNode(TensorNode *node);

    unsigned int run_count;
    Tensor prev_result;

    public:
    /**
     * Constructor for a Session
     */
    Session();

    /**
     * Clears the session stores and initializes the variables.
     */
    void initialize(TensorNode *node);

    /**
     * Evaluates the nodes within the session
     */
    Tensor run(TensorNode *node, std::map<std::string, Tensor> placeholder = std::map<std::string, Tensor> ());
    /**
     * Evaluates the nodes for multiple values of x simultaneously.
     * Accepts a map of node tag to a list of values for each node.
     */
    // std::list<Tensor> run(TensorNode *node, const std::map<std::string, std::list<Tensor *>> placeholders);

    Tensor getPrev();
    Tensor getVar(const TensorNode *node);
    Tensor getVarTag(std::string tag);
    void setVar(std::string tag, Tensor t);
    Tensor getPlaceholder(const TensorNode *node);
    Tensor getEval(const TensorNode *node);

    std::map<std::string, Tensor> getVarStore();
};

#endif
