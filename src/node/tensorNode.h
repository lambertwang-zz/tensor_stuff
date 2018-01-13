/**
 * Abstract TensorNode Class Header
 * 
 */

#ifndef __TENSORNODE_H__
#define __TENSORNODE_H__

// Library
#include <vector>

// 
#include "tensor/tensor.h"
#include "session/session.h"

class Session;

#define DEFAULT_TAG_PRE "node_"

class TensorNode {
    protected:
    std::vector<const TensorNode *> input;

    std::string tag;
    std::string createTag(std::string n_tag = "");
    std::string setTag(std::string n_tag);
    virtual std::string getDefaultTag();

    public:
    TensorNode();

    virtual Tensor evaluate() const =0;
    virtual Tensor evaluate(Session *session) const =0;
    virtual void initialize(Session *session) const;
    /**
     * Computes the partial derivative of this node with respect to one of its inputs.
     */
    virtual Tensor derivative(const TensorNode *dx, Session *session) const;

    std::string getTag() const;
    std::vector<const TensorNode *> getInput() const;
};

std::ostream& operator<<(std::ostream& out, const TensorNode *data);

#endif