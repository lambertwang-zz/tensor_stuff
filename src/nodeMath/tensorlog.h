/**
 * TensorLog Class Header
 * element-wise log function
 */

#ifndef __TENSORLOG_H__
#define __TENSORLOG_H__

#include "node/tensorNode.h"

class TensorLog: public TensorNode {
    protected:
    std::string getDefaultTag();
    public:
    /**
     * Constructor for a TensorLog
     */
    TensorLog(const TensorNode *value);

    Tensor evaluate() const;
    Tensor evaluate(Session *session) const;
    /**
     * Always returns 1
     */
    Tensor derivative(const TensorNode *dx, Session *session) const;
};

#endif
