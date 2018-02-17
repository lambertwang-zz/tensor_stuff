/**
 * Node index Header
 * 
 */

#ifndef __NODE_H__
#define __NODE_H__

#include "tensorNode.h"

#include "constant.h"
#include "placeholder.h"
#include "variable.h"

#include "nodeMath/clamp.h"
#include "nodeMath/softmax.h"
#include "nodeMath/tensorlog.h"

#include "nodeOperations/add.h"
#include "nodeOperations/dotproduct.h"
#include "nodeOperations/matmult.h"
#include "nodeOperations/mult.h"
#include "nodeOperations/square.h"
#include "nodeOperations/subtract.h"


#include "nodeReduce/reduceSum.h"
#include "nodeReduce/reduceMean.h"
#include "nodeReduce/vectornorm.h"

#include "trainNode/train.h"

#endif
