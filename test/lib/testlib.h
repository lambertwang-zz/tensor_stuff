#ifndef __TESTLIB_H__
#define __TESTLIB_H__

#include "node/node.h"

// Library
#include <sstream>

std::ostream& operator<<(std::ostream& out, const std::vector<double>& data);
void expectLt(Tensor t, double n);
void expect(Tensor t, std::string e);
void expectShape(Tensor t, std::string e);
void printTestResults();

Tensor readMnistImages(unsigned int max_images, unsigned int *p_img_size);
Tensor readMnistLabels(unsigned int max_images);

#endif
