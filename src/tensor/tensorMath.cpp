/**
 * Math operations for the Tensor class
 */

#include "tensor.h"

#include "tensoralgebra.h"

#include <algorithm>
#include <cmath>
#include <limits>

Tensor Tensor::tensor_log() const {
    Tensor output = Tensor(*this);
    for (unsigned int i = 0; i < data_count; i++) {
        output.data[i] = log(data[i]);
    }
    return output;
}

Tensor Tensor::clamp(double min, double max) const {
    Tensor output = Tensor(*this);
    for (unsigned int i = 0; i < data_count; i++) {
        output.data[i] = std::min(max, std::max(output.data[i], min));
    }

    return output;
}

/**
 * Perform component-wise multiplcation on higher order derivations of tensors.
 */
Tensor Tensor::derivMult(const Tensor& lhs, const Tensor &rhs) {
#ifdef DEBUG
    std::cout << "Starting deriv mult" << std::endl;
    std::cout << "lhs : " << lhs << std::endl;
    std::cout << "rhs : " << rhs << std::endl;
#endif
    if (rhs.rank < lhs.rank) {
        throw std::invalid_argument("TensorMath::derivMult(): rhs rank must be at least lhs rank.");
    }

    if (lhs.rank == 1 && lhs.shape[0] == 1) {
        // Perform scalar mult
        Tensor result = Tensor(rhs);
        result *= lhs;
        return result;
    }

    for (unsigned int i = 0; i < lhs.rank; i++) {
        if (lhs.shape[i] != rhs.shape[i]) {
            // TODO: Define whatever this means
            throw std::invalid_argument("TensorMath::derivMult(): lhs shape must be higher-order subtensor of rhs.");
        }
    }
    unsigned int subtensor_count = 1;
    for (unsigned int i = lhs.rank; i < rhs.rank; i++) {
        subtensor_count *= rhs.shape[i];
    }

    Tensor val = Tensor(std::vector<unsigned int>(rhs.shape.begin() + lhs.rank, rhs.shape.end()));
    for (unsigned int i = 0; i < subtensor_count; i++) {
        for (unsigned int j = 0; j < lhs.data_count; j++) {
            val.data[i] += lhs.data[j] * rhs.data[j * subtensor_count + i];
        }
    }
#ifdef DEBUG
    std::cout << "Computed value : " << val << std::endl;
#endif
    return val; 
}

Tensor Tensor::reduceSum() const {
    if (data_count == 0) {
        return Tensor();
    }
    if (data_count == 1 || rank == 0) {
        return Tensor(*this);
    }
    Tensor val = Tensor(std::vector<unsigned int>(shape.begin(), shape.end() - 1));

    unsigned int vec_count = 1;
    for (unsigned int i = 0; i < rank - 1; i++) {
        vec_count *= shape[i];
    }

    for (unsigned int i = 0; i < vec_count; i++) {
        double sum = 0;
        for (unsigned int j = 0; j < shape[rank - 1]; j++) {
            sum += data[i * shape[rank - 1] + j];
        }
        val.data[i] = sum;
    }

    return val;
}

Tensor Tensor::reduceMean() const {
    Tensor val = reduceSum();

    for (unsigned int i = 0; i < val.data_count; i++) {
        val.data[i] /= shape[rank - 1];
    }

    return val;
}

Tensor Tensor::vectorNorm(unsigned int l) const {
    if (data_count == 0) {
        return Tensor();
    }
    if (data_count == 1 || rank == 0) {
        return Tensor(*this);
    }
    Tensor val = Tensor(std::vector<unsigned int>(shape.begin(), shape.end() - 1));

    unsigned int vec_count = 1;
    for (unsigned int i = 0; i < rank - 1; i++) {
        vec_count *= shape[i];
    }

    for (unsigned int i = 0; i < vec_count; i++) {
        double sum = 0;
        for (unsigned int j = 0; j < shape[rank - 1]; j++) {
            sum += std::pow(data[i * shape[rank - 1] + j], l);
        }
        val.data[i] = std::pow(sum, 1.0 / (double) l);
    }
    return val;
}

Tensor Tensor::softMax() const {
    Tensor val = Tensor(*this);

    if (data_count == 0) {
        return val;
    }
    if (data_count == 1) {
        val.setAllData(1);
        return val;
    }

    unsigned int vec_count = 1;
    for (unsigned int i = 0; i < rank - 1; i++) {
        vec_count *= shape[i];
    }

    double *exp_calcs = (double *) malloc(sizeof(double) * shape[rank -  1]);
    for (unsigned int i = 0; i < vec_count; i++) {
        double exp_min = std::numeric_limits<double>::infinity();
        double sum = 0;
        // Calculate and subtract a K value from our exponents to ensure that the calculation does not overflow.
        for (unsigned int j = 0; j < shape[rank - 1]; j++) {
            if (data[i * shape[rank - 1] + j] < exp_min) {
                exp_min = data[i * shape[rank - 1] + j];
            }
        }
        for (unsigned int j = 0; j < shape[rank - 1]; j++) {
            exp_calcs[j] = exp(data[i * shape[rank - 1] + j] - exp_min);
            sum += exp_calcs[j];
        }
        for (unsigned int j = 0; j < shape[rank - 1]; j++) {
            val.data[i * shape[rank - 1] + j] = exp_calcs[j] / sum;
        }
    }
    free(exp_calcs);

    return val;
}
