/**
 * Math operations for the Tensor class
 */

#include "tensor.h"

#include "tensoralgebra.h"

#include <algorithm>
#include <cmath>

Tensor Tensor::square() const {
    Tensor output = Tensor(*this);
    for (unsigned int i = 0; i < data_count; i++) {
        output.data[i] *= output.data[i];
    }
    return output;
}

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

Tensor Tensor::dotProduct(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.compareShape(rhs)) {
        std::cout << lhs << std::endl;
        std::cout << rhs << std::endl;
        throw std::invalid_argument("TensorMath::dotProduct(): Tensors must be same shape");
    }
    Tensor output = Tensor(lhs);
    for (unsigned int i = 0; i < lhs.data_count; i++) {
        output.data[i] *= rhs.data[i];
    }
    return output;
}

Tensor Tensor::crossProduct(const Tensor& lhs, const Tensor& rhs) {
    (void)lhs; (void)rhs;
    return Tensor();
}

Tensor Tensor::product(const Tensor& lhs, const Tensor& rhs) {
    (void)lhs; (void)rhs;
    return Tensor();
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
        std::cout << "lhs : " << lhs << std::endl;
        std::cout << "rhs : " << rhs << std::endl;
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
            std::cout << "lhs : " << lhs << std::endl;
            std::cout << "rhs : " << rhs << std::endl;
            throw std::invalid_argument("TensorMath::derivMult(): lhs shape must be higher-order subtensor of rhs.");
        }
    }
    unsigned int subtensor_count = 1;
    for (unsigned int i = lhs.rank; i < rhs.rank; i++) {
        subtensor_count *= rhs.shape[i];
    }

    Tensor val = Tensor(std::vector<unsigned int>(rhs.shape.begin() + lhs.rank, rhs.shape.end()));
    // std::cout << "lhs data_count : " << lhs.data_count << std::endl;
    // std::cout << "rhs data_count : " << rhs.data_count << std::endl;
    // std::cout << "val data_count : " << val.data_count << std::endl;
    for (unsigned int i = 0; i < subtensor_count; i++) {
        // std::cout << "val index : " << i << std::endl;
        for (unsigned int j = 0; j < lhs.data_count; j++) {
            // std::cout << "lhs index : " << j << std::endl;
            // std::cout << "rhs index : " << (j * subtensor_count + i) << std::endl;

            val.data[i] += lhs.data[j] * rhs.data[j * subtensor_count + i];
        }
    }
#ifdef DEBUG
    std::cout << "Computed value : " << val << std::endl;
#endif
    return val; 
}

Tensor Tensor::reduceSum() const {
    Tensor val = Tensor(std::vector<unsigned int>(shape.begin(), shape.end() - 1));
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

    for (unsigned int i = 0; i < vec_count; i++) {
        double sum = 0;
        for (unsigned int j = 0; j < shape[rank - 1]; j++) {
            sum += exp(data[i * shape[rank - 1] + j]);
        }
        for (unsigned int j = 0; j < shape[rank - 1]; j++) {
            val.data[i * shape[rank - 1] + j] = exp(data[i * shape[rank - 1] + j]) / sum;
        }
    }

    return val;
}


/**
 * Tensor addition
 * 
 * 2 + 2 = 4
 * [2, 3, 4] + 2 = [4, 5, 6]
 * [2, 3, 4] + [1, 2] = error
 * [1, 2] + [[10, 20], [30, 40]] = [[11, 21], [32, 42]]
 */
Tensor& Tensor::operator+=(const Tensor& rhs) {
    if (rhs.rank > rank) {
        throw std::invalid_argument("Tensor::operator+=(): rhs has a greater rank than lhs.");
    }
    /**
     * Check if the shape of rhs is a subtensor of lhs
     * ie: lhs.shape = [3, 4, 7, 2], rhs = [7, 2]
     */
    for (unsigned int i = 0; i < rhs.rank; i++) {
        if (rhs.shape[rhs.rank - i - 1] != shape[rank - i - 1]) {
            throw std::invalid_argument("Tensor::operator+=(): rhs is not a subtensor of lhs.");
        }
    }

    unsigned int i = 0;
    while (i < data_count) {
        data[i] += rhs.data[i % rhs.data_count];
        i++;
    }

    return *this;
}

/**
 * Partial piecewise mathematics operations.
 */
Tensor& Tensor::operator-=(const Tensor& rhs) {
    if (rhs.data_count != 1 && !compareShape(rhs)) {
        std::cout << *this << std::endl;
        std::cout << rhs << std::endl;
        throw std::invalid_argument("Tensor::operator-=(): Operands have different shapes.");
    }

    // If rhs is not a scalar, lhs cannot be a scalar
    // If rhs is scalar
    if (rhs.data_count == 1) {
        for (unsigned int i = 0; i < data_count; i++) {
            data[i] -= rhs.data[0];
        }
    } else {
        for (unsigned int i = 0; i < data_count; i++) {
            data[i] -= rhs.data[i];
        }
    }
    return *this;
}

/**
 * Tensor multiplication
 * 
 * Scalar multiplication
 * 2 * [[1, 2], [3, 4]] = [[2, 4], [6, 8]]
 * 
 * Element wise multiplication
 * [[2, 4], [6, 8]] * [[1, 3], [5, 7]] = [[2, 12], [30, 56]]
 * 
 * [2] * [[1, 2, 3]] = [[2, 4, 6]] (not supported)
 * [2] * [[1], [2]] = error (not supported)
 * 
 * TODO: Improve performance
 * TODO: Subtensor multiplication
 */
Tensor& Tensor::operator*=(const Tensor& rhs) {
    // Check if lhs and rhs are scalars, do simple Scalar multiplication
    if (rank == 0 &&
        data_count == 1 &&
        compareShape(rhs)) {
        data[0] *= rhs.data[0];
        return *this;
    }

    // Check if sizes are correct for multiplication
    unsigned int lhs_x = rank == 2 ? shape[1] : 1,
        lhs_y = rank >= 1 ? shape[0] : 1,
        rhs_x = rhs.rank == 2 ? rhs.shape[1] : 1,
        rhs_y = rhs.rank >= 1 ? rhs.shape[0] : 1;

    // std::cout << "lhs_x : " << lhs_x << std::endl;
    // std::cout << "lhs_y : " << lhs_y << std::endl;
    // std::cout << "rhs_x : " << rhs_x << std::endl;
    // std::cout << "rhs_y : " << rhs_y << std::endl;

    // Shape is correct, do matrix multiplication
    if (lhs_x == rhs_y) {
        Tensor result;
        // Setup the shape of the result tensor
        if (rhs.rank < 2) {
            if (lhs_y == 1) {
                result = Tensor(0);
            } else {
                result = Tensor({lhs_y});
            }
        } else {
            result = Tensor({lhs_y, rhs_x});
        }

        for (unsigned int j = 0; j < lhs_y; j++) {
            for (unsigned int i = 0; i < rhs_x; i++) {
                // std::cout << "Result Index: x = " << i << ", y = " << j << std::endl;
                for (unsigned int k = 0; k < lhs_x; k++) {
                    // std::cout << "Multiplying " << data[k + i * lhs_x] << " and " << rhs.data[j + k * rhs_x] << std::endl;
                    result.data[i + j * rhs_x] += 
                        data[k + j * lhs_x] *
                        rhs.data[i + k * rhs_x];
                }
            }
        }
        copyFrom(result);
        return *this;
    }

    // Otherwise, do element-wise multiplication
    if (rhs.data_count != 1 && !compareShape(rhs)) {
        std::cout << *this << std::endl;
        std::cout << rhs << std::endl;
        throw std::invalid_argument("Tensor::operator*=(): Operands have different shapes.");
    }

    if (rhs.data_count == 1) {
        for (unsigned int i = 0; i < data_count; i++) {
            data[i] *= rhs.data[0];
        }
        return *this;
    }

    for (unsigned int i = 0; i < data_count; i++) {
        data[i] *= rhs.data[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& rhs) {
    // If rhs is scalar
    if (rhs.rank == 0) {
        if (rhs.data_count == 1) {
            for (unsigned int i = 0; i < data_count; i++) {
                data[i] /= rhs.data[0];
            }
            return *this;
        }
    }

    if (!compareShape(rhs)) {
        std::cout << *this << std::endl;
        std::cout << rhs << std::endl;
        throw std::invalid_argument("Tensor::operator/=(): Operands have different shapes.");
    }

    for (unsigned int i = 0; i < data_count; i++) {
        data[i] /= rhs.data[i];
    }
    return *this;
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    Tensor output(lhs);
    output += rhs;
    return Tensor(output);
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    Tensor output(lhs);
    output -= rhs;
    return Tensor(output);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    Tensor output(lhs);
    output *= rhs;
    return Tensor(output);
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    Tensor output(lhs);
    output /= rhs;
    return Tensor(output);
}
