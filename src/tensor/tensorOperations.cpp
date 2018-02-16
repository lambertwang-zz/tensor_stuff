/**
 * Simple operations for the Tensor class
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

Tensor Tensor::dotProduct(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.compareShape(rhs)) {
        throw std::invalid_argument("TensorMath::dotProduct(): Tensors must be same shape");
    }
    Tensor output = Tensor(lhs);
    for (unsigned int i = 0; i < lhs.data_count; i++) {
        output.data[i] *= rhs.data[i];
    }
    return output;
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
                for (unsigned int k = 0; k < lhs_x; k++) {
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
