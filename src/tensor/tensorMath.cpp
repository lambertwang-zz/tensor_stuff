/**
 * Math operations for the Tensor class
 */

#include "tensor.h"

#include <algorithm>

Tensor Tensor::square() const {
    Tensor output = Tensor(*this);
    for (unsigned int i = 0; i < data_count; i++) {
        output.data[i] *= output.data[i];
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
    if (lhs.rank != 1 || !lhs.compareShape(rhs)) {
        throw std::invalid_argument("TensorMath::dotProduct(): Tensors must be rank 1 and same shape");
    }
    Tensor output = Tensor(lhs);
    output *= rhs;
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

Tensor Tensor::reduceSum() const {
    Tensor val = Tensor(std::vector<unsigned int>(shape.begin() + 1, shape.end()));
    for (double d: this->data) {
        val += d;
    }
    return Tensor(val);
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
    if (rhs.data_count != 1 && !compareShape(rhs)) {
        throw std::invalid_argument("Tensor::operator+=(): Operands have different shapes.");
    }

    // Scalar addition
    // If lhs is scalar
    if (rank == 0) {
        if (data_count == 1) {
            for (unsigned int i = 0; i < rhs.data_count; i++) {
                data[0] += rhs.data[i];
            }
            return *this;
        }
    }

    // If rhs is scalar
    if (rhs.data_count == 1) {
        for (unsigned int i = 0; i < data_count; i++) {
            data[i] += rhs.data[0];
        }
    } else {
        for (unsigned int i = 0; i < data_count; i++) {
            data[i] += rhs.data[i];
        }
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
    unsigned int lhs_x = rank >= 1 ? shape[0] : 1,
        lhs_y = rank == 2 ? shape[1] : 1,
        rhs_x = rhs.rank >= 1 ? rhs.shape[0] : 1,
        rhs_y = rhs.rank == 2 ? rhs.shape[1] : 1;

    // Shape is correct, do matrix multiplication
    if (lhs_x == rhs_y) {
        Tensor result;
        if (lhs_y == 1) {
            if (rhs_x == 1) {
                result = Tensor(0);
            } else {
                result = Tensor({rhs_x});
            }
        } else {
            result = Tensor({rhs_x, lhs_y});
        }
        for (unsigned int i = 0; i < rhs_x; i++) {
            for (unsigned int j = 0; j < lhs_y; j++) {
                result.data[j + i * lhs_y] = 0.0;
                for (unsigned int k = 0; k < rhs_y; k++) {
                    result.data[j + i * lhs_y] += 
                        data[j + k * lhs_y] *
                        rhs.data[k + i * rhs_y];
                }
            }
        }
        copyFrom(result);
        return *this;
    }

    // Otherwise, do element-wise multiplication
    if (!compareShape(rhs)) {
        throw std::invalid_argument("Tensor::operator*=(): Operands have different shapes.");
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

Tensor& operator+(const Tensor& lhs, const Tensor& rhs) {
    static Tensor output;
    output = Tensor(lhs);
    output += rhs;
    return output;
}

Tensor& operator-(const Tensor& lhs, const Tensor& rhs) {
    static Tensor output;
    output = Tensor(lhs);
    output -= rhs;
    return output;
}

Tensor& operator*(const Tensor& lhs, const Tensor& rhs) {
    static Tensor output;
    output = Tensor(lhs);
    output *= rhs;
    return output;
}

Tensor& operator/(const Tensor& lhs, const Tensor& rhs) {
    static Tensor output;
    output = Tensor(lhs);
    output /= rhs;
    return output;
}
