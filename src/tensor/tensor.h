/**
 * Tensor Class Header
 * 
 */

#ifndef __TENSOR_H__
#define __TENSOR_H__

// Library
#include <iostream>
#include <vector>

class Tensor {
    private:
    /**
     * The values stored in the tensor
     */
    std::vector<double> data;

    /**
     * A tensor's rank is its number of dimensions.
     */
    unsigned int rank;

    /**
     * A tensor's shape is an array containing the size of each of it's dimensions.
     */
    std::vector<unsigned int> shape;

    unsigned int data_count;

    public:
    /**
     * Constructor for a Tensor
     */
    Tensor();
    /**
     * Constructor for a rank 0 Tensor.
     */
    Tensor(const double n_data);
    Tensor(std::vector<unsigned int> n_shape);
    Tensor(const std::initializer_list<unsigned int>& n_shape);
    Tensor(const std::vector<double> n_data, std::vector<unsigned int> n_shape);
    Tensor(const std::initializer_list<double>& n_data, const std::initializer_list<unsigned int>& n_shape);
    Tensor(const Tensor &t);

    /**
     * for a tensor
     */
    virtual ~Tensor();

    /**
     * Tensor shape comparator
     * Returns true if tensors are the same shape
     */
    bool compareShape(const Tensor other) const;

    unsigned int getRank() const;
    std::vector<unsigned int> getShape() const;

    double getData(unsigned int index) const;
    void setData(unsigned int index, double val);
    std::vector<double> getAllData() const;
    void setAllData(double val);

    unsigned int getDataCount() const;

    /**
     * Copies data from another tensor with an identical shape.
     */
    void copyFrom(const Tensor t);
    /**
     * Accessor operator overload
     */
    double& operator[](size_t i);

    /**
     * Mathematics operations
     * These operations should not be in-place
     */
    /**
     * Element-wise square operation (not in-place)
     */
    Tensor square() const;

    /**
     * Returns a tensor of with the values clamped between some min and maximum;
     */
    Tensor clamp(double min = 0.0, double max = 1.0) const;
    /**
     * Vector dot product of two rank 1 tensors with identical shapes
     */
    static Tensor dotProduct(const Tensor& lhs, const Tensor& rhs);
    /**
     * Vector cross product of two rank 1 tensors with identical shapes????? TODO: Figure out how to multiply higher order vectors
     */
    static Tensor crossProduct(const Tensor& lhs, const Tensor& rhs);
    /**
     * Matrix product of two rank 2 tensors with identical shapes
     */
    static Tensor product(const Tensor& lhs, const Tensor& rhs);
    Tensor reduceSum() const;
    /**
     * Element-wise operations
     * These are in-place operations
     */
    Tensor& operator+=(const Tensor& rhs);
    Tensor& operator-=(const Tensor& rhs);
    Tensor& operator*=(const Tensor& rhs);
    Tensor& operator/=(const Tensor& rhs);

    int streamTensorValues(std::ostream& out, unsigned int rank, unsigned int index) const;
};

/**
 * NOTE: THESE FUNCTIONS ARE NOT THREAD SAVE
 * TODO: Make these functions thread safe
 */
Tensor& operator+(const Tensor& lhs, const Tensor& rhs);
Tensor& operator-(const Tensor& lhs, const Tensor& rhs);
Tensor& operator*(const Tensor& lhs, const Tensor& rhs);
Tensor& operator/(const Tensor& lhs, const Tensor& rhs);

std::ostream& operator<<(std::ostream& out, const Tensor& data);

#endif