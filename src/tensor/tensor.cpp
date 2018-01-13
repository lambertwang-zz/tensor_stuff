/**
 * Tensor class
 */

#include "tensor.h"

// Libraries
#include <stdlib.h>
#include <string.h>

Tensor::Tensor() {
    rank = 0;
    data_count = 0;
    shape = std::vector<int>(rank);
    data = std::vector<float>(data_count);
}

Tensor::Tensor(const float n_data) {
    rank = 0;
    data_count = 1;
    shape = std::vector<int>(rank);
    data = std::vector<float>(data_count, n_data);
}

Tensor::Tensor(const std::vector<int> n_shape) {
    data_count = 1;
    rank = n_shape.size();
    for (unsigned int i = 0; i < rank; i++) {
        data_count *= n_shape[i];
    }

    shape = n_shape;
    data = std::vector<float>(data_count, 0.0);
}

Tensor::Tensor(const std::vector<float> n_data, const std::vector<int> n_shape) {
    data_count = 1;
    rank = n_shape.size();
    for (unsigned int i = 0; i < rank; i++) {
        data_count *= n_shape[i];
    }

    shape = n_shape;
    data = n_data;
}

Tensor::Tensor(const std::initializer_list<float>& n_data, const std::initializer_list<int>& n_shape) {
    rank = n_shape.size();
    shape = std::vector<int>(rank);

    data_count = 1;
    int index = 0;
    std::initializer_list<int>::iterator it;
    for (it = n_shape.begin(); it != n_shape.end(); it++) {
        data_count *= *it;
        shape[index++] = *it;
    }

    data = std::vector<float>(data_count);

    unsigned int i = 0;
    for (float d: n_data) {
        data[i++] = d;
    }
}

Tensor::Tensor(const Tensor &t) {
    rank = t.rank;
    data_count = t.data_count;

    shape = t.shape;
    data = t.data;
}

Tensor::~Tensor() {
}

bool Tensor::compareShape(const Tensor other) const {
    // If different ranks, shapes cannot be the same
    if (rank != other.rank) {
        return false;
    }
    // If different data counts, shapes cannot be the same
    if (data_count != other.data_count) {
        return false;
    }
    for (unsigned int i = 0; i < rank; i++) {
        if (shape[i] != other.shape[i]) {
            return false;
        }
    }

    return true;
}

unsigned int Tensor::getRank() const {
    return rank;
}

std::vector<int> Tensor::getShape() const {
    return shape;
}

float Tensor::getData(unsigned int index) const {
    if (index >= data_count) {
        return 0;
    }
    return data[index];
}

std::vector<float> Tensor::getAllData() const {
    return data;
}

void Tensor::setAllData(float val) {
    for (unsigned int i = 0; i < data_count; i++) {
        data[i] = val;
    }
}

float& Tensor::operator[](size_t i) {
    return data[i];
}

void Tensor::copyFrom(const Tensor t) {
    rank = t.rank;
    data_count = t.data_count;

    shape = t.shape;
    data = t.data;
}

int Tensor::streamTensorValues(std::ostream& out, unsigned int rank, unsigned int index) const {
    if (rank == this->rank) {
        out << data[index];
        return 0;
    }
    unsigned int rank_count = 1;
    for (unsigned int i = rank + 1; i < this->rank; i++) {
        rank_count *= shape[i];
    }
    out << "[";
    streamTensorValues(out, rank + 1, index);
    for (int i = 1; i < shape[rank]; i++) {
        out << ", ";
        streamTensorValues(out, rank + 1, index + i * rank_count);
    }
    out << "]";
    return 0;
}

std::ostream& operator<<(std::ostream& out, const Tensor& data) {
    out << "Tensor ";
    out << "rank: " << data.getRank() << " ";

    if (data.getRank() > 0) {
        out << "shape: [" << data.getShape()[0];
        for (unsigned int i = 1; i < data.getRank(); i++) {
            out << ", " << data.getShape()[i];
        }
        out << "] ";
    }

    out << "values: ";
    data.streamTensorValues(out, 0, 0);

    return out;
}
