#include "testlib.h"

// Library
#include <algorithm>
#include <climits>
#include <iostream>
#include <fstream>

std::ostream& operator<<(std::ostream& out, const std::vector<double>& data) {
    if (data.size() > 0) {
        out << data[0];
    } for (unsigned int i = 1; i < data.size(); i++) {
           out << ", " << data[i];
    }

    return out;
}

std::ostream& operator<<(std::ostream& out, const std::vector<unsigned int>& data) {
    if (data.size() > 0) {
        out << data[0];
    } for (unsigned int i = 1; i < data.size(); i++) {
           out << ", " << data[i];
    }

    return out;
}

static unsigned int error_count = 0;

void expectLt(Tensor t, double n) {
    if (t.getData(0) >= n) {
        error_count++;
        std::cout << "\e[1m\e[31mERROR, MISMATCH!\e[0m" << std::endl;
        std::cout << "Expecting       \"\e[1m\e[34m" << t.getData(0) << "\e[0m\"" << std::endl;
        std::cout << "To be less than \"\e[1m\e[36m" << n << "\e[0m\"" << std::endl;
    } else {
        std::cout << "Output in acceptable range: \"\e[1m\e[36m" << t.getData(0) << "\e[0m\"" << std::endl;
    }
}

void expect(Tensor t, std::string e) {
    std::ostringstream sstream;
    sstream << t.getAllData();
    std::string tmp = sstream.str();
    if (tmp.compare(e)) {
        error_count++;
        std::cout << "\e[1m\e[31mERROR, DATA MISMATCH!\e[0m" << std::endl;
        std::cout << "Expecting \"\e[1m\e[34m" << t.getAllData() << "\e[0m\"" << std::endl;
        std::cout << "To Equal  \"\e[1m\e[36m" << e << "\e[0m\"" << std::endl;
    } else {
        std::cout << "Received expected output: \"\e[1m\e[36m" << e << "\e[0m\"" << std::endl;
    }
}

void expectShape(Tensor t, std::string e) {
    std::ostringstream sstream;
    sstream << t.getShape();
    std::string tmp = sstream.str();
    if (tmp.compare(e)) {
        error_count++;
        std::cout << "\e[1m\e[31mERROR, SHAPE MISMATCH!\e[0m" << std::endl;
        std::cout << "Expecting shape \"\e[1m\e[34m" << t.getShape() << "\e[0m\"" << std::endl;
        std::cout << "To Equal  \"\e[1m\e[36m" << e << "\e[0m\"" << std::endl;
    } else {
        std::cout << "Received expected shape: \"\e[1m\e[36m" << e << "\e[0m\"" << std::endl;
    }
}

void printTestResults() {
    if (error_count) {
        std::cout << "\e[31mTests resulted in " << error_count << " Error(s)!\e[0m" << std::endl;

    } else {
        std::cout << "\e[32m" << "All tests sucessful!" << "\e[0m" << std::endl;
    }
}

int readInt(std::ifstream *filestream) {
    unsigned char data[4];
    filestream->read((char *) data, 4);
    return (data[3]) +
          ((data[2]) << 8) +
          ((data[1]) << 16) +
          ((data[0]) << 24);
}

// Returns size of file
size_t beginMnistFile(std::ifstream *training_file, unsigned int *count) {
    if (!*training_file) {
        throw std::runtime_error("Cannot open training training file.\n");
    }
    size_t file_size;

    // Send head to end
    training_file->seekg(0, std::ios::end);
    file_size = training_file->tellg();
    training_file->seekg(0, std::ios::beg);
    // Skip magic number
    readInt(training_file);
    // Read number of items
    *count = (unsigned int) readInt(training_file);

    return file_size;
}

Tensor readMnistImages(unsigned int max_images, unsigned int *p_img_size) {
    size_t file_size;
    unsigned int count = 0;
    unsigned char *val;
    int img_rows = 0;
    int img_cols = 0;

    // Training image file
    std::ifstream tr_img_file("train-images-idx3-ubyte");
    file_size = beginMnistFile(&tr_img_file, &count);

    std::cout << "Data size:   " << file_size << std::endl;
    std::cout << "Image Count: " << count << std::endl;

    Tensor img_tensor;
    unsigned int img_size = UINT_MAX;
    // Assume all images are the same dimension
    for (size_t i = 0; i < count && i < max_images; i++) {
        img_rows = readInt(&tr_img_file);
        img_cols = readInt(&tr_img_file);
        if (img_size == UINT_MAX) { 
            img_size = img_rows * img_cols;
            std::cout << "Image Size: " << img_size << std::endl;
            img_tensor = Tensor({std::min(count, max_images), img_size});
            val = (unsigned char *) std::malloc(img_size * sizeof(unsigned int));
        }

        tr_img_file.seekg(16 + (8 + img_size) * i, std::ios::beg);
        tr_img_file.read((char *) val, img_size);
        for (unsigned int j = 0; j < img_size; j++) {
            img_tensor.setData(i * img_size + j, (double) val[j]);
        }
    }

    tr_img_file.close();
    *p_img_size = img_size;
    // std::cout << img_tensor << std::endl;
    return img_tensor;
}

Tensor readMnistLabels(unsigned int max_images) {
    size_t file_size;
    unsigned int count = 0;
    unsigned char val;

    // Training labels file
    std::ifstream tr_label_file("train-labels-idx1-ubyte");
    file_size = beginMnistFile(&tr_label_file, &count);

    std::cout << "Data size:   " << file_size << std::endl;
    std::cout << "Label Count: " << count << std::endl;

    Tensor label_tensor;
    label_tensor = Tensor({std::min(count, max_images), 10});
    // Assume all images are the same dimension
    for (size_t i = 0; i < count && i < max_images; i++) {
        tr_label_file.read((char *) &val, 1);
        label_tensor.setData(i * 10 + (int) val, 1);
    }

    tr_label_file.close();
    // std::cout << label_tensor << std::endl;
    return label_tensor;
}
