#include <cmath>
#include <numbers>
#include <random>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional> 
#include <vector>
#include <set>
#include <span>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <limits>
#include <string>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <iostream>

namespace py = pybind11;

class Storage;
class TensorImp;
class Tensor;
class DataLoaderCSV;
class DataLoaderIterator;
class Linear;

namespace nn {
    Tensor linear3d(const Tensor&, Linear&);
}

[[maybe_unused]] constexpr float inf = std::numeric_limits<float>::max();
[[maybe_unused]] constexpr float ninf = std::numeric_limits<float>::lowest();

enum class Operation {
    NONE,
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    MATMUL,
    CONV2D,
    MAXPOOL2D,
    RELU,
    LEAKYRELU,
    GELU,
    SIGMOID,
    SUM,
    MEAN,
    VARIANCE,
    TRANSPOSE,
    CONCAT,
    SOFTMAX,
    SOFTMAX_CE,
    SPARSE_SCE,
    RESHAPE,
    EMBED,
    SLICE
};

struct Conv2DMetadata {
    std::vector<float> col;
    size_t padding;
    size_t kernel_size;
    size_t stride;
};

struct Slice {
    size_t start;
    size_t stop;
    size_t step = 1;
    
public:
    Slice(const size_t start, const size_t stop, const size_t step = 1) : start(start), stop(stop), step(step) {}

    Slice(const std::vector<size_t>& bounds) {
        if (bounds.size() != 3) {
            throw std::invalid_argument("Invalid slice");
        }
        start = bounds[0];
        stop = bounds[1];
        step = bounds[2];
    }
};

class Storage {
public:
    std::vector<float> data;

    Storage(const size_t size = 0, const float init_value = 0) : data(size, init_value) {}
};

class TensorImp {
public:
    std::shared_ptr<Storage> storage = std::make_shared<Storage>();
    std::shared_ptr<Storage> grad_storage = nullptr;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t offset = 0;
    size_t length = 0;
    bool requires_grad = false;
    Operation op_type = Operation::NONE;
    std::vector<std::shared_ptr<TensorImp>> parents;
    int op_axis = -1;
    float alpha = 0.01;
    int tra, trb;
    std::shared_ptr<Conv2DMetadata> convmd = std::make_shared<Conv2DMetadata>();
    std::shared_ptr<std::vector<size_t>> buffer = std::make_shared<std::vector<size_t>>();
    std::shared_ptr<std::vector<size_t>> buffer2 = std::make_shared<std::vector<size_t>>();
    std::shared_ptr<std::vector<float>> bufferf = std::make_shared<std::vector<float>>();

    TensorImp() = default;

    bool is_contiguous() const {
        size_t expected = 1;
        for (int i = static_cast<int>(this->shape.size()) - 1; i >= 0; --i) {
            if (strides[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }

    TensorImp transpose(const int a, const int b) const {
        if (a < 0 || b < 0 || a >= this->shape.size() || b >= this->shape.size() || a == b) {
            throw std::out_of_range("Invalid transpose dimensions");
        }

        TensorImp out = *this;
        std::swap(out.shape[a], out.shape[b]);
        std::swap(out.strides[a], out.strides[b]);
        if (out.requires_grad) {
            out.op_type = Operation::NONE;
            out.parents.clear();
        }

        return out;
    }

    std::vector<float> max(const int axis = -1) const {
        if (axis < -1 || axis >= this->shape.size()) {
            throw std::invalid_argument("Invalid axis for max()");
        }

        if (axis == -1 && this->is_contiguous()) {
            float xm = ninf;
            for (size_t i = 0; i < this->length; ++i) {
                if (this->storage->data[this->offset + i] > xm) xm = this->storage->data[this->offset + i];
            }

            return std::vector<float>{xm};
        }
        else if (axis == -1) {
            float xm = ninf;
            std::vector<size_t> coord(this->shape.size(), 0);
            size_t ti = this->offset;
            for (size_t n = 0; n < this->length; ++n) {
                if (this->storage->data[ti] > xm) xm = this->storage->data[ti];

                for (int d = this->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->strides[d];

                    if (coord[d] < this->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->shape[d] * this->strides[d];
                }
            }

            return std::vector<float>{xm};
        }
        else {
            std::vector<float> outdata(this->length / this->shape[axis]);
            std::vector<size_t> coord(this->shape.size(), 0);
            int x = 0;

            while (true) {
                size_t index = this->offset;
                float xm = ninf;

                for (int d = 0; d < this->shape.size(); ++d) {
                    index += coord[d] * this->strides[d];
                }

                for (size_t i = 0; i < this->shape[axis]; ++i) {
                    if (this->storage->data[index] > xm) {
                        xm = this->storage->data[index];
                    }
                    index += this->strides[axis];
                }

                outdata[x] = xm;

                bool completed = true;

                for (int d = this->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < this->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;

                ++x;
            }

            return outdata;
        }
    }
};

class Tensor {
private:
    std::shared_ptr<Storage> make_storage(size_t n, float init_value) const {
        auto s = std::make_shared<Storage>(n, init_value);
        return s;
    }

    std::vector<float> max_data(const int axis = -1) const {
        if (axis < -1 || axis >= this->imp->shape.size()) {
            throw std::invalid_argument("Invalid axis for max()");
        }

        if (axis == -1 && this->is_contiguous()) {
            float xm = ninf;
            for (size_t i = 0; i < this->imp->length; ++i) {
                if (this->imp->storage->data[this->imp->offset + i] > xm) xm = this->imp->storage->data[this->imp->offset + i];
            }
            
            return std::vector<float>{xm};
        }
        else if (axis == -1) {
            float xm = ninf;
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            size_t ti = this->imp->offset;
            for (size_t n = 0; n < this->imp->length; ++n) {
                if (this->imp->storage->data[ti] > xm) xm = this->imp->storage->data[ti];

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->imp->strides[d];

                    if (coord[d] < this->imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->imp->shape[d] * this->imp->strides[d];
                }
            }

            return std::vector<float>{xm};
        }
        else {
            std::vector<float> outdata(this->imp->length / this->imp->shape[axis]);
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            int x = 0;

            while (true) {
                size_t index = this->imp->offset;
                float xm = ninf;

                for (int d = 0; d < this->imp->shape.size(); ++d) {
                    index += coord[d] * this->imp->strides[d];
                }

                for (size_t i = 0; i < this->imp->shape[axis]; ++i) {
                    if (this->imp->storage->data[index] > xm) {
                        xm = this->imp->storage->data[index];
                    }
                    index += this->imp->strides[axis];
                }

                outdata[x] = xm;

                bool completed = true;

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < this->imp->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;

                ++x;
            }

            return outdata;
        }
    }

    Tensor add_kernel_128x256_256_2d_contiguous(const Tensor& other) const {
        Tensor out(std::vector<size_t>{128, 256}, this->imp->requires_grad || other.imp->requires_grad);

        size_t ai = this->imp->offset;
        size_t bi = other.imp->offset;
        size_t oi = out.imp->offset;

        for (size_t n = 0; n < out.imp->length; ++n) {
            out.imp->storage->data[oi] = this->imp->storage->data[ai] + other.imp->storage->data[bi / (size_t)256];
            ++ai;
            ++bi;
            ++oi;
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::ADD;
            out.imp->parents = {this->imp, other.imp};
        }

        return out;
    }

    std::vector<float> im2col(const float* a, size_t c, size_t h, size_t w, size_t k, size_t stride, size_t padding) const {
        const size_t ho = (h + 2 * padding - k) / stride + 1;
        const size_t wo = (w + 2 * padding - k) / stride + 1;
        const size_t l = ho * wo;

        std::vector<float> out(c * k * k * l, 0.0f);

        for (size_t p = 0; p < c; ++p) {
            for (size_t kh = 0; kh < k; ++kh) {
                for (size_t kw = 0; kw < k; ++kw) {
                    const size_t q = (p * k + kh) * k + kw;

                    for (size_t x = 0; x < ho; ++x) {
                        const int ih = static_cast<int>(x * stride + kh) - static_cast<int>(padding);
                        for (size_t y = 0; y < wo; ++y) {
                            const int iw = static_cast<int>(y * stride + kw) - static_cast<int>(padding);

                            if (ih >= 0 && iw >= 0 && ih < static_cast<int>(h) && iw < static_cast<int>(w)) {
                                out[q * l + x * wo + y] = a[(p * h + ih) * w + iw];
                            }
                        }
                    }
                }
            }
        }

        return out;
    }

    void im2col_direct(const float* a, float* b, size_t c, size_t h, size_t w, size_t k, size_t stride, size_t padding) const {
        const size_t ho = (h + 2 * padding - k) / stride + 1;
        const size_t wo = (w + 2 * padding - k) / stride + 1;
        const size_t l = ho * wo;

        for (size_t p = 0; p < c; ++p) {
            for (size_t kh = 0; kh < k; ++kh) {
                for (size_t kw = 0; kw < k; ++kw) {
                    const size_t q = (p * k + kh) * k + kw;

                    for (size_t x = 0; x < ho; ++x) {
                        const int ih = static_cast<int>(x * stride + kh) - static_cast<int>(padding);
                        for (size_t y = 0; y < wo; ++y) {
                            const int iw = static_cast<int>(y * stride + kw) - static_cast<int>(padding);

                            if (ih >= 0 && iw >= 0 && ih < static_cast<int>(h) && iw < static_cast<int>(w)) {
                                b[q * l + x * wo + y] = a[(p * h + ih) * w + iw];
                            }
                        }
                    }
                }
            }
        }
    }

    void col2im(const float* a, float* b, size_t c, size_t h, size_t w, size_t k, size_t stride, size_t padding) const {
        const size_t ho = (h + 2 * padding - k) / stride + 1;
        const size_t wo = (w + 2 * padding - k) / stride + 1;
        const size_t l = ho * wo;

        for (size_t p = 0; p < c; ++p) {
            for (size_t kh = 0; kh < k; ++kh) {
                for (size_t kw = 0; kw < k; ++kw) {
                    const size_t q = (p * k + kh) * k + kw;

                    for (size_t x = 0; x < ho; ++x) {
                        const int ih = static_cast<int>(x * stride + kh) - static_cast<int>(padding);
                        for (size_t y = 0; y < wo; ++y) {
                            const int iw = static_cast<int>(y * stride + kw) - static_cast<int>(padding);

                            if (ih >= 0 && iw >= 0 && ih < static_cast<int>(h) && iw < static_cast<int>(w)) {
                                b[(p * h + ih) * w + iw] += a[q * l + x * wo + y];
                            }
                        }
                    }
                }
            }
        }
    }

    //Multiplication Kernels

    void matmul_kernel_add_offset(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset, const size_t a_rows, const size_t a_cols, const size_t b_rows, const size_t b_cols) const {
        if (a_cols != b_rows) {
            throw std::invalid_argument("Invalid dimensions for multiplication");
        }
        constexpr int BS = 64;
        for (size_t ii = 0; ii < a_rows; ii += BS) {
            size_t i_end = std::min(ii + BS, a_rows);
            for (size_t kk = 0; kk < a_cols; kk += BS) {
                size_t k_end = std::min(kk + BS, a_cols);
                for (size_t jj = 0; jj < b_cols; jj += BS) {
                    size_t j_end = std::min(jj + BS, b_cols);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + i * a_cols + k];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * b_cols + j] += aik * b_data[b_offset + k * b_cols + j];
                                out[out_offset + i * b_cols + j + 1] += aik * b_data[b_offset + k * b_cols + j + 1];
                                out[out_offset + i * b_cols + j + 2] += aik * b_data[b_offset + k * b_cols + j + 2];
                                out[out_offset + i * b_cols + j + 3] += aik * b_data[b_offset + k * b_cols + j + 3];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * b_cols + j] += aik * b_data[b_offset + k * b_cols + j];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset, const size_t a_rows, const size_t a_cols, const size_t b_rows, const size_t b_cols, const size_t a_row_stride, const size_t a_col_stride, const size_t b_row_stride, const size_t b_col_stride, const size_t out_row_stride, const size_t out_col_stride) const {
        if (a_cols != b_rows) {
            throw std::invalid_argument("Invalid dimensions for multiplication");
        }
        constexpr int BS = 64;
        #pragma omp parallel for schedule(static)
        for (size_t ii = 0; ii < a_rows; ii += BS) {
            size_t i_end = std::min(ii + BS, a_rows);
            for (size_t kk = 0; kk < a_cols; kk += BS) {
                size_t k_end = std::min(kk + BS, a_cols);
                for (size_t jj = 0; jj < b_cols; jj += BS) {
                    size_t j_end = std::min(jj + BS, b_cols);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + i * a_row_stride + k * a_col_stride];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * out_row_stride + (j) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j) * b_col_stride];
                                out[out_offset + i * out_row_stride + (j + 1) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j + 1) * b_col_stride];
                                out[out_offset + i * out_row_stride + (j + 2) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j + 2) * b_col_stride];
                                out[out_offset + i * out_row_stride + (j + 3) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j + 3) * b_col_stride];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * out_row_stride + (j) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j) * b_col_stride];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_atranspose(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset, const size_t a_rows, const size_t a_cols, const size_t b_rows, const size_t b_cols, const size_t a_row_stride, const size_t a_col_stride, const size_t b_row_stride, const size_t b_col_stride, const size_t out_row_stride, const size_t out_col_stride) const {
        if (a_rows != b_rows) {
            throw std::invalid_argument("Invalid dimensions for multiplication");
        }
        constexpr int BS = 64;
        #pragma omp parallel for schedule(static)
        for (size_t ii = 0; ii < a_cols; ii += BS) {
            size_t i_end = std::min(ii + BS, a_cols);
            for (size_t kk = 0; kk < a_rows; kk += BS) {
                size_t k_end = std::min(kk + BS, a_rows);
                for (size_t jj = 0; jj < b_cols; jj += BS) {
                    size_t j_end = std::min(jj + BS, b_cols);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + k * a_row_stride + i * a_col_stride];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * out_row_stride + (j) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j) * b_col_stride];
                                out[out_offset + i * out_row_stride + (j + 1) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j + 1) * b_col_stride];
                                out[out_offset + i * out_row_stride + (j + 2) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j + 2) * b_col_stride];
                                out[out_offset + i * out_row_stride + (j + 3) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j + 3) * b_col_stride];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * out_row_stride + (j) * out_col_stride] += aik * b_data[b_offset + k * b_row_stride + (j) * b_col_stride];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_atranspose(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset, const size_t a_rows, const size_t a_cols, const size_t b_rows, const size_t b_cols) const {
        if (a_rows != b_rows) {
            throw std::invalid_argument("Invalid dimensions for multiplication");
        }
        constexpr int BS = 64;
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t ii = 0; ii < a_cols; ii += BS) {
            size_t i_end = std::min(ii + BS, a_cols);
            for (size_t kk = 0; kk < a_rows; kk += BS) {
                size_t k_end = std::min(kk + BS, a_rows);
                for (size_t jj = 0; jj < b_cols; jj += BS) {
                    size_t j_end = std::min(jj + BS, b_cols);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + k * a_cols + i];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * b_cols + (j)] += aik * b_data[b_offset + k * b_cols + (j)];
                                out[out_offset + i * b_cols + (j + 1)] += aik * b_data[b_offset + k * b_cols + (j + 1)];
                                out[out_offset + i * b_cols + (j + 2)] += aik * b_data[b_offset + k * b_cols + (j + 2)];
                                out[out_offset + i * b_cols + (j + 3)] += aik * b_data[b_offset + k * b_cols + (j + 3)];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * b_cols + (j)] += aik * b_data[b_offset + k * b_cols + (j)];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_btranspose(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset, const size_t a_rows, const size_t a_cols, const size_t b_rows, const size_t b_cols, const size_t a_row_stride, const size_t a_col_stride, const size_t b_row_stride, const size_t b_col_stride, const size_t out_row_stride, const size_t out_col_stride) const {
        if (a_cols != b_cols) {
            throw std::invalid_argument("Invalid dimensions for multiplication");
        }
        constexpr int BS = 64;
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t ii = 0; ii < a_rows; ii += BS) {
            size_t i_end = std::min(ii + BS, a_rows);
            for (size_t jj = 0; jj < b_rows; jj += BS) {
                size_t j_end = std::min(jj + BS, b_rows);
                for (size_t kk = 0; kk < a_cols; kk += BS) {
                    size_t k_end = std::min(kk + BS, a_cols);
                    for (int i = ii; i < i_end; ++i) {
                        for (int j = jj; j < j_end; ++j) {
                            float oij = 0.0f;
                            int k = kk;
                            for (; k + 3 < k_end; k += 4) {
                                oij += a_data[a_offset + i * a_row_stride + (k) * a_col_stride] * b_data[b_offset + j * b_row_stride + (k) * b_col_stride];
                                oij += a_data[a_offset + i * a_row_stride + (k + 1) * a_col_stride] * b_data[b_offset + j * b_row_stride + (k + 1) * b_col_stride];
                                oij += a_data[a_offset + i * a_row_stride + (k + 2) * a_col_stride] * b_data[b_offset + j * b_row_stride + (k + 2) * b_col_stride];
                                oij += a_data[a_offset + i * a_row_stride + (k + 3) * a_col_stride] * b_data[b_offset + j * b_row_stride + (k + 3) * b_col_stride];
                            }
                            for (; k < k_end; ++k) {
                                oij += a_data[a_offset + i * a_row_stride + (k) * a_col_stride] * b_data[b_offset + j * b_row_stride + (k) * b_col_stride];
                            }
                            out[out_offset + i * out_row_stride + j * out_col_stride] += oij;
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_btranspose(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset, const size_t a_rows, const size_t a_cols, const size_t b_rows, const size_t b_cols) const {
        if (a_cols != b_cols) {
            throw std::invalid_argument("Invalid dimensions for multiplication");
        }
        constexpr int BS = 64;
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t ii = 0; ii < a_rows; ii += BS) {
            size_t i_end = std::min(ii + BS, a_rows);
            for (size_t jj = 0; jj < b_rows; jj += BS) {
                size_t j_end = std::min(jj + BS, b_rows);
                for (size_t kk = 0; kk < a_cols; kk += BS) {
                    size_t k_end = std::min(kk + BS, a_cols);
                    for (int i = ii; i < i_end; ++i) {
                        for (int j = jj; j < j_end; ++j) {
                            float oij = 0.0f;
                            int k = kk;
                            for (; k + 3 < k_end; k += 4) {
                                oij += a_data[a_offset + i * a_cols + (k)] * b_data[b_offset + j * b_cols + (k)];
                                oij += a_data[a_offset + i * a_cols + (k + 1)] * b_data[b_offset + j * b_cols + (k + 1)];
                                oij += a_data[a_offset + i * a_cols + (k + 2)] * b_data[b_offset + j * b_cols + (k + 2)];
                                oij += a_data[a_offset + i * a_cols + (k + 3)] * b_data[b_offset + j * b_cols + (k + 3)];
                            }
                            for (; k < k_end; ++k) {
                                oij += a_data[a_offset + i * a_cols + (k)] * b_data[b_offset + j * b_cols + (k)];
                            }
                            out[out_offset + i * b_rows + j] += oij;
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_btranspose_acc(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset, const size_t a_rows, const size_t a_cols, const size_t b_rows, const size_t b_cols) const {
        if (a_cols != b_cols) {
            throw std::invalid_argument("Invalid dimensions for multiplication");
        }
        constexpr int BS = 64;
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t ii = 0; ii < a_rows; ii += BS) {
            size_t i_end = std::min(ii + BS, a_rows);
            for (size_t jj = 0; jj < b_rows; jj += BS) {
                size_t j_end = std::min(jj + BS, b_rows);
                for (size_t kk = 0; kk < a_cols; kk += BS) {
                    size_t k_end = std::min(kk + BS, a_cols);
                    for (int i = ii; i < i_end; ++i) {
                        for (int j = jj; j < j_end; ++j) {
                            float oij = 0.0f;
                            int k = kk;
                            for (; k + 3 < k_end; k += 4) {
                                oij += a_data[a_offset + i * a_cols + (k)] * b_data[b_offset + j * b_cols + (k)];
                                oij += a_data[a_offset + i * a_cols + (k + 1)] * b_data[b_offset + j * b_cols + (k + 1)];
                                oij += a_data[a_offset + i * a_cols + (k + 2)] * b_data[b_offset + j * b_cols + (k + 2)];
                                oij += a_data[a_offset + i * a_cols + (k + 3)] * b_data[b_offset + j * b_cols + (k + 3)];
                            }
                            for (; k < k_end; ++k) {
                                oij += a_data[a_offset + i * a_cols + (k)] * b_data[b_offset + j * b_cols + (k)];
                            }
                            out[out_offset + i * b_rows + j] += oij;
                        }
                    }
                }
            }
        }
    }

    //Specialised

    void matmul_kernel_strided_atranspose_128x784x128x256_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        #pragma omp parallel for schedule(static)
        for (size_t ii = 0; ii < 784; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)784);
            for (size_t kk = 0; kk < 128; kk += BS) {
                size_t k_end = std::min(kk + BS, (size_t)128);
                for (size_t jj = 0; jj < 256; jj += BS) {
                    size_t j_end = std::min(jj + BS, (size_t)256);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + k * 784 + i];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * 256 + (j)] += aik * b_data[b_offset + k * 256 + (j)];
                                out[out_offset + i * 256 + (j + 1)] += aik * b_data[b_offset + k * 256 + (j + 1)];
                                out[out_offset + i * 256 + (j + 2)] += aik * b_data[b_offset + k * 256 + (j + 2)];
                                out[out_offset + i * 256 + (j + 3)] += aik * b_data[b_offset + k * 256 + (j + 3)];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * 256 + (j)] += aik * b_data[b_offset + k * 256 + (j)];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_atranspose_128x256x128x128_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        for (size_t ii = 0; ii < 256; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)256);
            for (size_t kk = 0; kk < 128; kk += BS) {
                size_t k_end = std::min(kk + BS, (size_t)128);
                for (size_t jj = 0; jj < 128; jj += BS) {
                    size_t j_end = std::min(jj + BS, (size_t)128);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + k * 256 + i];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * 128 + (j)] += aik * b_data[b_offset + k * 128 + (j)];
                                out[out_offset + i * 128 + (j + 1)] += aik * b_data[b_offset + k * 128 + (j + 1)];
                                out[out_offset + i * 128 + (j + 2)] += aik * b_data[b_offset + k * 128 + (j + 2)];
                                out[out_offset + i * 128 + (j + 3)] += aik * b_data[b_offset + k * 128 + (j + 3)];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * 128 + (j)] += aik * b_data[b_offset + k * 128 + (j)];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_atranspose_128x128x128x10_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        for (size_t ii = 0; ii < 128; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)128);
            for (size_t kk = 0; kk < 128; kk += BS) {
                size_t k_end = std::min(kk + BS, (size_t)128);
                for (size_t jj = 0; jj < 10; jj += BS) {
                    size_t j_end = std::min(jj + BS, (size_t)10);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + k * 128 + i];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * 10 + (j)] += aik * b_data[b_offset + k * 10 + (j)];
                                out[out_offset + i * 10 + (j + 1)] += aik * b_data[b_offset + k * 10 + (j + 1)];
                                out[out_offset + i * 10 + (j + 2)] += aik * b_data[b_offset + k * 10 + (j + 2)];
                                out[out_offset + i * 10 + (j + 3)] += aik * b_data[b_offset + k * 10 + (j + 3)];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * 10 + (j)] += aik * b_data[b_offset + k * 10 + (j)];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_btranspose_128x256x784x256_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t ii = 0; ii < 128; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)128);
            for (size_t jj = 0; jj < 784; jj += BS) {
                size_t j_end = std::min(jj + BS, (size_t)784);
                for (size_t kk = 0; kk < 256; kk += BS) {
                    size_t k_end = std::min(kk + BS, (size_t)256);
                    for (int i = ii; i < i_end; ++i) {
                        for (int j = jj; j < j_end; ++j) {
                            float oij = 0.0f;
                            int k = kk;
                            for (; k + 3 < k_end; k += 4) {
                                oij += a_data[a_offset + i * 256 + (k)] * b_data[b_offset + j * 256 + (k)];
                                oij += a_data[a_offset + i * 256 + (k + 1)] * b_data[b_offset + j * 256 + (k + 1)];
                                oij += a_data[a_offset + i * 256 + (k + 2)] * b_data[b_offset + j * 256 + (k + 2)];
                                oij += a_data[a_offset + i * 256 + (k + 3)] * b_data[b_offset + j * 256 + (k + 3)];
                            }
                            for (; k < k_end; ++k) {
                                oij += a_data[a_offset + i * 256 + (k)] * b_data[b_offset + j * 256 + (k)];
                            }
                            out[out_offset + i * 784 + j] += oij;
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_btranspose_128x128x256x128_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        for (size_t ii = 0; ii < 128; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)128);
            for (size_t jj = 0; jj < 256; jj += BS) {
                size_t j_end = std::min(jj + BS, (size_t)256);
                for (size_t kk = 0; kk < 128; kk += BS) {
                    size_t k_end = std::min(kk + BS, (size_t)128);
                    for (int i = ii; i < i_end; ++i) {
                        for (int j = jj; j < j_end; ++j) {
                            float oij = 0.0f;
                            int k = kk;
                            for (; k + 3 < k_end; k += 4) {
                                oij += a_data[a_offset + i * 128 + (k)] * b_data[b_offset + j * 128 + (k)];
                                oij += a_data[a_offset + i * 128 + (k + 1)] * b_data[b_offset + j * 128 + (k + 1)];
                                oij += a_data[a_offset + i * 128 + (k + 2)] * b_data[b_offset + j * 128 + (k + 2)];
                                oij += a_data[a_offset + i * 128 + (k + 3)] * b_data[b_offset + j * 128 + (k + 3)];
                            }
                            for (; k < k_end; ++k) {
                                oij += a_data[a_offset + i * 128 + (k)] * b_data[b_offset + j * 128 + (k)];
                            }
                            out[out_offset + i * 256 + j] += oij;
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_btranspose_128x10x128x10_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        for (size_t ii = 0; ii < 128; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)128);
            for (size_t jj = 0; jj < 128; jj += BS) {
                size_t j_end = std::min(jj + BS, (size_t)128);
                for (size_t kk = 0; kk < 10; kk += BS) {
                    size_t k_end = std::min(kk + BS, (size_t)10);
                    for (int i = ii; i < i_end; ++i) {
                        for (int j = jj; j < j_end; ++j) {
                            float oij = 0.0f;
                            int k = kk;
                            for (; k + 3 < k_end; k += 4) {
                                oij += a_data[a_offset + i * 10 + (k)] * b_data[b_offset + j * 10 + (k)];
                                oij += a_data[a_offset + i * 10 + (k + 1)] * b_data[b_offset + j * 10 + (k + 1)];
                                oij += a_data[a_offset + i * 10 + (k + 2)] * b_data[b_offset + j * 10 + (k + 2)];
                                oij += a_data[a_offset + i * 10 + (k + 3)] * b_data[b_offset + j * 10 + (k + 3)];
                            }
                            for (; k < k_end; ++k) {
                                oij += a_data[a_offset + i * 10 + (k)] * b_data[b_offset + j * 10 + (k)];
                            }
                            out[out_offset + i * 128 + j] += oij;
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_128x784x256_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        #pragma omp parallel for schedule(static)
        for (size_t ii = 0; ii < 128; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)128);
            for (size_t kk = 0; kk < 784; kk += BS) {
                size_t k_end = std::min(kk + BS, (size_t)784);
                for (size_t jj = 0; jj < 256; jj += BS) {
                    size_t j_end = std::min(jj + BS, (size_t)256);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + i * 784 + k];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * 256 + (j)] += aik * b_data[b_offset + k * 256 + (j)];
                                out[out_offset + i * 256 + (j + 1)] += aik * b_data[b_offset + k * 256 + (j + 1)];
                                out[out_offset + i * 256 + (j + 2)] += aik * b_data[b_offset + k * 256 + (j + 2)];
                                out[out_offset + i * 256 + (j + 3)] += aik * b_data[b_offset + k * 256 + (j + 3)];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * 256 + (j)] += aik * b_data[b_offset + k * 256 + (j)];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_128x256x128_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        for (size_t ii = 0; ii < 128; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)128);
            for (size_t kk = 0; kk < 256; kk += BS) {
                size_t k_end = std::min(kk + BS, (size_t)256);
                for (size_t jj = 0; jj < 128; jj += BS) {
                    size_t j_end = std::min(jj + BS, (size_t)128);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + i * 256 + k];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * 128 + (j)] += aik * b_data[b_offset + k * 128 + (j)];
                                out[out_offset + i * 128 + (j + 1)] += aik * b_data[b_offset + k * 128 + (j + 1)];
                                out[out_offset + i * 128 + (j + 2)] += aik * b_data[b_offset + k * 128 + (j + 2)];
                                out[out_offset + i * 128 + (j + 3)] += aik * b_data[b_offset + k * 128 + (j + 3)];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * 128 + (j)] += aik * b_data[b_offset + k * 128 + (j)];
                            }
                        }
                    }
                }
            }
        }
    }

    void matmul_kernel_strided_128x128x10_contiguous(const float* a_data, const float* b_data, float* out, const size_t a_offset, const size_t b_offset, const size_t out_offset) const {
        constexpr int BS = 64;
        for (size_t ii = 0; ii < 128; ii += BS) {
            size_t i_end = std::min(ii + BS, (size_t)128);
            for (size_t kk = 0; kk < 128; kk += BS) {
                size_t k_end = std::min(kk + BS, (size_t)128);
                for (size_t jj = 0; jj < 10; jj += BS) {
                    size_t j_end = std::min(jj + BS, (size_t)10);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float aik = a_data[a_offset + i * 128 + k];
                            size_t j = jj;
                            for (; j + 3 < j_end; j += 4) {
                                out[out_offset + i * 10 + (j)] += aik * b_data[b_offset + k * 10 + (j)];
                                out[out_offset + i * 10 + (j + 1)] += aik * b_data[b_offset + k * 10 + (j + 1)];
                                out[out_offset + i * 10 + (j + 2)] += aik * b_data[b_offset + k * 10 + (j + 2)];
                                out[out_offset + i * 10 + (j + 3)] += aik * b_data[b_offset + k * 10 + (j + 3)];
                            }
                            for (; j < j_end; ++j) {
                                out[out_offset + i * 10 + (j)] += aik * b_data[b_offset + k * 10 + (j)];
                            }
                        }
                    }
                }
            }
        }
    }

    //Backpropagation

    //Topological Sort

    void build_topo(std::vector<std::shared_ptr<TensorImp>>* topo, std::set<TensorImp*>* visited, std::shared_ptr<TensorImp>& n) {
        if (visited->contains(n.get())) return;
        visited->insert(n.get());
        for (auto& x : n->parents) {
            build_topo(topo, visited, x);
        }
        topo->push_back(n);
    }

    //Backward Methods

    void backward_reshape(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];

        if (!n->is_contiguous()) {
            throw std::runtime_error("backward_reshape requires contiguous Tensors");
        }

        for (size_t i = 0; i < n->length; ++i) {
            p->grad_storage->data[p->offset + i] += n->grad_storage->data[n->offset + i];
        }
    }

    void backward_slice(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];

        std::vector<size_t> coord(p->shape.size(), 0);
        size_t ni = n->offset;

        for (size_t x = 0; x < n->length; ++x) {
            size_t pi = p->offset;
            for (int d = 0; d < p->shape.size(); ++d) {
                pi += ((*n->buffer)[d] + (*n->buffer2)[d] * coord[d]) * p->strides[d];
            }

            p->grad_storage->data[pi] += n->grad_storage->data[ni];

            for (int d = p->shape.size() - 1; d >= 0; --d) {
                ++coord[d];

                ni += n->strides[d];

                if (coord[d] < n->shape[d]) break;

                coord[d] = 0;

                ni -= n->shape[d] * n->strides[d];
            }
        }
    }

    void backward_add(std::shared_ptr<TensorImp> n) {
        if (n->op_axis == -1) {
            auto& p1 = n->parents[0];
            auto& p2 = n->parents[1];

            const size_t ndim = n->shape.size();

            std::vector<size_t> st1(ndim, 0), st2(ndim, 0);

            size_t d1 = ndim - p1->shape.size();
            size_t d2 = ndim - p2->shape.size();

            for (size_t d = 0; d < p1->shape.size(); ++d) {
                st1[d1 + d] = (p1->shape[d] == 1) ? 0 : p1->strides[d];
            }

            for (size_t d = 0; d < p2->shape.size(); ++d) {
                st2[d2 + d] = (p2->shape[d] == 1) ? 0 : p2->strides[d];
            }

            std::vector<size_t> coord(ndim, 0);

            size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

            for (size_t x = 0; x < n->length; ++x) {

                float g = n->grad_storage->data[ni];

                if (p1->requires_grad)
                    p1->grad_storage->data[p1i] += g;

                if (p2->requires_grad)
                    p2->grad_storage->data[p2i] += g;
                
                for (int d = ndim - 1; d >= 0; --d) {

                    ++coord[d];

                    p1i += st1[d];
                    p2i += st2[d];
                    ni  += n->strides[d];

                    if (coord[d] < n->shape[d])
                        break;

                    coord[d] = 0;

                    p1i -= n->shape[d] * st1[d];
                    p2i -= n->shape[d] * st2[d];
                    ni  -= n->shape[d] * n->strides[d];
                }
            }
        }
        else {
            auto& p = n->parents[0];

            std::vector<size_t> coord(p->shape.size(), 0);
            size_t pi = p->offset;
            size_t ni = n->offset;

            for (size_t x = 0; x < p->length; ++x) {
                p->grad_storage->data[pi] += n->grad_storage->data[ni];

                for (int d = p->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    pi += p->strides[d];
                    ni += n->strides[d];

                    if (coord[d] < p->shape[d]) break;

                    coord[d] = 0;

                    pi -= p->shape[d] * p->strides[d];
                    ni -= n->shape[d] * n->strides[d];
                }
            }
        }
    }

    void backward_sub(std::shared_ptr<TensorImp> n) {
        if (n->op_axis == -1) {
            auto& p1 = n->parents[0];
            auto& p2 = n->parents[1];

            const size_t ndim = n->shape.size();

            std::vector<size_t> st1(ndim, 0), st2(ndim, 0);

            size_t d1 = ndim - p1->shape.size();
            size_t d2 = ndim - p2->shape.size();

            for (size_t d = 0; d < p1->shape.size(); ++d) {
                st1[d1 + d] = (p1->shape[d] == 1) ? 0 : p1->strides[d];
            }

            for (size_t d = 0; d < p2->shape.size(); ++d) {
                st2[d2 + d] = (p2->shape[d] == 1) ? 0 : p2->strides[d];
            }

            std::vector<size_t> coord(ndim, 0);

            size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

            for (size_t x = 0; x < n->length; ++x) {

                float g = n->grad_storage->data[ni];

                if (p1->requires_grad)
                    p1->grad_storage->data[p1i] += g;

                if (p2->requires_grad)
                    p2->grad_storage->data[p2i] -= g;
                
                for (int d = ndim - 1; d >= 0; --d) {

                    ++coord[d];

                    p1i += st1[d];
                    p2i += st2[d];
                    ni  += n->strides[d];

                    if (coord[d] < n->shape[d])
                        break;

                    coord[d] = 0;

                    p1i -= n->shape[d] * st1[d];
                    p2i -= n->shape[d] * st2[d];
                    ni  -= n->shape[d] * n->strides[d];
                }
            }
        }
        else {
            auto& p = n->parents[0];

            std::vector<size_t> coord(p->shape.size(), 0);
            size_t pi = p->offset;
            size_t ni = n->offset;

            for (size_t x = 0; x < p->length; ++x) {
                p->grad_storage->data[pi] += n->grad_storage->data[ni];

                for (int d = p->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    pi += p->strides[d];
                    ni += n->strides[d];

                    if (coord[d] < p->shape[d]) break;

                    coord[d] = 0;

                    pi -= p->shape[d] * p->strides[d];
                    ni -= n->shape[d] * n->strides[d];
                }
            }
        }
    }

    void backward_mul(std::shared_ptr<TensorImp> n) {
        if (n->op_axis == -1) {
            auto& p1 = n->parents[0];
            auto& p2 = n->parents[1];

            const size_t ndim = n->shape.size();

            std::vector<size_t> s1(ndim, 1), s2(ndim, 1), st1(ndim, 0), st2(ndim, 0);

            size_t d1 = ndim - p1->shape.size();
            size_t d2 = ndim - p2->shape.size();

            for (size_t d = 0; d < p1->shape.size(); ++d) {
                s1[d1 + d] = p1->shape[d];
                st1[d1 + d] = (p1->shape[d] == 1) ? 0 : p1->strides[d];
            }

            for (size_t d = 0; d < p2->shape.size(); ++d) {
                s2[d2 + d] = p2->shape[d];
                st2[d2 + d] = (p2->shape[d] == 1) ? 0 : p2->strides[d];
            }

            std::vector<size_t> coord(ndim, 0);

            size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

            for (size_t x = 0; x < n->length; ++x) {

                float g = n->grad_storage->data[ni];

                if (p1->requires_grad)
                    p1->grad_storage->data[p1i] += g * p2->storage->data[p2i];

                if (p2->requires_grad)
                    p2->grad_storage->data[p2i] += g * p1->storage->data[p1i];
                
                for (int d = ndim - 1; d >= 0; --d) {

                    ++coord[d];

                    p1i += st1[d];
                    p2i += st2[d];
                    ni  += n->strides[d];

                    if (coord[d] < n->shape[d])
                        break;

                    coord[d] = 0;

                    p1i -= n->shape[d] * st1[d];
                    p2i -= n->shape[d] * st2[d];
                    ni  -= n->shape[d] * n->strides[d];
                }
            }
        }
        else {
            auto& p = n->parents[0];

            std::vector<size_t> coord(p->shape.size(), 0);
            size_t pi = p->offset;
            size_t ni = n->offset;

            for (size_t x = 0; x < p->length; ++x) {
                p->grad_storage->data[pi] += n->grad_storage->data[ni] * n->alpha;

                for (int d = p->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    pi += p->strides[d];
                    ni += n->strides[d];

                    if (coord[d] < p->shape[d]) break;

                    coord[d] = 0;

                    pi -= p->shape[d] * p->strides[d];
                    ni -= n->shape[d] * n->strides[d];
                }
            }
        }
    }

    void backward_div(std::shared_ptr<TensorImp> n) {
        if (n->op_axis == -1) {
            auto& p1 = n->parents[0];
            auto& p2 = n->parents[1];

            const size_t ndim = n->shape.size();

            std::vector<size_t> s1(ndim, 1), s2(ndim, 1), st1(ndim, 0), st2(ndim, 0);

            size_t d1 = ndim - p1->shape.size();
            size_t d2 = ndim - p2->shape.size();

            for (size_t d = 0; d < p1->shape.size(); ++d) {
                s1[d1 + d] = p1->shape[d];
                st1[d1 + d] = (p1->shape[d] == 1) ? 0 : p1->strides[d];
            }

            for (size_t d = 0; d < p2->shape.size(); ++d) {
                s2[d2 + d] = p2->shape[d];
                st2[d2 + d] = (p2->shape[d] == 1) ? 0 : p2->strides[d];
            }

            std::vector<size_t> coord(ndim, 0);

            size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

            for (size_t x = 0; x < n->length; ++x) {

                float g = n->grad_storage->data[ni];

                if (p1->requires_grad)
                    p1->grad_storage->data[p1i] += g / p2->storage->data[p2i];

                if (p2->requires_grad)
                    p2->grad_storage->data[p2i] -= g * p1->storage->data[p1i] / (p2->storage->data[p2i] * p2->storage->data[p2i]);
                
                for (int d = ndim - 1; d >= 0; --d) {

                    ++coord[d];

                    p1i += st1[d];
                    p2i += st2[d];
                    ni  += n->strides[d];

                    if (coord[d] < n->shape[d])
                        break;

                    coord[d] = 0;

                    p1i -= n->shape[d] * st1[d];
                    p2i -= n->shape[d] * st2[d];
                    ni  -= n->shape[d] * n->strides[d];
                }
            }
        }
        else {
            auto& p = n->parents[0];

            std::vector<size_t> coord(p->shape.size(), 0);
            size_t pi = p->offset;
            size_t ni = n->offset;

            for (size_t x = 0; x < p->length; ++x) {
                p->grad_storage->data[pi] += n->grad_storage->data[ni] / n->alpha;

                for (int d = p->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    pi += p->strides[d];
                    ni += n->strides[d];

                    if (coord[d] < p->shape[d]) break;

                    coord[d] = 0;

                    pi -= p->shape[d] * p->strides[d];
                    ni -= n->shape[d] * n->strides[d];
                }
            }
        }
    }

    void backward_pow(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];

        std::vector<size_t> coord(p->shape.size(), 0);
        size_t pi = p->offset;
        size_t ni = n->offset;

        for (size_t x = 0; x < p->length; ++x) {
            p->grad_storage->data[pi] += n->grad_storage->data[ni] * n->alpha * std::pow(p->storage->data[pi], n->alpha - 1);

            for (int d = p->shape.size() - 1; d >= 0; --d) {
                ++coord[d];

                pi += p->strides[d];
                ni += n->strides[d];

                if (coord[d] < p->shape[d]) break;

                coord[d] = 0;

                pi -= p->shape[d] * p->strides[d];
                ni -= n->shape[d] * n->strides[d];
            }
        }
    }

    void backward_matmul(std::shared_ptr<TensorImp> n) {
        auto& p1 = n->parents[0];
        auto& p2 = n->parents[1];

        if (p1->requires_grad) {
            const bool z = p2->is_contiguous() && n->is_contiguous();

            if (z && n->shape[n->shape.size() - 2] == 128 && n->shape[n->shape.size() - 1] == 256 && p2->shape[p2->shape.size() - 2] == 784 && p2->shape[p2->shape.size() - 1] == 256) {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

                for (size_t x = 0; x < n->length; x += n->shape[n->shape.size() - 1] * n->shape[n->shape.size() - 2]) {
                    matmul_kernel_strided_btranspose_128x256x784x256_contiguous(n->grad_storage->data.data(), p2->storage->data.data(), p1->grad_storage->data.data(), ni, p2i, p1i);

                    for (int d = static_cast<int>(n->shape.size()) - 3;
                        d >= 0; --d) {
                        ++coord[d];

                        p1i += p1->strides[d];
                        p2i += p2->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        p1i -= n->shape[d] * p1->strides[d];
                        p2i -= n->shape[d] * p2->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
            else if (z && n->shape[n->shape.size() - 2] == 128 && n->shape[n->shape.size() - 1] == 128 && p2->shape[p2->shape.size() - 2] == 256 && p2->shape[p2->shape.size() - 1] == 128) {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

                for (size_t x = 0; x < n->length; x += n->shape[n->shape.size() - 1] * n->shape[n->shape.size() - 2]) {
                    matmul_kernel_strided_btranspose_128x128x256x128_contiguous(n->grad_storage->data.data(), p2->storage->data.data(), p1->grad_storage->data.data(), ni, p2i, p1i);

                    for (int d = static_cast<int>(n->shape.size()) - 3;
                        d >= 0; --d) {
                        ++coord[d];

                        p1i += p1->strides[d];
                        p2i += p2->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        p1i -= n->shape[d] * p1->strides[d];
                        p2i -= n->shape[d] * p2->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
            else if (z && n->shape[n->shape.size() - 2] == 128 && n->shape[n->shape.size() - 1] == 10 && p2->shape[p2->shape.size() - 2] == 128 && p2->shape[p2->shape.size() - 1] == 10) {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

                for (size_t x = 0; x < n->length; x += n->shape[n->shape.size() - 1] * n->shape[n->shape.size() - 2]) {
                    matmul_kernel_strided_btranspose_128x10x128x10_contiguous(n->grad_storage->data.data(), p2->storage->data.data(), p1->grad_storage->data.data(), ni, p2i, p1i);

                    for (int d = static_cast<int>(n->shape.size()) - 3;
                        d >= 0; --d) {
                        ++coord[d];

                        p1i += p1->strides[d];
                        p2i += p2->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        p1i -= n->shape[d] * p1->strides[d];
                        p2i -= n->shape[d] * p2->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
            else {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

                for (size_t x = 0; x < n->length; x += n->shape[n->shape.size() - 1] * n->shape[n->shape.size() - 2]) {
                    matmul_kernel_strided_btranspose(n->grad_storage->data.data(), p2->storage->data.data(), p1->grad_storage->data.data(), ni, p2i, p1i, n->shape[n->shape.size() - 2], n->shape[n->shape.size() - 1], p2->shape[p2->shape.size() - 2], p2->shape[p2->shape.size() - 1], n->strides[n->shape.size() - 2], n->strides[n->shape.size() - 1], p2->strides[p2->shape.size() - 2], p2->strides[p2->shape.size() - 1], p1->strides[p1->shape.size() - 2], p1->strides[p1->shape.size() - 1]);

                    for (int d = static_cast<int>(n->shape.size()) - 3;
                        d >= 0; --d) {
                        ++coord[d];

                        p1i += p1->strides[d];
                        p2i += p2->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        p1i -= n->shape[d] * p1->strides[d];
                        p2i -= n->shape[d] * p2->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
        }

        if (p2->requires_grad) {
            const bool z = p1->is_contiguous() && n->is_contiguous();

            if (z && p1->shape[p1->shape.size() - 2] == 128 && p1->shape[p1->shape.size() - 1] == 784 && n->shape[n->shape.size() - 2] == 128 && n->shape[n->shape.size() - 1] == 256) {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

                for (size_t x = 0; x < n->length; x += n->shape[n->shape.size() - 1] * n->shape[n->shape.size() - 2]) {
                    matmul_kernel_strided_atranspose_128x784x128x256_contiguous(p1->storage->data.data(), n->grad_storage->data.data(), p2->grad_storage->data.data(), p1i, ni, p2i);

                    for (int d = static_cast<int>(n->shape.size()) - 3;
                        d >= 0; --d) {
                        ++coord[d];

                        p1i += p1->strides[d];
                        p2i += p2->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        p1i -= n->shape[d] * p1->strides[d];
                        p2i -= n->shape[d] * p2->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
            else if (z && p1->shape[p1->shape.size() - 2] == 128 && p1->shape[p1->shape.size() - 1] == 256 && n->shape[n->shape.size() - 2] == 128 && n->shape[n->shape.size() - 1] == 128) {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

                for (size_t x = 0; x < n->length; x += n->shape[n->shape.size() - 1] * n->shape[n->shape.size() - 2]) {
                    matmul_kernel_strided_atranspose_128x256x128x128_contiguous(p1->storage->data.data(), n->grad_storage->data.data(), p2->grad_storage->data.data(), p1i, ni, p2i);

                    for (int d = static_cast<int>(n->shape.size()) - 3;
                        d >= 0; --d) {
                        ++coord[d];

                        p1i += p1->strides[d];
                        p2i += p2->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        p1i -= n->shape[d] * p1->strides[d];
                        p2i -= n->shape[d] * p2->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
            else if (z && p1->shape[p1->shape.size() - 2] == 128 && p1->shape[p1->shape.size() - 1] == 128 && n->shape[n->shape.size() - 2] == 128 && n->shape[n->shape.size() - 1] == 10) {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

                for (size_t x = 0; x < n->length; x += n->shape[n->shape.size() - 1] * n->shape[n->shape.size() - 2]) {
                    matmul_kernel_strided_atranspose_128x128x128x10_contiguous(p1->storage->data.data(), n->grad_storage->data.data(), p2->grad_storage->data.data(), p1i, ni, p2i);

                    for (int d = static_cast<int>(n->shape.size()) - 3;
                        d >= 0; --d) {
                        ++coord[d];

                        p1i += p1->strides[d];
                        p2i += p2->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        p1i -= n->shape[d] * p1->strides[d];
                        p2i -= n->shape[d] * p2->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
            else {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t p1i = p1->offset, p2i = p2->offset, ni = n->offset;

                for (size_t x = 0; x < n->length; x += n->shape[n->shape.size() - 1] * n->shape[n->shape.size() - 2]) {
                    matmul_kernel_strided_atranspose(p1->storage->data.data(), n->grad_storage->data.data(), p2->grad_storage->data.data(), p1i, ni, p2i, p1->shape[p1->shape.size() - 2], p1->shape[p1->shape.size() - 1], n->shape[n->shape.size() - 2], n->shape[n->shape.size() - 1], p1->strides[p1->shape.size() - 2], p1->strides[p1->shape.size() - 1], n->strides[n->shape.size() - 2], n->strides[n->shape.size() - 1], p2->strides[p2->shape.size() - 2], p2->strides[p2->shape.size() - 1]);

                    for (int d = static_cast<int>(n->shape.size()) - 3;
                        d >= 0; --d) {
                        ++coord[d];

                        p1i += p1->strides[d];
                        p2i += p2->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        p1i -= n->shape[d] * p1->strides[d];
                        p2i -= n->shape[d] * p2->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
        }
    }

    void backward_conv2d(std::shared_ptr<TensorImp> n) {
        auto& p1 = n->parents[0];
        auto& p2 = n->parents[1];
        auto& p3 = n->parents[2];

        if (p3->requires_grad) {
            size_t c = n->shape[1], b = n->shape[0], h = n->shape[2], w = n->shape[3];
            size_t b_stride = n->strides[0], c_stride = n->strides[1];
            #pragma omp parallel for schedule(static)
            for (size_t o = 0; o < c; ++o) {
                float s = 0.0f;
                for (size_t x = 0; x < b; ++x) {
                    for (size_t i = 0; i < h; ++i) {
                        for (size_t j = 0; j < w; ++j) {
                            s += n->grad_storage->data[x * b_stride + o * c_stride + i * w + j];
                        }
                    }
                }
                p3->grad_storage->data[o] += s;
            }
        }

        if (p1->requires_grad) {
            size_t b = n->shape[0], c = n->shape[1], h = n->shape[2], w = n->shape[3], k = n->convmd->kernel_size;
            std::vector<float> dx(b * p1->shape[1] * k * k * h * w, 0.0f);
            #pragma omp parallel for schedule(static)
            for (size_t x = 0; x < b; ++x) {
                matmul_kernel_atranspose(p2->storage->data.data(), n->grad_storage->data.data() + x * c * h * w, dx.data() + x * k * k * h * w * p1->shape[1], 0, 0, 0, c, p1->shape[1] * k * k, c, h * w);
                col2im(dx.data() + x * k * k * h * w * p1->shape[1], p1->grad_storage->data.data() + x * p1->shape[1] * p1->shape[2] * p1->shape[3], p1->shape[1], p1->shape[2], p1->shape[3], k, n->convmd->stride, n->convmd->padding);
            }
        }

        if (p2->requires_grad) {
            size_t b = n->shape[0], c = n->shape[1], h = n->shape[2], w = n->shape[3], k = n->convmd->kernel_size;
            for (size_t x = 0; x < b; ++x) {
                matmul_kernel_btranspose_acc(n->grad_storage->data.data() + x * c * h * w, n->convmd->col.data() + x * k * k * h * w * p1->shape[1], p2->grad_storage->data.data(), 0, 0, 0, c, h * w, p1->shape[1] * k * k, h * w);
            }
        }
    }

    void backward_maxpool2d(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];

        if (p->requires_grad) {
            for (size_t i = 0; i < n->length; ++i) {
                p->grad_storage->data[(*(n->buffer))[i]] += n->grad_storage->data[n->offset + i];
            }
        }
    }

    void backward_sum(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            if (n->op_axis == -1) {
                float x = n->grad_storage->data[0];
                std::vector<size_t> coord(p->shape.size(), 0);
                size_t pi = p->offset;
                for (size_t y = 0; y < p->length; ++y) {
                    p->grad_storage->data[pi] += x;

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];

                        pi += p->strides[d];

                        if (coord[d] < p->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->shape[d] * p->strides[d];
                    }
                }
            }
            else {
                std::vector<size_t> coord(p->shape.size(), 0);
                size_t axis = n->op_axis;

                while (true) {
                    size_t ti = p->offset;
                    size_t oi = n->offset;

                    for (int d = 0; d < p->shape.size(); ++d) {
                        ti += coord[d] * p->strides[d];
                        oi += coord[d] * n->strides[d];
                    }

                    for (size_t x = 0; x < p->shape[axis]; ++x) {
                        p->grad_storage->data[ti] += n->grad_storage->data[oi];
                        ti += p->strides[axis];
                    }

                    bool completed = true;

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        if (d == axis) continue;

                        ++coord[d];

                        if (coord[d] < p->shape[d]) {
                            completed = false;
                            break;
                        }

                        coord[d] = 0;
                    }

                    if (completed) break;
                }
            }
        }
    }

    void backward_mean(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            if (n->op_axis == -1) {
                float x = n->grad_storage->data[0];
                std::vector<size_t> coord(p->shape.size(), 0);
                size_t pi = p->offset;
                for (size_t y = 0; y < p->length; ++y) {
                    p->grad_storage->data[pi] += x / p->length;

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];

                        pi += p->strides[d];

                        if (coord[d] < p->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->shape[d] * p->strides[d];
                    }
                }
            }
            else {
                std::vector<size_t> coord(p->shape.size(), 0);
                size_t axis = n->op_axis;

                while (true) {
                    size_t ti = p->offset;
                    size_t oi = n->offset;

                    for (int d = 0; d < p->shape.size(); ++d) {
                        ti += coord[d] * p->strides[d];
                        oi += coord[d] * n->strides[d];
                    }

                    for (size_t x = 0; x < p->shape[axis]; ++x) {
                        p->grad_storage->data[ti] += n->grad_storage->data[oi] / p->shape[axis];
                        ti += p->strides[axis];
                    }

                    bool completed = true;

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        if (d == axis) continue;

                        ++coord[d];

                        if (coord[d] < p->shape[d]) {
                            completed = false;
                            break;
                        }

                        coord[d] = 0;
                    }

                    if (completed) break;
                }
            }
        }
    }

    void backward_variance(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            if (n->op_axis == -1) {
                float x = n->grad_storage->data[0];
                float z = (*n->bufferf)[0];
                std::vector<size_t> coord(p->shape.size(), 0);
                size_t pi = p->offset;
                for (size_t y = 0; y < p->length; ++y) {
                    p->grad_storage->data[pi] += x * 2 * (p->storage->data[pi] - z) / p->length;

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];

                        pi += p->strides[d];

                        if (coord[d] < p->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->shape[d] * p->strides[d];
                    }
                }
            }
            else {
                std::vector<size_t> coord(p->shape.size(), 0);
                size_t axis = n->op_axis;
                std::vector<float>& m = *(n->bufferf);

                while (true) {
                    size_t ti = p->offset;
                    size_t oi = n->offset;
                    size_t mi = 0;

                    for (int d = 0; d < p->shape.size(); ++d) {
                        ti += coord[d] * p->strides[d];
                        oi += coord[d] * n->strides[d];
                        mi += (d == axis ? 0 : coord[d]) * (*n->buffer)[d];
                    }

                    for (size_t x = 0; x < p->shape[axis]; ++x) {
                        p->grad_storage->data[ti] += n->grad_storage->data[oi] * 2 * (p->storage->data[ti] - m[mi]) / p->shape[axis];
                        ti += p->strides[axis];
                    }

                    bool completed = true;

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        if (d == axis) continue;

                        ++coord[d];

                        if (coord[d] < p->shape[d]) {
                            completed = false;
                            break;
                        }

                        coord[d] = 0;
                    }

                    if (completed) break;
                }
            }
        }
    }

    void backward_transpose(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            std::vector<size_t> coord(p->shape.size(), 0);
            for (size_t x = 0; x < p->length; ++x) {
                size_t r = x;

                for (int d = static_cast<int>(n->shape.size() - 1); d >= 0; --d) {
                    coord[d] = r % n->shape[d];
                    r /= n->shape[d];
                }

                std::vector<size_t> pc = coord;
                std::swap(pc[n->tra], pc[n->trb]);

                size_t pi = p->offset;
                size_t ni = n->offset;

                for (int d = 0; d < n->shape.size(); ++d) {
                    pi += pc[d] * p->strides[d];
                    ni += coord[d] * n->strides[d];
                }

                p->grad_storage->data[pi] += n->grad_storage->data[ni];
            }
        }
    }

    void backward_relu(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            if (p->is_contiguous() && n->is_contiguous()) {
                for (size_t i = 0; i < p->length; ++i) {
                    if (p->storage->data[p->offset + i] >= 0) {
                        p->grad_storage->data[p->offset + i] += n->grad_storage->data[n->offset + i];
                    }
                }
            }
            else {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t ni = n->offset;
                size_t pi = p->offset;
                for (size_t x = 0; x < n->length; ++x) {
                    if (p->storage->data[pi] >= 0) {
                        p->grad_storage->data[pi] += n->grad_storage->data[ni];
                    }

                    for (int d = n->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];
                        
                        pi += p->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->shape[d] * p->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
        }
    }

    void backward_leakyrelu(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            if (p->is_contiguous() && n->is_contiguous()) {
                for (size_t i = 0; i < p->length; ++i) {
                    if (p->storage->data[p->offset + i] >= 0) {
                        p->grad_storage->data[p->offset + i] += n->grad_storage->data[n->offset + i];
                    }
                    else {
                        p->grad_storage->data[p->offset + i] += n->alpha * n->grad_storage->data[n->offset + i];
                    }
                }
            }
            else {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t ni = n->offset;
                size_t pi = p->offset;
                for (size_t x = 0; x < n->length; ++x) {
                    float a = p->storage->data[pi];
                    p->grad_storage->data[pi] += n->grad_storage->data[ni] * (a >= 0 ? 1 : n->alpha);

                    for (int d = n->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];
                        
                        pi += p->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->shape[d] * p->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
        }
    }

    void backward_gelu(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            if (p->is_contiguous() && n->is_contiguous()) {
                for (size_t i = 0; i < p->length; ++i) {
                    float x = p->storage->data[p->offset + i];
                    float z = std::tanh(std::sqrt(2.0f / std::numbers::pi) * (x + 0.044715 * x * x * x));
                    p->grad_storage->data[p->offset + i] += n->grad_storage->data[n->offset + i] * 0.5f * (1 + z + x * std::sqrt(2.0f / std::numbers::pi) * (1 + 0.134145 * x * x) * (1 - z * z)); 
                }
            }
            else {
                std::vector<size_t> coord(n->shape.size(), 0);
                size_t ni = n->offset;
                size_t pi = p->offset;
                for (size_t x = 0; x < n->length; ++x) {
                    float v = p->storage->data[pi];
                    float z = std::tanh(std::sqrt(2.0f / std::numbers::pi) * (v + 0.044715 * v * v * v));
                    p->grad_storage->data[pi] += n->grad_storage->data[ni] * 0.5f * (1 + z + v * std::sqrt(2.0f / std::numbers::pi) * (1 + 0.134145 * v * v) * (1 - z * z));

                    for (int d = n->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];
                        
                        pi += p->strides[d];
                        ni += n->strides[d];

                        if (coord[d] < n->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->shape[d] * p->strides[d];
                        ni -= n->shape[d] * n->strides[d];
                    }
                }
            }
        }
    }

    void backward_sigmoid(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            std::vector<size_t> coord(p->shape.size(), 0);
            size_t pi = p->offset, ni = n->offset;
            for (size_t i = 0; i < p->length; ++i) {
                p->grad_storage->data[pi] += n->grad_storage->data[ni] * n->storage->data[ni] * (1.0f - n->storage->data[ni]);

                for (int d = p->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    pi += p->strides[d];
                    ni += n->strides[d];

                    if (coord[d] < p->shape[d]) break;

                    coord[d] = 0;

                    pi -= p->shape[d] * p->strides[d];
                    ni -= n->shape[d] * n->strides[d];
                }
            }
        }
    }

    void backward_concat(std::shared_ptr<TensorImp> n) {
        size_t axis = n->op_axis;
        size_t o = 0;
        for (auto& t : n->parents) {
            std::vector<size_t> coord(t->shape.size(), 0);
            size_t ti = t->offset;

            for (size_t x = 0; x < t->length; ++x) {
                size_t j = 0;

                for (int d = 0; d < t->shape.size(); ++d) {
                    j += (d == axis ? (coord[d] + o) : coord[d]) * n->strides[d];
                }

                t->grad_storage->data[ti] += n->grad_storage->data[j];

                for (int d = t->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += t->strides[d];

                    if (coord[d] < t->shape[d]) break;

                    coord[d] = 0;

                    ti -= t->shape[d] * t->strides[d];
                }
            }
            o += t->shape[axis];
        }
    }

    void backward_embed(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        if (p->requires_grad) {
            size_t sl = n->buffer->size();
            for (size_t i = 0; i < sl; ++i) {
                size_t token = (*n->buffer)[i];
                size_t x = p->offset + token * p->strides[0];
                for (size_t j = 0; j < p->shape[1]; ++j) {
                    p->grad_storage->data[x + j * p->strides[1]] += n->grad_storage->data[i * p->shape[1] + j];
                }
            }
        }
    }

    void backward_softmax(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        size_t axis = n->op_axis;

        std::vector<size_t> coord(p->shape.size(), 0);

        while (true) {
            float cs = 0;
            size_t pi = p->offset;
            size_t ni = n->offset;

            for (int d = 0; d < p->shape.size(); ++d) {
                pi += coord[d] * p->strides[d];
                ni += coord[d] * n->strides[d];
            }

            for (size_t j = 0; j < p->shape[axis]; ++j) {
                cs += n->grad_storage->data[ni] * n->storage->data[ni];
                ni += n->strides[axis];
            }

            ni -= n->shape[axis] * n->strides[axis];

            for (size_t j = 0; j < p->shape[axis]; ++j) {
                p->grad_storage->data[pi] += n->storage->data[ni] * (n->grad_storage->data[ni] - cs);
                pi += p->strides[axis];
                ni += n->strides[axis];
            }

            bool completed = true;

            for (int d = p->shape.size() - 1; d >= 0; --d) {
                if (d == axis) continue;

                ++coord[d];

                if (coord[d] < p->shape[d]) {
                    completed = false;
                    break;
                }

                coord[d] = 0;
            }

            if (completed) break;
        }
    }

    void backward_sce(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        auto& t = n->parents[1];
        if (p->requires_grad) {
            if (n->op_axis == -1) {
                std::vector<float> exps(p->length);
                float x, cs = 0;
                std::vector<size_t> coord(p->shape.size(), 0);
                size_t pi = p->offset;
                float m = p->max(-1)[0];
                for (size_t i = 0; i < p->length; ++i) {
                    x = std::exp(p->storage->data[pi] - m);
                    cs += x;
                    exps[i] = x;

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];

                        pi += p->strides[d];

                        if (coord[d] < p->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->shape[d] * p->strides[d];
                    }
                }
                pi = p->offset;
                size_t tai = t->offset;
                std::fill(coord.begin(), coord.end(), 0);
                for (size_t i = 0; i < p->length; ++i) {
                    p->grad_storage->data[pi] += n->grad_storage->data[0] * (exps[i] / cs - t->storage->data[tai]);

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];

                        pi += p->strides[d];
                        tai += t->strides[d];

                        if (coord[d] < p->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->shape[d] * p->strides[d];
                        tai -= p->shape[d] * t->strides[d];
                    }
                }
            }
            else {
                std::vector<float> exps(p->shape[n->op_axis]);
                float x;
                std::vector<size_t> coord(p->shape.size(), 0);
                std::vector<float> m = p->max(n->op_axis);
                size_t y = 0;
                while (true) {
                    float cs = 0;
                    size_t pi = p->offset, tai = t->offset;
                    float ng = n->grad_storage->data[n->offset + y];

                    for (int d = 0; d < p->shape.size(); ++d) {
                        pi += coord[d] * p->strides[d];
                        tai += coord[d] * t->strides[d];
                    }

                    for (size_t i = 0; i < p->shape[n->op_axis]; ++i) {
                        x = std::exp(p->storage->data[pi] - m[y]);
                        cs += x;
                        exps[i] = x;
                        pi += p->strides[n->op_axis];
                    }

                    pi -= p->shape[n->op_axis] * p->strides[n->op_axis];

                    for (size_t i = 0; i < p->shape[n->op_axis]; ++i) {
                        p->grad_storage->data[pi] += ng * (exps[i] / cs - t->storage->data[tai]);
                        pi += p->strides[n->op_axis];
                        tai += t->strides[n->op_axis];
                    }

                    bool completed = true;

                    for (int d = p->shape.size() - 1; d >= 0; --d) {
                        if (d == n->op_axis) continue;

                        ++coord[d];

                        if (coord[d] < p->shape[d]) {
                            completed = false;
                            break;
                        }

                        coord[d] = 0;
                    }

                    if (completed) break;

                    ++y;
                }
            }
        }
    }

    void backward_sparse_sce(std::shared_ptr<TensorImp> n) {
        auto& p = n->parents[0];
        auto& q = n->parents[1];

        if (p->requires_grad) {
            const size_t axis = n->op_axis;
            std::vector<float> m = p->max(axis);
            std::vector<size_t> coord(p->shape.size(), 0);
            size_t y = 0;

            while (true) {
                float cs = 0;
                size_t ni = n->offset;
                size_t pi = p->offset;
                size_t qi = q->offset;
                
                for (size_t d = 0; d < p->shape.size(); ++d) {
                    pi += coord[d] * p->strides[d];
                }
                for (size_t d = 0; d < q->shape.size(); ++d) {
                    ni += coord[d < axis ? d : d + 1] * n->strides[d];
                    qi += coord[d < axis ? d : d + 1] * q->strides[d];
                }

                for (size_t d = 0; d < p->shape[axis]; ++d) {
                    cs += std::exp(p->storage->data[pi] - m[y]);
                    pi += p->strides[axis];
                }

                pi -= p->shape[axis] * p->strides[axis];

                for (size_t d = 0; d < p->shape[axis]; ++d) {
                    p->grad_storage->data[pi] += n->grad_storage->data[ni] * (std::exp(p->storage->data[pi] - m[y]) / cs - (d == static_cast<size_t>(q->storage->data[qi]) ? 1.0f : 0.0f));
                    pi += p->strides[axis];
                }

                bool completed = true;

                for (int d = p->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < p->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;

                ++y;
            }
        }
    }

    Tensor clone_metadata() const {
        Tensor out;
        out.imp = std::make_shared<TensorImp>(*this->imp);
        
        return out;
    }

public:
    std::shared_ptr<TensorImp> imp;

    std::vector<float>& get_data() const {
        return this->imp->storage->data;
    }
    
    std::vector<float>& get_grads() const {
        if (this->imp->requires_grad) return this->imp->grad_storage->data;
        else throw std::invalid_argument("This Tensor does not have gradients");
    }

    std::vector<size_t>& get_shape() const {
        return this->imp->shape;
    }

    size_t get_length() {
        return this->imp->length;
    }

    void set_item(size_t index, float value) {
        this->imp->storage->data[index] = value;
    }

    void set_data(const std::vector<float>& values) {
        if (values.size() != imp->length) {
            throw std::invalid_argument("Data size does not match tensor length");
        }

        std::vector<size_t> coord(imp->shape.size(), 0);
        size_t index = imp->offset;

        for (size_t n = 0; n < imp->length; ++n) {
            imp->storage->data[index] = values[n];

            for (int d = static_cast<int>(imp->shape.size()) - 1;
                d >= 0; --d) {
                ++coord[d];
                index += imp->strides[d];

                if (coord[d] < imp->shape[d]) {
                    break;
                }

                coord[d] = 0;
                index -= imp->shape[d] * imp->strides[d];
            }
        }
    }

    Tensor() = default;

    Tensor(std::vector<size_t> shape, const bool requires_grad = false) : imp(std::make_shared<TensorImp>()) {
        imp->shape = shape;
        imp->requires_grad = requires_grad;
        imp->length = 1;
        for (size_t x : this->imp->shape) imp->length *= x;
        imp->strides.resize(shape.size());
        imp->storage->data.resize(imp->length, 0);
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            imp->strides[i] = stride;
            stride *= imp->shape[i];
        }
        if (imp->requires_grad) {
            imp->grad_storage = std::make_shared<Storage>();
            imp->grad_storage->data.resize(imp->length, 0);
        }
    }

    Tensor(std::vector<size_t> shape, const std::vector<float>& data, const bool requires_grad = false) : imp(std::make_shared<TensorImp>()) {
        imp->shape = shape;
        imp->requires_grad = requires_grad;
        imp->length = 1;
        for (size_t x : this->imp->shape) imp->length *= x;
        if (data.size() != imp->length) {
            throw std::invalid_argument("Incorrect length of data");
        }

        imp->storage->data = data;

        imp->strides.resize(imp->shape.size());
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            imp->strides[i] = stride;
            stride *= imp->shape[i];
        }

        if (imp->requires_grad) {
            imp->grad_storage = std::make_shared<Storage>();
            imp->grad_storage->data.resize(imp->length, 0);
        }
    }

    Tensor(const TensorImp& x) : imp(std::make_shared<TensorImp>(x)) {}

    bool is_contiguous() const {
        size_t expected = 1;
        for (int i = static_cast<int>(this->imp->shape.size()) - 1; i >= 0; --i) {
            if (imp->strides[i] != expected) return false;
            expected *= imp->shape[i];
        }
        return true;
    }

    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t new_length = 1;
        for (auto x : new_shape) new_length *= x;
        if (new_length != imp->length) {
            throw std::invalid_argument("Invalid reshape dimensions");
        }
        if (!this->is_contiguous()) {
            throw std::invalid_argument("Tensor must be contiguous for reshape");
        }

        Tensor out = clone_metadata();

        out.imp->shape = new_shape;
        out.imp->strides.resize(new_shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
            out.imp->strides[i] = stride;
            stride *= new_shape[i];
        }

        out.imp->op_type = Operation::RESHAPE;
        out.imp->parents = {this->imp};

        if (out.imp->requires_grad) {
            out.imp->grad_storage = std::make_shared<Storage>(out.imp->length, 0.0f);
        }

        return out;
    }

    Tensor view(const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, size_t new_offset) const {
        Tensor out = clone_metadata();

        out.imp->shape = new_shape;
        out.imp->strides = new_strides;
        out.imp->offset = new_offset;
        
        return out;
    }

    Tensor flatten(const int dim) const {
        if (dim < 0 || dim >= this->imp->shape.size()) {
            throw std::invalid_argument("Invalid dimension for flatten");
        }
        size_t c = 1;
        for (int i = dim; i < this->imp->shape.size(); ++i) c *= this->imp->shape[i];
        std::vector<size_t> out_shape(dim + 1);
        for (int i = 0; i < dim; ++i) out_shape[i] = this->imp->shape[i];
        out_shape[dim] = c;

        return this->reshape(out_shape);
    }

    Tensor transpose(const int a, const int b) const {
        if (a < 0 || b < 0 || a >= this->imp->shape.size() || b >= this->imp->shape.size() || a == b) {
            throw std::out_of_range("Invalid transpose dimensions");
        }

        Tensor out = clone_metadata();

        out.imp->op_type = Operation::TRANSPOSE;
        out.imp->parents = {this->imp};
        out.imp->tra = a;
        out.imp->trb = b;

        std::swap(out.imp->shape[a], out.imp->shape[b]);
        std::swap(out.imp->strides[a], out.imp->strides[b]);

        if (out.imp->requires_grad) {
            out.imp->grad_storage = std::make_shared<Storage>(out.imp->length, 0);
        }

        return out;
    }

    Tensor slice(const std::vector<Slice>& slices) const {
        if (slices.size() != this->imp->shape.size()) {
            throw std::invalid_argument("Number of slices must equal rank");
        }

        Tensor out = clone_metadata();
        out.imp->length = 1;
        out.imp->buffer = std::make_shared<std::vector<size_t>>(this->imp->shape.size(), 0);
        out.imp->buffer2 = std::make_shared<std::vector<size_t>>(this->imp->shape.size(), 1);

        for (int d = 0; d < this->imp->shape.size(); ++d) {
            auto& s = slices[d];
            
            if (s.step == 0) {
                throw std::invalid_argument("Invalid step value");
            }

            if (s.start > s.stop || s.stop > this->imp->shape[d]) {
                throw std::invalid_argument("Invalid slice bounds");
            }

            size_t len = (s.stop - s.start + s.step - 1) / s.step;

            out.imp->shape[d] = len;
            out.imp->strides[d] = this->imp->strides[d] * s.step;
            out.imp->length *= len;
            out.imp->offset += s.start * this->imp->strides[d];
            (*out.imp->buffer)[d] = s.start;
            (*out.imp->buffer2)[d] = s.step;
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::SLICE;
            out.imp->grad_storage = std::make_shared<Storage>(this->imp->length, 0.0f);
            out.imp->parents = {this->imp};
        }

        return out;
    }

    float& at(std::initializer_list<size_t> coordinates) const {
        size_t index = 0;
        int i = 0;
        for (int x : coordinates) {
            index += x * this->imp->strides[i];
            ++i;
        }
        index += this->imp->offset;

        return this->imp->storage->data[index];
    }

    Tensor max(const int axis = -1) const {
        if (axis < -1 || axis >= this->imp->shape.size()) {
            throw std::invalid_argument("Invalid axis for max()");
        }

        if (axis == -1 && this->is_contiguous()) {
            float xm = ninf;
            for (size_t i = 0; i < this->imp->length; ++i) {
                if (this->imp->storage->data[this->imp->offset + i] > xm) xm = this->imp->storage->data[this->imp->offset + i];
            }
            
            Tensor out(std::vector<size_t>{1}, std::vector<float>{xm}, this->imp->requires_grad);

            return out;
        }
        else if (axis == -1) {
            float xm = ninf;
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            size_t ti = this->imp->offset;
            for (size_t n = 0; n < this->imp->length; ++n) {
                if (this->imp->storage->data[ti] > xm) xm = this->imp->storage->data[ti];

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->imp->strides[d];

                    if (coord[d] < this->imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->imp->shape[d] * this->imp->strides[d];
                }
            }

            Tensor out(std::vector<size_t>{1}, std::vector<float>{xm}, this->imp->requires_grad);

            return out;
        }
        else {
            std::vector<float> outdata(this->imp->length / this->imp->shape[axis]);
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            int x = 0;

            while (true) {
                size_t index = this->imp->offset;
                float xm = ninf;

                for (int d = 0; d < this->imp->shape.size(); ++d) {
                    index += coord[d] * this->imp->strides[d];
                }

                for (size_t i = 0; i < this->imp->shape[axis]; ++i) {
                    if (this->imp->storage->data[index] > xm) {
                        xm = this->imp->storage->data[index];
                    }
                    index += this->imp->strides[axis];
                }

                outdata[x] = xm;

                bool completed = true;

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < this->imp->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;

                ++x;
            }

            std::vector<size_t> shape(this->imp->shape.size() - 1, 0);
            for (int n = 0; n < this->imp->shape.size(); ++n) {
                if (n < axis) shape[n] = this->imp->shape[n];
                else if (n > axis) shape[n - 1] = this->imp->shape[n];
            }

            Tensor out(shape, outdata, this->imp->requires_grad);

            return out;
        }
    }

    Tensor argmax(const int axis = -1) const {
        if (axis < -1 || axis >= this->imp->shape.size()) {
            throw std::invalid_argument("Invalid axis for argmax()");
        }

        if (axis == -1 && this->is_contiguous()) {
            float xm = ninf;
            size_t mi = 0;
            for (size_t i = 0; i < this->imp->length; ++i) {
                if (this->imp->storage->data[this->imp->offset + i] > xm) {
                    xm = this->imp->storage->data[this->imp->offset + i];
                    mi = i;
                }
            }
            
            Tensor out(std::vector<size_t>{1}, std::vector<float>{(float)mi}, this->imp->requires_grad);

            return out;
        }
        else if (axis == -1) {
            float xm = ninf;
            size_t mi = 0;
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            size_t ti = this->imp->offset;
            for (size_t n = 0; n < this->imp->length; ++n) {
                if (this->imp->storage->data[ti] > xm) {
                    xm = this->imp->storage->data[ti];
                    mi = ti;
                }

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->imp->strides[d];

                    if (coord[d] < this->imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->imp->shape[d] * this->imp->strides[d];
                }
            }

            Tensor out(std::vector<size_t>{1}, std::vector<float>{(float)mi}, this->imp->requires_grad);

            return out;
        }
        else {
            std::vector<float> outdata(this->imp->length / this->imp->shape[axis]);
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            int x = 0;

            while (true) {
                size_t index = this->imp->offset;
                float xm = ninf;
                size_t mi = 0;

                for (int d = 0; d < this->imp->shape.size(); ++d) {
                    index += coord[d] * this->imp->strides[d];
                }

                for (size_t i = 0; i < this->imp->shape[axis]; ++i) {
                    if (this->imp->storage->data[index] > xm) {
                        xm = this->imp->storage->data[index];
                        mi = i;
                    }
                    index += this->imp->strides[axis];
                }

                outdata[x] = (float)mi;

                bool completed = true;

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < this->imp->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;

                ++x;
            }

            std::vector<size_t> shape(this->imp->shape.size() - 1, 0);
            for (int n = 0; n < this->imp->shape.size(); ++n) {
                if (n < axis) shape[n] = this->imp->shape[n];
                else if (n > axis) shape[n - 1] = this->imp->shape[n];
            }

            Tensor out(shape, outdata, this->imp->requires_grad);

            return out;
        }
    }

    Tensor operator+(const Tensor& other) const {
        if (this->imp->shape.size() == 2 && this->imp->shape[0] == 128 && this->imp->shape[1] == 256 && ((other.imp->shape[0] == 256 && other.imp->shape.size() == 1) || (other.imp->shape.size() == 2 && other.imp->shape[0] == 1 && other.imp->shape[1] == 256)) && this->is_contiguous() && other.is_contiguous()) {
            return this->add_kernel_128x256_256_2d_contiguous(other);
        }
        size_t ndim = std::max(this->imp->shape.size(), other.imp->shape.size());

        std::vector<size_t> ashape(ndim, 1);
        std::vector<size_t> bshape(ndim, 1);

        std::vector<size_t> astrides(ndim, 0);
        std::vector<size_t> bstrides(ndim, 0);

        size_t adiff = ndim - this->imp->shape.size();
        size_t bdiff = ndim - other.imp->shape.size();

        for (size_t d = 0; d < this->imp->shape.size(); ++d) {
            ashape[adiff + d] = this->imp->shape[d];
            astrides[adiff + d] = this->imp->strides[d];
        }

        for (size_t d = 0; d < other.imp->shape.size(); ++d) {
            bshape[bdiff + d] = other.imp->shape[d];
            bstrides[bdiff + d] = other.imp->strides[d];
        }

        std::vector<size_t> out_shape(ndim);

        for (size_t d = 0; d < ndim; ++d) {
            if (ashape[d] != bshape[d] && ashape[d] != 1 && bshape[d] != 1) {
                throw std::invalid_argument("Invalid dimensions for addition");
            }

            out_shape[d] = std::max(ashape[d], bshape[d]);

            if (ashape[d] == 1) astrides[d] = 0;
            if (bshape[d] == 1) bstrides[d] = 0;
        }

        Tensor out(out_shape, this->imp->requires_grad || other.imp->requires_grad);

        std::vector<size_t> coord(ndim, 0);

        size_t ai = this->imp->offset;
        size_t bi = other.imp->offset;
        size_t oi = out.imp->offset;

        for (size_t n = 0; n < out.imp->length; ++n) {

            out.imp->storage->data[oi] = this->imp->storage->data[ai] + other.imp->storage->data[bi];

            for (int d = ndim - 1; d >= 0; --d) {

                ++coord[d];

                ai += astrides[d];
                bi += bstrides[d];
                oi += out.imp->strides[d];

                if (coord[d] < out_shape[d])
                    break;

                coord[d] = 0;

                ai -= out_shape[d] * astrides[d];
                bi -= out_shape[d] * bstrides[d];
                oi -= out_shape[d] * out.imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::ADD;
            out.imp->parents = {this->imp, other.imp};
        }

        return out;
    }

    Tensor operator+(const float other) const {
        std::vector<size_t> coord(this->imp->shape.size(), 0);

        Tensor out(this->imp->shape, this->imp->requires_grad);

        size_t oi = 0;
        size_t ti = this->imp->offset;

        for (size_t n = 0; n < this->imp->length; ++n) {
            out.imp->storage->data[oi] = this->imp->storage->data[ti] + other;

            for (int d = 0; d < this->imp->shape.size(); ++d) {
                ++coord[d];

                oi += out.imp->strides[d];
                ti += this->imp->strides[d];

                if (coord[d] < this->imp->shape[d]) break;

                coord[d] = 0;

                oi -= out.imp->shape[d] * out.imp->strides[d];
                ti -= this->imp->shape[d] * this->imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::ADD;
            out.imp->op_axis = 0;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor operator-(const Tensor& other) const {
        size_t ndim = std::max(this->imp->shape.size(), other.imp->shape.size());

        std::vector<size_t> ashape(ndim, 1);
        std::vector<size_t> bshape(ndim, 1);

        std::vector<size_t> astrides(ndim, 0);
        std::vector<size_t> bstrides(ndim, 0);

        size_t adiff = ndim - this->imp->shape.size();
        size_t bdiff = ndim - other.imp->shape.size();

        for (size_t d = 0; d < this->imp->shape.size(); ++d) {
            ashape[adiff + d] = this->imp->shape[d];
            astrides[adiff + d] = this->imp->strides[d];
        }

        for (size_t d = 0; d < other.imp->shape.size(); ++d) {
            bshape[bdiff + d] = other.imp->shape[d];
            bstrides[bdiff + d] = other.imp->strides[d];
        }

        std::vector<size_t> out_shape(ndim);

        for (size_t d = 0; d < ndim; ++d) {
            if (ashape[d] != bshape[d] && ashape[d] != 1 && bshape[d] != 1) {
                throw std::invalid_argument("Invalid dimensions for subtraction");
            }

            out_shape[d] = std::max(ashape[d], bshape[d]);

            if (ashape[d] == 1) astrides[d] = 0;
            if (bshape[d] == 1) bstrides[d] = 0;
        }

        Tensor out(out_shape, this->imp->requires_grad || other.imp->requires_grad);

        std::vector<size_t> coord(ndim, 0);

        size_t ai = this->imp->offset;
        size_t bi = other.imp->offset;
        size_t oi = out.imp->offset;

        for (size_t n = 0; n < out.imp->length; ++n) {

            out.imp->storage->data[oi] = this->imp->storage->data[ai] - other.imp->storage->data[bi];

            for (int d = ndim - 1; d >= 0; --d) {

                ++coord[d];

                ai += astrides[d];
                bi += bstrides[d];
                oi += out.imp->strides[d];

                if (coord[d] < out_shape[d])
                    break;

                coord[d] = 0;

                ai -= out_shape[d] * astrides[d];
                bi -= out_shape[d] * bstrides[d];
                oi -= out_shape[d] * out.imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::SUB;
            out.imp->parents = {this->imp, other.imp};
        }

        return out;
    }

    Tensor operator*(const Tensor& other) const {
        size_t ndim = std::max(this->imp->shape.size(), other.imp->shape.size());

        std::vector<size_t> ashape(ndim, 1);
        std::vector<size_t> bshape(ndim, 1);

        std::vector<size_t> astrides(ndim, 0);
        std::vector<size_t> bstrides(ndim, 0);

        size_t adiff = ndim - this->imp->shape.size();
        size_t bdiff = ndim - other.imp->shape.size();

        for (size_t d = 0; d < this->imp->shape.size(); ++d) {
            ashape[adiff + d] = this->imp->shape[d];
            astrides[adiff + d] = this->imp->strides[d];
        }

        for (size_t d = 0; d < other.imp->shape.size(); ++d) {
            bshape[bdiff + d] = other.imp->shape[d];
            bstrides[bdiff + d] = other.imp->strides[d];
        }

        std::vector<size_t> out_shape(ndim);

        for (size_t d = 0; d < ndim; ++d) {
            if (ashape[d] != bshape[d] && ashape[d] != 1 && bshape[d] != 1) {
                throw std::invalid_argument("Invalid dimensions for multiplication");
            }

            out_shape[d] = std::max(ashape[d], bshape[d]);

            if (ashape[d] == 1) astrides[d] = 0;
            if (bshape[d] == 1) bstrides[d] = 0;
        }

        Tensor out(out_shape, this->imp->requires_grad || other.imp->requires_grad);

        std::vector<size_t> coord(ndim, 0);

        size_t ai = this->imp->offset;
        size_t bi = other.imp->offset;
        size_t oi = out.imp->offset;

        for (size_t n = 0; n < out.imp->length; ++n) {

            out.imp->storage->data[oi] = this->imp->storage->data[ai] * other.imp->storage->data[bi];

            for (int d = ndim - 1; d >= 0; --d) {

                ++coord[d];

                ai += astrides[d];
                bi += bstrides[d];
                oi += out.imp->strides[d];

                if (coord[d] < out_shape[d])
                    break;

                coord[d] = 0;

                ai -= out_shape[d] * astrides[d];
                bi -= out_shape[d] * bstrides[d];
                oi -= out_shape[d] * out.imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::MUL;
            out.imp->parents = {this->imp, other.imp};
        }

        return out;
    }

    Tensor operator*(const float other) const {
        std::vector<size_t> coord(this->imp->shape.size(), 0);

        Tensor out(this->imp->shape, this->imp->requires_grad);

        size_t oi = 0;
        size_t ti = this->imp->offset;

        for (size_t n = 0; n < this->imp->length; ++n) {
            out.imp->storage->data[oi] = this->imp->storage->data[ti] * other;

            for (int d = 0; d < this->imp->shape.size(); ++d) {
                ++coord[d];

                oi += out.imp->strides[d];
                ti += this->imp->strides[d];

                if (coord[d] < this->imp->shape[d]) break;

                coord[d] = 0;

                oi -= out.imp->shape[d] * out.imp->strides[d];
                ti -= this->imp->shape[d] * this->imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::MUL;
            out.imp->op_axis = 0;
            out.imp->alpha = other;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor operator/(const Tensor& other) const {
        size_t ndim = std::max(this->imp->shape.size(), other.imp->shape.size());

        std::vector<size_t> ashape(ndim, 1);
        std::vector<size_t> bshape(ndim, 1);

        std::vector<size_t> astrides(ndim, 0);
        std::vector<size_t> bstrides(ndim, 0);

        size_t adiff = ndim - this->imp->shape.size();
        size_t bdiff = ndim - other.imp->shape.size();

        for (size_t d = 0; d < this->imp->shape.size(); ++d) {
            ashape[adiff + d] = this->imp->shape[d];
            astrides[adiff + d] = this->imp->strides[d];
        }

        for (size_t d = 0; d < other.imp->shape.size(); ++d) {
            bshape[bdiff + d] = other.imp->shape[d];
            bstrides[bdiff + d] = other.imp->strides[d];
        }

        std::vector<size_t> out_shape(ndim);

        for (size_t d = 0; d < ndim; ++d) {
            if (ashape[d] != bshape[d] && ashape[d] != 1 && bshape[d] != 1) {
                throw std::invalid_argument("Invalid dimensions for division");
            }

            out_shape[d] = std::max(ashape[d], bshape[d]);

            if (ashape[d] == 1) astrides[d] = 0;
            if (bshape[d] == 1) bstrides[d] = 0;
        }

        Tensor out(out_shape, this->imp->requires_grad || other.imp->requires_grad);

        std::vector<size_t> coord(ndim, 0);

        size_t ai = this->imp->offset;
        size_t bi = other.imp->offset;
        size_t oi = out.imp->offset;

        for (size_t n = 0; n < out.imp->length; ++n) {

            out.imp->storage->data[oi] = this->imp->storage->data[ai] / other.imp->storage->data[bi];

            for (int d = ndim - 1; d >= 0; --d) {

                ++coord[d];

                ai += astrides[d];
                bi += bstrides[d];
                oi += out.imp->strides[d];

                if (coord[d] < out_shape[d])
                    break;

                coord[d] = 0;

                ai -= out_shape[d] * astrides[d];
                bi -= out_shape[d] * bstrides[d];
                oi -= out_shape[d] * out.imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::DIV;
            out.imp->parents = {this->imp, other.imp};
        }

        return out;
    }

    Tensor operator/(const float other) const {
        std::vector<size_t> coord(this->imp->shape.size(), 0);

        Tensor out(this->imp->shape, this->imp->requires_grad);

        size_t oi = 0;
        size_t ti = this->imp->offset;

        for (size_t n = 0; n < this->imp->length; ++n) {
            out.imp->storage->data[oi] = this->imp->storage->data[ti] / other;

            for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                ++coord[d];

                oi += out.imp->strides[d];
                ti += this->imp->strides[d];

                if (coord[d] < this->imp->shape[d]) break;

                coord[d] = 0;

                oi -= out.imp->shape[d] * out.imp->strides[d];
                ti -= this->imp->shape[d] * this->imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::DIV;
            out.imp->op_axis = 0;
            out.imp->alpha = other;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor pow(const float other) const {
        std::vector<size_t> coord(this->imp->shape.size(), 0);

        Tensor out(this->imp->shape, this->imp->requires_grad);

        size_t oi = 0;
        size_t ti = this->imp->offset;

        for (size_t n = 0; n < this->imp->length; ++n) {
            out.imp->storage->data[oi] = std::pow(this->imp->storage->data[ti], other);

            for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                ++coord[d];

                oi += out.imp->strides[d];
                ti += this->imp->strides[d];

                if (coord[d] < this->imp->shape[d]) break;

                coord[d] = 0;

                oi -= out.imp->shape[d] * out.imp->strides[d];
                ti -= this->imp->shape[d] * this->imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::POW;
            out.imp->op_axis = 0;
            out.imp->alpha = other;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor sqrt() const {
        return this->pow(0.5f);
    }

    Tensor matmul(const Tensor& other) const {
        if (this->imp->shape.size() != other.imp->shape.size() || this->imp->shape[this->imp->shape.size() - 1] != other.imp->shape[this->imp->shape.size() - 2]) {
            throw std::invalid_argument("Invalid dimensions for extended 2D Tensor multiplication");
        }
        for (int i = 0; i < this->imp->shape.size() - 2; ++i) {
            if (this->imp->shape[i] != other.imp->shape[i]) {
                throw std::invalid_argument("Invalid dimensions for extended 2D Tensor multiplication");
            }
        }

        std::vector<size_t> out_shape(this->imp->shape.size());
        for (int i = 0; i < this->imp->shape.size() - 1; ++i) out_shape[i] = this->imp->shape[i];
        out_shape[this->imp->shape.size() - 1] = other.imp->shape[this->imp->shape.size() - 1];

        Tensor out(out_shape, this->imp->requires_grad || other.imp->requires_grad);

        std::vector<size_t> coord(this->imp->shape.size(), 0);
        size_t ti = this->imp->offset;
        size_t oi = other.imp->offset;
        size_t outi = out.imp->offset;

        const bool a_contiguous = this->is_contiguous();
        const bool b_contiguous = other.is_contiguous();
        const bool x = a_contiguous && b_contiguous;

        for (size_t n = 0; n < out.imp->length; n += out.imp->shape[this->imp->shape.size() - 1] * out.imp->shape[this->imp->shape.size() - 2]) {
            if (x && this->imp->shape[this->imp->shape.size() - 2] == 128 && this->imp->shape[this->imp->shape.size() - 1] == 784 && other.imp->shape[other.imp->shape.size() - 2] == 784 && other.imp->shape[other.imp->shape.size() - 1] == 256) {
                matmul_kernel_strided_128x784x256_contiguous(this->imp->storage->data.data(), other.imp->storage->data.data(), out.imp->storage->data.data(), ti, oi, outi);
            }
            else if (x && this->imp->shape[this->imp->shape.size() - 2] == 128 && this->imp->shape[this->imp->shape.size() - 1] == 256 && other.imp->shape[other.imp->shape.size() - 2] == 256 && other.imp->shape[other.imp->shape.size() - 1] == 128) {
                matmul_kernel_strided_128x256x128_contiguous(this->imp->storage->data.data(), other.imp->storage->data.data(), out.imp->storage->data.data(), ti, oi, outi);
            }
            else if (x && this->imp->shape[this->imp->shape.size() - 2] == 128 && this->imp->shape[this->imp->shape.size() - 1] == 128 && other.imp->shape[other.imp->shape.size() - 2] == 128 && other.imp->shape[other.imp->shape.size() - 1] == 10) {
                matmul_kernel_strided_128x128x10_contiguous(this->imp->storage->data.data(), other.imp->storage->data.data(), out.imp->storage->data.data(), ti, oi, outi);
            }
            else if (x) {
                matmul_kernel_add_offset(this->imp->storage->data.data(), other.imp->storage->data.data(), out.imp->storage->data.data(), ti, oi, outi, this->imp->shape[this->imp->shape.size() - 2], this->imp->shape[this->imp->shape.size() - 1], other.imp->shape[this->imp->shape.size() - 2], other.imp->shape[this->imp->shape.size() - 1]);
            }
            else {
                matmul_kernel_strided(this->imp->storage->data.data(), other.imp->storage->data.data(), out.imp->storage->data.data(), ti, oi, outi, this->imp->shape[this->imp->shape.size() - 2], this->imp->shape[this->imp->shape.size() - 1], other.imp->shape[this->imp->shape.size() - 2], other.imp->shape[this->imp->shape.size() - 1], this->imp->strides[this->imp->shape.size() - 2], this->imp->strides[this->imp->shape.size() - 1], other.imp->strides[this->imp->shape.size() - 2], other.imp->strides[this->imp->shape.size() - 1], out.imp->strides[this->imp->shape.size() - 2], out.imp->strides[this->imp->shape.size() - 1]);
            }

            for (int d = static_cast<int>(this->imp->shape.size()) - 3; d >= 0; --d) {
                ++coord[d];

                ti += this->imp->strides[d];
                oi += other.imp->strides[d];
                outi += out.imp->strides[d];

                if (coord[d] < out.imp->shape[d]) break;

                coord[d] = 0;

                ti -= out.imp->shape[d] * this->imp->strides[d];
                oi -= out.imp->shape[d] * other.imp->strides[d];
                outi -= out.imp->shape[d] * out.imp->strides[d];
            }
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::MATMUL;
            out.imp->parents = {this->imp, other.imp};
        }

        return out;
    }

    Tensor conv2d(const Tensor& weights, const Tensor& biases, const size_t stride, const size_t padding) const {
        if (stride == 0) {
            throw std::invalid_argument("Stride must be positive");
        }
        if (this->imp->shape.size() != 4 || weights.imp->shape.size() != 4 || biases.imp->shape.size() != 1) {
            throw std::invalid_argument("Invalid shapes of inputs");
        }
        
        size_t kernel_size = weights.imp->shape[2];

        if (this->imp->shape[2] + 2 * padding < kernel_size || this->imp->shape[3] + 2 * padding < kernel_size) {
            throw std::invalid_argument("Kernel is larger than padded input");
        }

        const size_t cin = this->imp->shape[1];
        const size_t c_out = weights.imp->shape[0];
        const size_t h = this->imp->shape[2];
        const size_t w = this->imp->shape[3];
        const size_t k = weights.imp->shape[3];
        const size_t h_out = 1 + (h + 2 * padding - k) / stride;
        const size_t w_out = 1 + (w + 2 * padding - k) / stride;

        const bool z = this->imp->requires_grad || weights.imp->requires_grad || biases.imp->requires_grad;

        Tensor out(std::vector<size_t>{this->imp->shape[0], c_out, h_out, w_out}, z);
        
        out.imp->convmd->col.resize(this->imp->shape[0] * cin * k * k * h_out * w_out);

        #pragma omp parallel for schedule(static)
        for (size_t n = 0; n < this->imp->shape[0]; ++n) {
            im2col_direct(this->imp->storage->data.data() + n * cin * h * w, out.imp->convmd->col.data() + n * cin * k * k * h_out * w_out, cin, h, w, k, stride, padding);
            matmul_kernel_add_offset(weights.imp->storage->data.data(), out.imp->convmd->col.data() + n * cin * k * k * h_out * w_out, out.imp->storage->data.data() + n * c_out * h_out * w_out, weights.imp->offset, 0, 0, c_out, cin * k * k, cin * k * k, h_out * w_out);
            for (size_t c = 0; c < c_out; ++c) {
                for (size_t i = 0; i < h_out; ++i) {
                    for (size_t j = 0; j < w_out; ++j) {
                        out.imp->storage->data[n * c_out * h_out * w_out + c * h_out * w_out + i * w_out + j] += biases.imp->storage->data[c];
                    }
                }
            }
        }

        if (z) {
            out.imp->op_type = Operation::CONV2D;
            out.imp->convmd->stride = stride;
            out.imp->convmd->padding = padding;
            out.imp->convmd->kernel_size = kernel_size;
            out.imp->parents = {this->imp, weights.imp, biases.imp};
        }

        return out;
    }

    Tensor maxpool2d(const size_t kernel_size, const size_t stride, const size_t padding) const {
        if (stride == 0) {
            throw std::invalid_argument("Stride must be positive");
        }
        if (this->imp->shape.size() != 4) {
            throw std::invalid_argument("Invalid shapes of inputs");
        }
        if (this->imp->shape[2] + 2 * padding < kernel_size || this->imp->shape[3] + 2 * padding < kernel_size) {
            throw std::invalid_argument("Kernel is larger than padded input");
        }

        const size_t b = this->imp->shape[0];
        const size_t c = this->imp->shape[1];
        const size_t h = this->imp->shape[2];
        const size_t w = this->imp->shape[3];
        const size_t h_out = 1 + (h + 2 * padding - kernel_size) / stride;
        const size_t w_out = 1 + (w + 2 * padding - kernel_size) / stride;

        std::vector<float> outdata(b * c * h_out * w_out, 0.0f);
        std::vector<size_t> buffer(b * c * h_out * w_out, 0);

        #pragma omp parallel for schedule(static)
        for (size_t n = 0; n < b; ++n) {
            for (size_t o = 0; o < c; ++o) {
                for (size_t oh = 0; oh < h_out; ++oh) {
                    for (size_t ow = 0; ow < w_out; ++ow) {
                        float m = ninf;
                        size_t mi = 0;

                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            const int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                const int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);
                                if (ih >= 0 && ih < static_cast<int>(h) && iw >= 0 && iw < static_cast<int>(w)) {
                                    size_t i = ((n * c + o) * h + ih) * w + iw;
                                    float v = this->imp->storage->data[i];

                                    if (v > m) {
                                        m = v;
                                        mi = i;
                                    }
                                }
                            }
                        }

                        size_t oi = ((n * c + o) * h_out + oh) * w_out + ow;

                        outdata[oi] = m;
                        buffer[oi] = mi;
                    }
                }
            }
        }

        Tensor out(std::vector<size_t>{b, c, h_out, w_out}, outdata, this->imp->requires_grad);

        *(out.imp->buffer) = std::move(buffer);

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::MAXPOOL2D;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor sum(const int axis = -1) const {
        if (axis < -1 || axis >= static_cast<int>(this->imp->shape.size())) {
            throw std::invalid_argument("Invalid axis");
        }
        
        if (axis == -1) {
            float cs = 0;

            Tensor out(std::vector<size_t>{1}, this->imp->requires_grad);

            size_t ti = this->imp->offset;
            std::vector<size_t> coord(this->imp->shape.size(), 0);

            for (size_t n = 0; n < this->imp->length; ++n) {
                cs += this->imp->storage->data[ti];

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->imp->strides[d];

                    if (coord[d] < this->imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->imp->shape[d] * this->imp->strides[d];
                }
            }

            out.imp->storage->data[0] = cs;

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::SUM;
                out.imp->parents = {this->imp};
            }

            return out;
        }
        else {
            std::vector<size_t> os = this->imp->shape;
            os[axis] = 1;

            Tensor out(os, this->imp->requires_grad);

            std::vector<size_t> coord(this->imp->shape.size(), 0);

            while (true) {
                float cs = 0;
                size_t ti = this->imp->offset;
                size_t oi = 0;

                for (int d = 0; d < this->imp->shape.size(); ++d) {
                    ti += coord[d] * this->imp->strides[d];
                    oi += coord[d] * out.imp->strides[d];
                }

                for (size_t n = 0; n < this->imp->shape[axis]; ++n) {
                    cs += this->imp->storage->data[ti];
                    ti += this->imp->strides[axis];
                }

                out.imp->storage->data[oi] = cs;

                bool completed = true;

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < this->imp->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;
            }

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::SUM;
                out.imp->op_axis = axis;
                out.imp->parents = {this->imp};
            }

            return out;
        }
    }

    Tensor mean(const int axis = -1) const {
        if (axis < -1 || axis >= static_cast<int>(this->imp->shape.size())) {
            throw std::invalid_argument("Invalid axis");
        }

        if (axis == -1) {
            float cs = 0;

            Tensor out(std::vector<size_t>{1}, this->imp->requires_grad);

            size_t ti = this->imp->offset;
            std::vector<size_t> coord(this->imp->shape.size(), 0);

            for (size_t n = 0; n < this->imp->length; ++n) {
                cs += this->imp->storage->data[ti];

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->imp->strides[d];

                    if (coord[d] < this->imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->imp->shape[d] * this->imp->strides[d];
                }
            }

            out.imp->storage->data[0] = cs / this->imp->length;

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::MEAN;
                out.imp->parents = {this->imp};
            }

            return out;
        }
        else {
            std::vector<size_t> os = this->imp->shape;
            os[axis] = 1;

            Tensor out(os, this->imp->requires_grad);

            std::vector<size_t> coord(this->imp->shape.size(), 0);

            while (true) {
                float cs = 0;
                size_t ti = this->imp->offset;
                size_t oi = 0;

                for (int d = 0; d < this->imp->shape.size(); ++d) {
                    ti += coord[d] * this->imp->strides[d];
                    oi += coord[d] * out.imp->strides[d];
                }

                for (size_t n = 0; n < this->imp->shape[axis]; ++n) {
                    cs += this->imp->storage->data[ti];
                    ti += this->imp->strides[axis];
                }

                out.imp->storage->data[oi] = cs / this->imp->shape[axis];

                bool completed = true;

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < this->imp->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;
            }

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::MEAN;
                out.imp->op_axis = axis;
                out.imp->parents = {this->imp};
            }

            return out;
        }
    }

    Tensor variance(const int axis = -1) const {
        if (axis < -1 || axis >= static_cast<int>(this->imp->shape.size())) {
            throw std::invalid_argument("Invalid axis");
        }

        if (axis == -1) {
            Tensor out(std::vector<size_t>{1}, this->imp->requires_grad);

            float m = this->mean(-1).imp->storage->data[0];
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            size_t ti = this->imp->offset;
            float cs = 0;

            for (size_t n = 0; n < this->imp->length; ++n) {
                cs += (this->imp->storage->data[ti] - m) * (this->imp->storage->data[ti] - m);

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->imp->strides[d];

                    if (coord[d] < this->imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->imp->shape[d] * this->imp->strides[d];
                }
            }

            out.imp->storage->data[0] = cs / this->imp->length;

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::VARIANCE;
                *(out.imp->bufferf) = {m};
                out.imp->parents = {this->imp};
            }

            return out;
        }
        else {
            std::vector<size_t> os = this->imp->shape;
            os[axis] = 1;

            Tensor out(os, this->imp->requires_grad);

            Tensor m = this->mean(axis);

            std::vector<size_t> coord(this->imp->shape.size(), 0);

            while (true) {
                float cs = 0;
                size_t ti = this->imp->offset;
                size_t mi = 0;
                size_t oi = 0;

                for (int d = 0; d < this->imp->shape.size(); ++d) {
                    ti += coord[d] * this->imp->strides[d];
                    oi += coord[d] * out.imp->strides[d];
                    mi += (d == axis ? 0 : coord[d]) * m.imp->strides[d];
                }

                for (size_t n = 0; n < this->imp->shape[axis]; ++n) {
                    cs += (this->imp->storage->data[ti] - m.imp->storage->data[mi]) * (this->imp->storage->data[ti] - m.imp->storage->data[mi]);
                    ti += this->imp->strides[axis];
                }

                out.imp->storage->data[oi] = cs / this->imp->shape[axis];

                bool completed = true;

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < this->imp->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;
            }

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::VARIANCE;
                *(out.imp->bufferf) = m.imp->storage->data;
                *(out.imp->buffer) = m.imp->strides;
                out.imp->op_axis = axis;
                out.imp->parents = {this->imp};
            }

            return out;
        }
    }

    Tensor relu() const {
        Tensor out(this->imp->shape, this->imp->storage->data, this->imp->requires_grad);

        for (auto& x : out.imp->storage->data) {
            if (x < 0) x = 0;
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::RELU;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor leaky_relu(float alpha = 0.01) const {
        Tensor out(this->imp->shape, this->imp->storage->data, this->imp->requires_grad);

        for (auto& x : out.imp->storage->data) {
            if (x < 0) x = alpha * x;
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::LEAKYRELU;
            out.imp->alpha = alpha;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor gelu() const {
        Tensor out(this->imp->shape, this->imp->storage->data, this->imp->requires_grad);

        for (auto& x : out.imp->storage->data) {
            x = 0.5f * x * (1 + std::tanh(std::sqrt(2.0f / std::numbers::pi) * (x + 0.044715 * x * x * x)));
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::GELU;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor sigmoid() const {
        std::vector<float> outdata(this->imp->length);
        std::vector<size_t> coord(this->imp->shape.size(), 0);
        size_t i = 0;

        for (size_t n = 0; n < this->imp->length; ++n) {
            float x = this->imp->storage->data[i];
            outdata[n] = ((x > 0) ? (1.0f / (1.0f + std::exp(-x))) : (std::exp(x) / (1.0f + std::exp(x))));

            for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                ++coord[d];

                i += this->imp->strides[d];

                if (coord[d] < this->imp->shape[d]) break;

                coord[d] = 0;

                i -= this->imp->shape[d] * this->imp->strides[d];
            }
        }

        Tensor out(this->imp->shape, outdata, this->imp->requires_grad);

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::SIGMOID;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    static Tensor causal_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask) {
        if (q.imp->shape[q.imp->shape.size() - 1] != k.imp->shape[k.imp->shape.size() - 1] || q.imp->shape[q.imp->shape.size() - 2] != mask.imp->shape[mask.imp->shape.size() - 2] || k.imp->shape[k.imp->shape.size() - 2] != mask.imp->shape[mask.imp->shape.size() - 1] || k.imp->shape[k.imp->shape.size() - 2] != v.imp->shape[v.imp->shape.size() - 2]) {
            throw std::invalid_argument("Invalid dimensions");
        }

        return ((q.matmul(k.transpose(k.imp->shape.size() - 2, k.imp->shape.size() - 1)) * (1.0 / std::sqrt(static_cast<float>(q.imp->shape[q.imp->shape.size() - 1]))) + mask).softmax(q.imp->shape.size() - 1).matmul(v));
    }

    static Tensor concat(const std::vector<Tensor>& tensors, const size_t axis) {
        if (tensors.empty()) {
            throw std::invalid_argument("Tensors must be provided to concatenate");
        }
        if (axis >= tensors[0].imp->shape.size()) {
            throw std::invalid_argument("Invalid axis");
        }

        std::vector<size_t> os = tensors[0].imp->shape;
        os[axis] = 0;
        bool rg = false;

        for (auto& t : tensors) {
            if (t.imp->shape.size() != tensors[0].imp->shape.size()) {
                throw std::invalid_argument("Tensors must have same rank");
            }

            for (int d = 0; d < t.imp->shape.size(); ++d) {
                if (d != axis && t.imp->shape[d] != os[d]) {
                    throw std::invalid_argument("Tensors must have equal shapes in all dimensions other than axis");
                }
            }

            os[axis] += t.imp->shape[axis];
            rg |= t.imp->requires_grad;
        }

        Tensor out(os, rg);

        size_t o = 0;

        for (auto& t : tensors) {
            std::vector<size_t> coord(t.imp->shape.size(), 0);
            size_t ti = t.imp->offset;

            for (size_t n = 0; n < t.imp->length; ++n) {
                size_t j = 0;

                for (int d = 0; d < t.imp->shape.size(); ++d) {
                    j += (d == axis ? (coord[d] + o) : coord[d]) * out.imp->strides[d];
                }

                out.imp->storage->data[j] = t.imp->storage->data[ti];

                for (int d = t.imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += t.imp->strides[d];

                    if (coord[d] < t.imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= t.imp->shape[d] * t.imp->strides[d];
                }
            }

            o += t.imp->shape[axis];
        }

        if (rg) {
            out.imp->op_type = Operation::CONCAT;
            out.imp->op_axis = axis;
            out.imp->parents.reserve(tensors.size());
            for (auto& t : tensors) {
                out.imp->parents.push_back(t.imp);
            }
        }

        return out;
    }

    static Tensor embedding(const std::vector<size_t>& tokens, const Tensor& weights) {
        for (auto token : tokens) {
            if (token >= weights.imp->shape[0]) {
                throw std::invalid_argument("Token must be in vocabulary");
            }
        }

        if (weights.imp->shape.size() == 2) {
            Tensor out({tokens.size(), weights.imp->shape[1]}, weights.imp->requires_grad);

            for (size_t i = 0; i < tokens.size(); ++i) {
                size_t token = tokens[i];
                size_t x = token * weights.imp->strides[0] + weights.imp->offset;
                for (size_t j = 0; j < weights.imp->shape[1]; ++j) {
                    out.imp->storage->data[i * weights.imp->shape[1] + j] = weights.imp->storage->data[x + j * weights.imp->strides[1]];
                }
            }

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::EMBED;
                *(out.imp->buffer) = tokens;
                out.imp->parents = {weights.imp};
            }

            return out;
        }
        else {
            throw std::invalid_argument("Weights must be two dimensional");
        }
    }

    static Tensor embedding(const Tensor& tokens, const Tensor& weights) {
        if (weights.imp->shape.size() != 2) {
            throw std::invalid_argument("Weights must be two dimensional");
        }

        if (tokens.imp->shape.size() == 1) {
            std::vector<size_t> data(tokens.imp->length);
            for (size_t i = 0; i < tokens.imp->length; ++i) {
                data[i] = static_cast<size_t>(tokens.imp->storage->data[tokens.imp->offset + i]);
            }
            return embedding(data, weights);
        }
        else if (tokens.imp->shape.size() == 2) {
            Tensor out({tokens.imp->shape[0], tokens.imp->shape[1], weights.imp->shape[1]}, weights.imp->requires_grad);

            for (size_t b = 0; b < tokens.imp->shape[0]; ++b) {
                size_t z = tokens.imp->offset + b * tokens.imp->strides[0];
                for (size_t i = 0; i < tokens.imp->shape[1]; ++i) {
                    size_t token = tokens.imp->storage->data[z + i * tokens.imp->strides[1]];
                    size_t x = token * weights.imp->strides[0] + weights.imp->offset;
                    for (size_t j = 0; j < weights.imp->shape[1]; ++j) {
                        out.imp->storage->data[b * out.imp->strides[0] + i * weights.imp->shape[1] + j] = weights.imp->storage->data[x + j * weights.imp->strides[1]];
                    }
                }
            }

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::EMBED;
                out.imp->buffer->resize(tokens.imp->length);
                for (size_t b = 0; b < tokens.imp->shape[0]; ++b) {
                    for (size_t i = 0; i < tokens.imp->shape[1]; ++i) {
                        (*out.imp->buffer)[b * tokens.imp->shape[1] + i] = static_cast<size_t>(tokens.imp->storage->data[tokens.imp->offset + b * tokens.imp->strides[0] + i * tokens.imp->strides[1]]);
                    }
                }
                out.imp->parents = {weights.imp};
            }

            return out;
        }
        else {
            throw std::invalid_argument("Invalid dimensions of tokens");
        }
    }

    static Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& bias, const float epsilon = 1e-5) {
        if (x.imp->shape.size() != 3 || scale.imp->shape.size() != 1 || bias.imp->shape.size() != 1 || x.imp->shape[2] != scale.imp->shape[0] || x.imp->shape[2] != bias.imp->shape[0]) {
            throw std::invalid_argument("Invalid dimensions");
        }

        return ((x - x.mean(2)) / ((x.variance(2) + epsilon).sqrt())) * scale + bias;
    }

    Tensor softmax(const int axis = 1) const {
        if (axis < 0 || axis >= this->imp->shape.size()) {
            throw std::invalid_argument("Invalid axis");
        }

        std::vector<float> m = this->max_data(axis);

        Tensor out(this->imp->shape, this->imp->requires_grad);

        std::vector<size_t> coord(this->imp->shape.size(), 0);
        size_t y = 0;

        while (true) {
            float cs = 0;
            size_t ti = this->imp->offset;
            size_t oi = 0;

            for (int d = 0; d < this->imp->shape.size(); ++d) {
                ti += coord[d] * this->imp->strides[d];
                oi += coord[d] * out.imp->strides[d];
            }

            for (size_t n = 0; n < this->imp->shape[axis]; ++n) {
                cs += std::exp(this->imp->storage->data[ti] - m[y]);
                ti += this->imp->strides[axis];
            }

            ti -= this->imp->shape[axis] * this->imp->strides[axis];

            for (size_t n = 0; n < this->imp->shape[axis]; ++n) {
                out.imp->storage->data[oi + n * out.imp->strides[axis]] = std::exp(this->imp->storage->data[ti] - m[y]) / cs;
                ti += this->imp->strides[axis];
            }

            bool completed = true;

            for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                if (d == axis) continue;

                ++coord[d];

                if (coord[d] < this->imp->shape[d]) {
                    completed = false;
                    break;
                }

                coord[d] = 0;
            }

            if (completed) break;

            ++y;
        }

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::SOFTMAX;
            out.imp->op_axis = axis;
            out.imp->parents = {this->imp};
        }

        return out;
    }

    Tensor softmax_cross_entropy(const Tensor& target, const int axis = 1, const int reduction = 0) const {
        if (this->imp->shape != target.imp->shape) {
            throw std::runtime_error("target must be of the same shape");
        }
        if (axis < -1 || axis >= this->imp->shape.size()) {
            throw std::invalid_argument("Invalid axis");
        }

        if (axis == -1 && this->is_contiguous() && target.is_contiguous()) {
            std::vector<float> outdata(1);
            float m = this->max(-1).imp->storage->data[0];
            float cs = 0;
            for (size_t i = 0; i < this->imp->length; ++i) {
                cs += std::exp(this->imp->storage->data[i + this->imp->offset] - m);
            }
            float lse{m + std::log(cs)};
            for (size_t i = 0; i < this->imp->length; ++i) {
                outdata[0] -= target.imp->storage->data[i + target.imp->offset] * (this->imp->storage->data[i + this->imp->offset] - lse);
            }
            
            Tensor out(std::vector<size_t>{1}, outdata, this->imp->requires_grad);

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::SOFTMAX_CE;
                out.imp->parents = {this->imp, target.imp};
            }

            switch (reduction) {
            case 0: {
                return out.sum();
            }
            case 1: {
                return out.mean();
            }
            case 2: {
                return out;
            }
            default: {
                throw std::invalid_argument("Invalid reduction type");
            }
            }
        }
        else if (axis == -1) {
            std::vector<float> outdata(1);
            float m = this->max(-1).imp->storage->data[0];
            float cs = 0;
            float ti = this->imp->offset, tai = target.imp->offset;
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            for (size_t n = 0; n < this->imp->length; ++n) {
                cs += std::exp(this->imp->storage->data[ti] - m);

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->imp->strides[d];

                    if (coord[d] < this->imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->imp->shape[d] * this->imp->strides[d];
                }
            }

            ti = this->imp->offset;
            float lse{m + std::log(cs)};
            std::fill(coord.begin(), coord.end(), 0);

            for (size_t n = 0; n < this->imp->length; ++n) {
                outdata[0] -= target.imp->storage->data[tai] * (this->imp->storage->data[ti] - lse);

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    ti += this->imp->strides[d];
                    tai += target.imp->strides[d];

                    if (coord[d] < this->imp->shape[d]) break;

                    coord[d] = 0;

                    ti -= this->imp->shape[d] * this->imp->strides[d];
                    tai -= this->imp->shape[d] * this->imp->strides[d];
                }
            }

            Tensor out(std::vector<size_t>{1}, outdata, this->imp->requires_grad);

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::SOFTMAX_CE;
                out.imp->parents = {this->imp, target.imp};
            }

            switch (reduction) {
            case 0: {
                return out.sum();
            }
            case 1: {
                return out.mean();
            }
            case 2: {
                return out;
            }
            default: {
                throw std::invalid_argument("Invalid reduction type");
            }
            }
        }
        else {
            std::vector<float> outdata(this->imp->length / this->imp->shape[axis]);
            std::vector<float> m = this->max_data(axis);
            std::vector<float> lse(this->imp->length / this->imp->shape[axis]);
            float cs = 0;
            std::vector<size_t> coord(this->imp->shape.size(), 0);
            size_t y = 0;

            while (true) {
                cs = 0;

                size_t ti = this->imp->offset;
                size_t tai = target.imp->offset;

                for (int d = 0; d < this->imp->shape.size(); ++d) {
                    ti += coord[d] * this->imp->strides[d];
                    tai += coord[d] * target.imp->strides[d];
                }

                for (size_t n = 0; n < this->imp->shape[axis]; ++n) {
                    cs += std::exp(this->imp->storage->data[ti] - m[y]);
                    ti += this->imp->strides[axis];
                }

                lse[y] = m[y] + std::log(cs);

                ti = this->imp->offset;
                tai = target.imp->offset;
                for (int d = 0; d < this->imp->shape.size(); ++d) {
                    ti += coord[d] * this->imp->strides[d];
                    tai += coord[d] * target.imp->strides[d];
                }

                for (size_t n = 0; n < this->imp->shape[axis]; ++n) {
                    outdata[y] -= target.imp->storage->data[tai] * (this->imp->storage->data[ti] - lse[y]);
                    ti += this->imp->strides[axis];
                    tai += target.imp->strides[axis];
                }

                bool completed = true;

                for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                    if (d == axis) continue;

                    ++coord[d];

                    if (coord[d] < this->imp->shape[d]) {
                        completed = false;
                        break;
                    }

                    coord[d] = 0;
                }

                if (completed) break;

                ++y;
            }

            std::vector<size_t> out_shape(this->imp->shape.size() - 1);
            for (int d = 0; d < this->imp->shape.size(); ++d) {
                if (d < axis) out_shape[d] = this->imp->shape[d];
                else if (d > axis) out_shape[d - 1] = this->imp->shape[d];
            }

            Tensor out(out_shape, outdata, this->imp->requires_grad);

            if (out.imp->requires_grad) {
                out.imp->op_type = Operation::SOFTMAX_CE;
                out.imp->parents = {this->imp, target.imp};
                out.imp->op_axis = axis;
            }

            switch (reduction) {
            case 0: {
                return out.sum();
            }
            case 1: {
                return out.mean();
            }
            case 2: {
                return out;
            }
            default: {
                throw std::invalid_argument("Invalid reduction type");
            }
            }
        }
    }

    Tensor sparse_softmax_cross_entropy(const Tensor& target, const int axis = 1, const int reduction = 0) const {
        if (axis < 0 || axis >= this->imp->shape.size()) {
            throw std::invalid_argument("Invalid axis");
        }
        std::vector<float> outdata(this->imp->length / this->imp->shape[axis]);
        std::vector<float> m = this->max_data(axis);
        std::vector<size_t> coord(this->imp->shape.size(), 0);
        size_t y = 0;

        while (true) {
            float cs = 0;

            size_t ti = this->imp->offset;
            size_t tai = target.imp->offset;
            size_t td = 0;

            for (int d = 0; d < this->imp->shape.size(); ++d) {
                ti += coord[d] * this->imp->strides[d];
                if (d != axis) {
                    tai += coord[d] * target.imp->strides[td];
                    ++td;
                }
            }

            for (size_t n = 0; n < this->imp->shape[axis]; ++n) {
                cs += std::exp(this->imp->storage->data[ti] - m[y]);
                ti += this->imp->strides[axis];
            }

            float lse = m[y] + std::log(cs);

            ti -= this->imp->shape[axis] * this->imp->strides[axis];

            float t = target.imp->storage->data[tai];

            if (!std::isfinite(t) || t < 0 || t >= this->imp->shape[axis] || std::floor(t) != t) {
                throw std::invalid_argument("Invalid target");
            }
            
            outdata[y] = lse - this->imp->storage->data[ti + static_cast<size_t>(t) * this->imp->strides[axis]];

            bool completed = true;

            for (int d = this->imp->shape.size() - 1; d >= 0; --d) {
                if (d == axis) continue;

                ++coord[d];

                if (coord[d] < this->imp->shape[d]) {
                    completed = false;
                    break;
                }

                coord[d] = 0;
            }

            if (completed) break;

            ++y;
        }

        std::vector<size_t> out_shape(this->imp->shape.size() - 1);
        for (int d = 0; d < this->imp->shape.size(); ++d) {
            if (d < axis) out_shape[d] = this->imp->shape[d];
            else if (d > axis) out_shape[d - 1] = this->imp->shape[d];
        }

        Tensor out(out_shape, outdata, this->imp->requires_grad);

        if (out.imp->requires_grad) {
            out.imp->op_type = Operation::SPARSE_SCE;
            
            out.imp->parents = {this->imp, target.imp};
            out.imp->op_axis = axis;
        }

        switch (reduction) {
        case 0: {
            return out.sum();
        }
        case 1: {
            return out.mean();
        }
        case 2: {
            return out;
        }
        default: {
            throw std::invalid_argument("Invalid reduction type");
        }
        }
    }

    void backward() {
        if (this->imp->length != 1) {
            throw std::runtime_error("backward() requires scalar output");
        }

        if (!this->imp->requires_grad) return;

        std::vector<std::shared_ptr<TensorImp>> topo;
        std::set<TensorImp*> visited;

        build_topo(&topo, &visited, this->imp);

        this->imp->grad_storage->data[this->imp->offset] = 1.0f;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            auto& n = *it;

            if (n->requires_grad) {
                switch (n->op_type) {
                case Operation::ADD: {
                    backward_add(n);
                    break;
                }
                case Operation::SUB: {
                    backward_sub(n);
                    break;
                }
                case Operation::MUL: {
                    backward_mul(n);
                    break;
                }
                case Operation::DIV: {
                    backward_div(n);
                    break;
                }
                case Operation::POW: {
                    backward_pow(n);
                    break;
                }
                case Operation::MATMUL: {
                    backward_matmul(n);
                    break;
                }
                case Operation::CONV2D: {
                    backward_conv2d(n);
                    break;
                }
                case Operation::MAXPOOL2D: {
                    backward_maxpool2d(n);
                    break;
                }
                case Operation::SUM: {
                    backward_sum(n);
                    break;
                }
                case Operation::MEAN: {
                    backward_mean(n);
                    break;
                }
                case Operation::VARIANCE: {
                    backward_variance(n);
                    break;
                }
                case Operation::TRANSPOSE: {
                    backward_transpose(n);
                    break;
                }
                case Operation::RELU: {
                    backward_relu(n);
                    break;
                }
                case Operation::LEAKYRELU: {
                    backward_leakyrelu(n);
                    break;
                }
                case Operation::GELU: {
                    backward_gelu(n);
                    break;
                }
                case Operation::SIGMOID: {
                    backward_sigmoid(n);
                    break;
                }
                case Operation::RESHAPE: {
                    backward_reshape(n);
                    break;
                }
                case Operation::SLICE: {
                    backward_slice(n);
                    break;
                }
                case Operation::CONCAT: {
                    backward_concat(n);
                    break;
                }
                case Operation::SOFTMAX: {
                    backward_softmax(n);
                    break;
                }
                case Operation::SOFTMAX_CE: {
                    backward_sce(n);
                    break;
                }
                case Operation::SPARSE_SCE: {
                    backward_sparse_sce(n);
                    break;
                }
                case Operation::EMBED: {
                    backward_embed(n);
                    break;
                }
                default: {
                    break;
                }
                }
            }
        }
    }

    void zero_grad() {
        if (this->imp->requires_grad && this->imp->grad_storage) {
            std::fill(this->imp->grad_storage->data.begin(), this->imp->grad_storage->data.end(), 0.0f);
        }
    }
};



Tensor create_causal_mask(const size_t t) {
    std::vector<float> mask(t * t, 0.0f);
    for (size_t i = 0; i < t; ++i) {
        for (size_t j = i + 1; j < t; ++j) {
            mask[i * t + j] = -1e9f;
        }
    }
    
    return Tensor({1, t, t}, mask, false);
}



//RNG

class RNG {
    std::mt19937 gen;
    std::normal_distribution<float> ndist{0.0f, 1.0f};

public:
    RNG(unsigned seed = std::random_device{}()) : gen(seed) {}

    float normal() {
        return ndist(gen);
    }

    std::mt19937& get_gen() {
        return gen;
    }
};

RNG rng(42);

//Modules

class Module {
public:
    virtual Tensor forward(const Tensor& x) = 0;
    virtual ~Module() = default;
    virtual std::vector<Tensor*> parameters() = 0;
};

class Linear : public Module {
    [[maybe_unused]] size_t nin;
    [[maybe_unused]] size_t nout;
    Tensor weights;
    Tensor biases;
    

public:
    Linear(size_t numin, size_t numout) : nin(numin), nout(numout) {
        float sd = std::sqrt(2.0f / numin);
        std::vector<float> ws(numin * numout);

        for (auto& x : ws) {
            x = rng.normal() * sd;
        }

        weights = Tensor(std::vector<size_t>{numin, numout}, ws, true);

        std::vector<float> bs(numout, 0.0f);

        biases = Tensor(std::vector<size_t>{numout}, bs, true);
    }

    Tensor get_weights() {
        return weights;
    }

    Tensor get_biases() {
        return biases;
    }

    Tensor forward(const Tensor& other) override {
        if (other.imp->shape.size() == 2) {
            return other.matmul(weights) + biases;
        }
        else if (other.imp->shape.size() == 3) {
            return nn::linear3d(other, *this);
        }
        else {
            throw std::invalid_argument("Input must be 2 or 3 dimensional");
        }
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>{&weights, &biases};
    }
};

class Conv2D : public Module {
    Tensor weights;
    Tensor biases;
    size_t stride;
    size_t padding;

public:
    Conv2D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0) : stride(stride), padding(padding) {
        float sd = std::sqrt(2.0f / (input_channels * kernel_size * kernel_size));
        std::vector<float> ws(output_channels * input_channels * kernel_size * kernel_size);

        for (auto& x : ws) {
            x = rng.normal() * sd;
        }

        weights = Tensor(std::vector<size_t>{output_channels, input_channels, kernel_size, kernel_size}, ws, true);

        std::vector<float> bs(output_channels, 0);

        biases = Tensor(std::vector<size_t>{output_channels}, bs, true);
    }

    Tensor get_weights() {
        return weights;
    }

    Tensor get_biases() {
        return biases;
    }

    Tensor forward(const Tensor& other) override {
        return other.conv2d(weights, biases, stride, padding);
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>{&weights, &biases};
    }
};

class ReLU : public Module {
public:
    Tensor forward(const Tensor& other) override {
        return other.relu();
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>();
    }
};

class LeakyReLU : public Module {
    float alpha = 0.01;

public:
    LeakyReLU(float alpha = 0.01) : alpha(alpha) {}

    Tensor forward(const Tensor& other) override {
        return other.leaky_relu(alpha);
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>();
    }
};

class GELU : public Module {
public:
    Tensor forward(const Tensor& other) override {
        return other.gelu();
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>();
    }
};

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& other) override {
        return other.sigmoid();
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>();
    }
};

class Softmax : public Module {
    int axis;

public:
    Softmax(const int axis = 1) : axis(axis) {
        if (axis < 0) {
            throw std::invalid_argument("Invalid axis");
        }
    }

    Tensor forward(const Tensor& other) override {
        return other.softmax(axis);
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>();
    }
};

class Flatten : public Module {
    int dim;

public:
    Flatten(const int dim = 0) : dim(dim) {}

    Tensor forward(const Tensor& other) override {
        return other.flatten(dim);
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>();
    }
};

class MaxPool2D : public Module {
    size_t kernel_size;
    size_t stride;
    size_t padding;

public:
    MaxPool2D(size_t kernel_size, size_t stride, size_t padding) : kernel_size(kernel_size), stride(stride), padding(padding) {}

    Tensor forward(const Tensor& other) override {
        return other.maxpool2d(kernel_size, stride, padding);
    }

    std::vector<Tensor*> parameters() override {
        return std::vector<Tensor*>();
    }
};

class Sequential : public Module {
    std::vector<std::unique_ptr<Module>> layers;

public:
    template<class... Layers>
    explicit Sequential(Layers&&... ls) {
        (layers.push_back(std::make_unique<std::decay_t<Layers>>(std::forward<Layers>(ls))), ...);
    }

    Sequential() = default;

    template<class T>
    void add(const T& layer) {
        layers.push_back(std::make_unique<T>(layer));
    }

    Tensor forward(const Tensor& other) override {
        Tensor x = other;
        for (auto& l : this->layers) {
            x = (*l).forward(x);
        }
        return x;
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> out;
        for (auto& x : layers) {
            auto pm = x->parameters();
            out.insert(out.end(), pm.begin(), pm.end());
        }
        return out;
    }

    auto& get_layers() {
        return layers;
    }
};

class AttentionHead : public Module {
    std::vector<Linear> qkv;

public:
    AttentionHead(const size_t d, const size_t dh) {
        qkv.emplace_back(d, dh);
        qkv.emplace_back(d, dh);
        qkv.emplace_back(d, dh);
    }

    using Module::forward;

    Tensor forward(const Tensor& x) override {
        Tensor mask = create_causal_mask(x.imp->shape[1]);
        return forward(x, mask);
    }

    Tensor forward(const Tensor& other, const Tensor& mask) {
        return Tensor::causal_attention(qkv[0].forward(other), qkv[1].forward(other), qkv[2].forward(other), mask);
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> out;
        for (auto& x : qkv) {
            auto pm = x.parameters();
            out.insert(out.end(), pm.begin(), pm.end());
        }
        return out;
    }
};

class MultiHeadAttention : public Module {
    std::vector<AttentionHead> heads;
    Linear out_proj;

public:
    MultiHeadAttention(const size_t dim, const size_t heads) : out_proj(dim, dim) {
        if (dim == 0 || heads == 0 || dim % heads != 0) {
            throw std::invalid_argument("Invalid attention dimensions");
        }

        const size_t dh = dim / heads;
        for (size_t i = 0; i < heads; ++i) {
            this->heads.emplace_back(dim, dh);
        }
    }

    Tensor forward(const Tensor& other) override {
        Tensor causal_mask = create_causal_mask(other.imp->shape[1]);
        std::vector<Tensor> outs(heads.size());
        for (size_t i = 0; i < heads.size(); ++i) {
            outs[i] = heads[i].forward(other, causal_mask);
        }
        return out_proj.forward(Tensor::concat(outs, 2));
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> out;
        for (auto& x : heads) {
            auto pm = x.parameters();
            out.insert(out.end(), pm.begin(), pm.end());
        }
        auto pm = out_proj.parameters();
        out.insert(out.end(), pm.begin(), pm.end());
        return out;
    }
};

class LayerNorm : public Module {
    Tensor scale;
    Tensor bias;

public:
    LayerNorm(const size_t dim) {
        scale = Tensor({dim}, std::vector<float>(dim, 1.0f), true);
        bias = Tensor({dim}, std::vector<float>(dim, 0.0f), true);
    }

    Tensor forward(const Tensor& other) override {
        return Tensor::layernorm(other, scale, bias);
    }

    std::vector<Tensor*> parameters() override {
        return {&scale, &bias};
    }
};

class TransformerBlock : public Module {
    MultiHeadAttention mh;
    Sequential sq;
    LayerNorm ln1;
    LayerNorm ln2;

public:
    TransformerBlock(const size_t dim, const size_t heads) : mh(dim, heads), sq(Linear(dim, 4 * dim), GELU(), Linear(4 * dim, dim)), ln1(dim), ln2(dim) {}

    Tensor forward(const Tensor& other) override {
        Tensor x = other;
        x = x + mh.forward(ln1.forward(x));
        x = x + sq.forward(ln2.forward(x));
        return x;
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> out;
        auto pm = mh.parameters();
        out.insert(out.end(), pm.begin(), pm.end());
        pm = sq.parameters();
        out.insert(out.end(), pm.begin(), pm.end());
        pm = ln1.parameters();
        out.insert(out.end(), pm.begin(), pm.end());
        pm = ln2.parameters();
        out.insert(out.end(), pm.begin(), pm.end());
        return out;
    }
};

class Embedding : public Module {
    Tensor weights;

public:
    Embedding(const size_t vocab, const size_t dim) {
        std::vector<float> ws(vocab * dim);

        for (auto& x : ws) {
            x = rng.normal();
        }

        weights = Tensor(std::vector<size_t>{vocab, dim}, ws, true);
    }

    Tensor forward(const Tensor& other) override {
        return Tensor::embedding(other, weights);
    }

    std::vector<Tensor*> parameters() override {
        return {&weights};
    }
};

class PositionalEmbedding : public Module {
    Tensor weights;
    size_t len;

public:
    PositionalEmbedding(const size_t len, const size_t dim) : len(len) {
        float sd = 1.0f;
        std::vector<float> ws(len * dim);

        for (auto& x : ws) {
            x = rng.normal() * sd;
        }

        weights = Tensor({len, dim}, ws, true);
    }

    Tensor forward(const Tensor& other) override {
        if (other.imp->shape[1] > len) {
            throw std::invalid_argument("Token sequence too long");
        }

        std::vector<size_t> pos(other.imp->shape[1]);
        std::iota(pos.begin(), pos.end(), 0);

        Tensor p = Tensor::embedding(pos, weights);

        return other + p;
    }

    std::vector<Tensor*> parameters() override {
        return {&weights};
    }
};

class GPT : public Module {
    Embedding token_embed;
    PositionalEmbedding pos_embed;
    std::vector<TransformerBlock> blocks;
    LayerNorm lnf;
    Linear ffend;

public:
    GPT(const size_t vocab, const size_t dim, const size_t heads, const size_t nblocks, const size_t len) : token_embed(vocab, dim), pos_embed(len, dim), lnf(dim), ffend(dim, vocab) {
        for (size_t i = 0; i < nblocks; ++i) {
            blocks.emplace_back(dim, heads);
        }
    }

    Tensor forward(const Tensor& other) override {
        Tensor x = pos_embed.forward(token_embed.forward(other));

        for (auto& b : blocks) {
            x = b.forward(x);
        }

        return ffend.forward(lnf.forward(x));
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> out;
        auto pm = token_embed.parameters();
        out.insert(out.end(), pm.begin(), pm.end());
        pm = pos_embed.parameters();
        out.insert(out.end(), pm.begin(), pm.end());
        for (auto& x : blocks) {
            pm = x.parameters();
            out.insert(out.end(), pm.begin(), pm.end());
        }
        pm = lnf.parameters();
        out.insert(out.end(), pm.begin(), pm.end());
        pm = ffend.parameters();
        out.insert(out.end(), pm.begin(), pm.end());

        return out;
    }
};

//Optimisers

class Adam {
    struct State {
        std::vector<float> m;
        std::vector<float> v;
    };

    std::vector<State> states;
    std::vector<Tensor*> parameters;
    float lr, beta1, beta2, epsilon, max_norm;
    size_t t = 1;

public:
    Adam(std::vector<Tensor*> parameters, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float max_norm = 1.0) : parameters(std::move(parameters)), lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), max_norm(max_norm) {
        states.resize(this->parameters.size());
        for (int i = 0; i < this->parameters.size(); ++i) {
            states[i] = {std::vector<float>(this->parameters[i]->imp->length, 0.0f), std::vector<float>(this->parameters[i]->imp->length, 0.0f)};
        }
    }

    void step() {
        float b1c = 1 - std::pow(beta1, t);
        float b2c = 1 - std::pow(beta2, t);
        double cs = 0;

        for (int k = 0; k < parameters.size(); ++k) {
            Tensor* p = parameters[k];
            std::vector<size_t> coord(p->imp->shape.size(), 0);
            size_t pi = p->imp->offset;
            for (size_t i = 0; i < p->imp->length; ++i) {
                float g = p->imp->grad_storage->data[pi];
                cs += g * g;

                for (int d = p->imp->shape.size() - 1; d >= 0; --d) {
                    ++coord[d];

                    pi += p->imp->strides[d];

                    if (coord[d] < p->imp->shape[d]) break;

                    coord[d] = 0;

                    pi -= p->imp->shape[d] * p->imp->strides[d];
                }
            }
        }
        float gnorm = std::sqrt(cs);
        if (gnorm > max_norm) {
            for (int k = 0; k < parameters.size(); ++k) {
                Tensor* p = parameters[k];
                std::vector<size_t> coord(p->imp->shape.size(), 0);
                size_t pi = p->imp->offset;
                for (size_t i = 0; i < p->imp->length; ++i) {
                    p->imp->grad_storage->data[pi] *= max_norm / gnorm;

                    for (int d = p->imp->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];

                        pi += p->imp->strides[d];

                        if (coord[d] < p->imp->shape[d]) break;

                        coord[d] = 0;

                        pi -= p->imp->shape[d] * p->imp->strides[d];
                    }
                }
            }
        }

        for (int k = 0; k < parameters.size(); ++k) {
            Tensor* p = parameters[k];
            auto& s = states[k];
            float mh, vh;

            if (p->is_contiguous()) {
                size_t j = 0;
                for (size_t i = p->imp->offset; i < p->imp->length + p->imp->offset; ++i) {
                    float g = p->imp->grad_storage->data[i];
                    s.m[j] = beta1 * s.m[j] + (1 - beta1) * g;
                    s.v[j] = beta2 * s.v[j] + (1 - beta2) * g * g;
                    mh = s.m[j] / b1c;
                    vh = s.v[j] / b2c;
                    p->imp->storage->data[i] -= lr * mh / (epsilon + std::sqrt(vh));
                    ++j;
                }
            }
            else {
                std::vector<size_t> coord(p->imp->shape.size(), 0);
                size_t i = p->imp->offset;
                size_t j = 0;

                for (size_t n = 0; n < p->imp->length; ++n) {
                    float g = p->imp->grad_storage->data[i];
                    s.m[j] = beta1 * s.m[j] + (1 - beta1) * g;
                    s.v[j] = beta2 * s.v[j] + (1 - beta2) * g * g;
                    mh = s.m[j] / b1c;
                    vh = s.v[j] / b2c;
                    p->imp->storage->data[i] -= lr * mh / (epsilon + std::sqrt(vh));

                    for (int d = p->imp->shape.size() - 1; d >= 0; --d) {
                        ++coord[d];

                        i += p->imp->strides[d];

                        if (coord[d] < p->imp->shape[d]) break;

                        coord[d] = 0;

                        i -= p->imp->shape[d] * p->imp->strides[d];
                    }

                    ++j;
                }
            }
        }
        ++t;
    }

    float get_lr() {
        return lr;
    }

    void set_lr(float x) {
        lr = x;
    }

    std::vector<std::vector<float>> get_m() const {
        std::vector<std::vector<float>> out;
        out.reserve(states.size());

        for (const auto& state : states) {
            out.push_back(state.m);
        }

        return out;
    }

    std::vector<std::vector<float>> get_v() const {
        std::vector<std::vector<float>> out;
        out.reserve(states.size());

        for (const auto& state : states) {
            out.push_back(state.v);
        }

        return out;
    }

    size_t get_t() const {
        return t;
    }

    float get_beta1() const { return beta1; }
    float get_beta2() const { return beta2; }
    float get_epsilon() const { return epsilon; }
    float get_max_norm() const { return max_norm; }

    void load_state(
        const std::vector<std::vector<float>>& m_in,
        const std::vector<std::vector<float>>& v_in,
        size_t t_in,
        float lr_in,
        float beta1_in,
        float beta2_in,
        float epsilon_in,
        float max_norm_in
    ) {
        if (m_in.size() != states.size() ||
            v_in.size() != states.size()) {
            throw std::invalid_argument("Adam state parameter count mismatch");
        }

        if (t_in < 1) {
            throw std::invalid_argument("Invalid Adam timestep");
        }

        for (size_t i = 0; i < states.size(); ++i) {
            if (m_in[i].size() != states[i].m.size() ||
                v_in[i].size() != states[i].v.size()) {
                throw std::invalid_argument("Adam moment size mismatch");
            }
        }

        for (size_t i = 0; i < states.size(); ++i) {
            states[i].m = m_in[i];
            states[i].v = v_in[i];
        }

        t = t_in;
        lr = lr_in;
        beta1 = beta1_in;
        beta2 = beta2_in;
        epsilon = epsilon_in;
        max_norm = max_norm_in;
    }
};

//DataLoader

class DataLoaderCSV {
    std::vector<float> x;
    std::vector<float> y;
    size_t num_examples;
    size_t example_size;
    size_t mb_size;
    bool shuffle;

    friend class DataLoaderIterator;

public:
    DataLoaderCSV(const std::string& filepath, size_t num_examples, size_t example_size, size_t mb_size = 128, bool shuffle = true) : num_examples(num_examples), example_size(example_size), mb_size(mb_size), shuffle(shuffle) {
        std::ifstream file(filepath);
        if (file.is_open()) {
            x.resize(example_size * num_examples);
            y.resize(num_examples * 10);

            std::string line;
            size_t i = 0;

            while (std::getline(file, line)) {
                const char* p = line.c_str();
                char* end = nullptr;

                long label = std::strtol(p, &end, 10);

                p = end + 1;

                for (size_t j = 0; j < example_size; ++j) {
                    long px = std::strtol(p, &end, 10);

                    x[i * example_size + j] = static_cast<float>(px) / 255.0f;
                    p = end;

                    if (j + 1 < example_size) ++p;
                }

                for (size_t k = 0; k < 10; ++k) {
                    y[i * 10 + k] = (k == static_cast<size_t>(label)) ? 1.0f : 0.0f;
                }

                ++i;
            }
        }
    }

    size_t size() const {
        return num_examples;
    }

    size_t num_batches() const {
        return (num_examples + mb_size - 1) / mb_size;
    }

    inline DataLoaderIterator iter();
};

class DataLoaderIterator {
    DataLoaderCSV* loader;
    std::vector<size_t> indices;
    size_t current_index = 0;

public:
    explicit DataLoaderIterator(DataLoaderCSV& loader) : loader(&loader), indices(loader.num_examples) {
        std::iota(indices.begin(), indices.end(), 0);
        if (this->loader->shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng.get_gen());
        }
    }

    DataLoaderIterator& iter() {
        return *this;
    }

    std::pair<Tensor, Tensor> next() {
        if (current_index >= indices.size()) {
            throw py::stop_iteration();
        }

        const size_t count = std::min(loader->mb_size, indices.size() - current_index);

        std::vector<float> xb(count * loader->example_size);
        std::vector<float> yb(count * 10);

        for (size_t i = 0; i < count; ++i) {
            size_t j = indices[current_index + i];
            std::copy_n(loader->x.data() + j * loader->example_size, loader->example_size, xb.data() + i * loader->example_size);
            std::copy_n(loader->y.data() + j * 10, 10, yb.data() + i * 10);
        }

        current_index += count;

        return {Tensor({count, loader->example_size}, xb, false), Tensor({count, 10}, yb, false)};
    }
};

inline DataLoaderIterator DataLoaderCSV::iter() {
    return DataLoaderIterator(*this);
}


namespace nn {
    Tensor linear3d(const Tensor& x, Linear& layer) {
        if (x.imp->shape.size() != 3) {
            throw std::invalid_argument("linear3d requires three dimensional input");
        }

        const size_t b = x.imp->shape[0];
        const size_t t = x.imp->shape[1];

        Tensor y = x.reshape({b * t, x.imp->shape[2]});
        y = layer.forward(y);
        y = y.reshape({b, t, y.imp->length / (b * t)});
        return y;
    }
}

std::vector<Slice> parse_slices(
    const py::object& key,
    const std::vector<size_t>& shape
) {
    const size_t rank = shape.size();

    std::vector<Slice> slices(rank, Slice{0, 0, 1});

    for (size_t d = 0; d < rank; ++d) {
        slices[d] = {0, shape[d], 1};
    }

    py::tuple indices;

    if (py::isinstance<py::tuple>(key)) {
        indices = py::cast<py::tuple>(key);
    } else {
        indices = py::make_tuple(key);
    }

    if (indices.size() > rank) {
        throw py::index_error("Too many indices for tensor");
    }

    for (size_t d = 0; d < indices.size(); ++d) {
        py::handle item = indices[d];

        if (!py::isinstance<py::slice>(item)) {
            throw py::type_error(
                "Integer indexing is not supported yet"
            );
        }

        py::slice s = py::reinterpret_borrow<py::slice>(item);

        py::tuple normalized = s.attr("indices")(
            py::int_(shape[d])
        );

        ssize_t start = normalized[0].cast<ssize_t>();
        ssize_t stop  = normalized[1].cast<ssize_t>();
        ssize_t step  = normalized[2].cast<ssize_t>();

        if (step <= 0) {
            throw py::value_error(
                "Only positive slice steps are supported"
            );
        }

        slices[d] = {
            static_cast<size_t>(start),
            static_cast<size_t>(stop),
            static_cast<size_t>(step)
        };
    }

    return slices;
}


PYBIND11_MODULE(tensorlib, t) {
    py::class_<Tensor>(t, "Tensor")
        .def(py::init<std::vector<size_t>, const bool>(), py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<std::vector<size_t>, std::vector<float>, const bool>(), py::arg("shape"), py::arg("data"), py::arg("requires_grad") = false)
        .def_property_readonly("data", &Tensor::get_data)
        .def_property_readonly("grads", &Tensor::get_grads)
        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("length", &Tensor::get_length)
        .def("__add__", py::overload_cast<const Tensor&>(&Tensor::operator+, py::const_))
        .def("__add__", py::overload_cast<const float>(&Tensor::operator+, py::const_))
        .def("__sub__", &Tensor::operator-)
        .def("__mul__", py::overload_cast<const Tensor&>(&Tensor::operator*, py::const_))
        .def("__mul__", py::overload_cast<const float>(&Tensor::operator*, py::const_))
        .def("__truediv__", py::overload_cast<const Tensor&>(&Tensor::operator/, py::const_))
        .def("__truediv__", py::overload_cast<const float>(&Tensor::operator/, py::const_))
        .def("__pow__", &Tensor::pow, py::arg("x"))
        .def("sqrt", &Tensor::sqrt)
        .def("__matmul__", &Tensor::matmul)
        .def("transpose", &Tensor::transpose, py::arg("a") = 0, py::arg("b") = 1)
        .def("backward", &Tensor::backward)
        .def("sum", &Tensor::sum, py::arg("axis") = -1)
        .def("mean", &Tensor::mean, py::arg("axis") = -1)
        .def("variance", &Tensor::variance, py::arg("axis") = -1)
        .def("relu", &Tensor::relu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("softmax_cross_entropy", &Tensor::softmax_cross_entropy, py::arg("target"), py::arg("axis") = 1, py::arg("reduction") = 0)
        .def("sparse_sce", &Tensor::sparse_softmax_cross_entropy, py::arg("target"), py::arg("axis") = 1, py::arg("reduction") = 0)
        .def("set_item", &Tensor::set_item, py::arg("index"), py::arg("value"))
        .def("set_data", &Tensor::set_data)
        .def("argmax", &Tensor::argmax, py::arg("axis") = -1)
        .def("zero_grad", &Tensor::zero_grad)
        .def("conv2d", &Tensor::conv2d, py::arg("weights"), py::arg("biases"), py::arg("stride") = 1, py::arg("padding") = 0)
        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("slice", &Tensor::slice, py::arg("slices"))
        .def("flatten", &Tensor::flatten, py::arg("dim"))
        .def("maxpool2d", &Tensor::maxpool2d, py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 0)
        .def("softmax", &Tensor::softmax, py::arg("axis"))
        .def_static("embedding", py::overload_cast<const std::vector<size_t>&, const Tensor&>(&Tensor::embedding), py::arg("tokens"), py::arg("weights"))
        .def_static("embedding", py::overload_cast<const Tensor&, const Tensor&>(&Tensor::embedding), py::arg("tokens"), py::arg("weights"))
        .def_static("causal_attention", &Tensor::causal_attention, py::arg("query"), py::arg("key"), py::arg("value"), py::arg("mask"))
        .def_static("concat", &Tensor::concat, py::arg("tensors"), py::arg("axis"))
        .def_static("layernorm", &Tensor::layernorm, py::arg("x"), py::arg("scale"), py::arg("bias"), py::arg("eps") = 1e-5)
        .def("__getitem__", [] (const Tensor& self, py::object key) {return self.slice(parse_slices(key, self.get_shape()));});
    
    py::class_<Slice>(t, "Slice")
        .def(py::init<const size_t, const size_t, const size_t>(), py::arg("start"), py::arg("stop"), py::arg("step"))
        .def(py::init<const std::vector<size_t>&>(), py::arg("bounds"));
    
    py::module_ neuralnet = t.def_submodule("nn", "Neural network frameworks");
    neuralnet.def("linear3d", &nn::linear3d, py::arg("x"), py::arg("layer"));

    py::class_<Linear>(t, "Linear")
        .def(py::init<size_t, size_t>(), py::arg("numin"), py::arg("numout"))
        .def("__call__", &Linear::forward, py::arg("x"))
        .def("parameters", &Linear::parameters, py::return_value_policy::reference_internal);
    
    py::class_<Conv2D>(t, "Conv2D")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(), py::arg("input_channels"), py::arg("output_channels"), py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 0)
        .def_property_readonly("weights", &Conv2D::get_weights, py::return_value_policy::reference)
        .def_property_readonly("biases", &Conv2D::get_biases, py::return_value_policy::reference)
        .def("__call__", &Conv2D::forward, py::arg("x"))
        .def("parameters", &Conv2D::parameters, py::return_value_policy::reference_internal);
    
    py::class_<MaxPool2D>(t, "MaxPool2D")
        .def(py::init<size_t, size_t, size_t>(), py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 0)
        .def("__call__", &MaxPool2D::forward, py::arg("x"))
        .def("parameters", &MaxPool2D::parameters);

    py::class_<ReLU>(t, "ReLU")
        .def(py::init<>())
        .def("__call__", &ReLU::forward, py::arg("x"))
        .def("parameters", &ReLU::parameters);

    py::class_<LeakyReLU>(t, "LeakyReLU")
        .def(py::init<float>(), py::arg("alpha") = 0.01)
        .def("__call__", &LeakyReLU::forward, py::arg("x"))
        .def("parameters", &LeakyReLU::parameters);
    
    py::class_<GELU>(t, "GELU")
        .def(py::init<>())
        .def("__call__", &GELU::forward, py::arg("x"))
        .def("parameters", &GELU::parameters);

    py::class_<Sigmoid>(t, "Sigmoid")
        .def(py::init<>())
        .def("__call__", &Sigmoid::forward, py::arg("x"))
        .def("parameters", &Sigmoid::parameters);

    py::class_<Flatten>(t, "Flatten")
        .def(py::init<const int>(), py::arg("dim"))
        .def("__call__", &Flatten::forward, py::arg("x"))
        .def("parameters", &Flatten::parameters);
    
    py::class_<Sequential>(t, "Sequential")
        .def(py::init([](py::args args){
            Sequential seq;
            for (py::handle arg : args) {
                if (py::isinstance<Linear>(arg)) {
                    seq.add(py::cast<const Linear&>(arg));
                }
                else if (py::isinstance<ReLU>(arg)) {
                    seq.add(py::cast<const ReLU&>(arg));
                }
                else if (py::isinstance<LeakyReLU>(arg)) {
                    seq.add(py::cast<const LeakyReLU&>(arg));
                }
                else if (py::isinstance<Sigmoid>(arg)) {
                    seq.add(py::cast<const Sigmoid&>(arg));
                }
                else if (py::isinstance<Flatten>(arg)) {
                    seq.add(py::cast<const Flatten&>(arg));
                }
                else if (py::isinstance<Conv2D>(arg)) {
                    seq.add(py::cast<const Conv2D&>(arg));
                }
                else if (py::isinstance<MaxPool2D>(arg)) {
                    seq.add(py::cast<const MaxPool2D&>(arg));
                }
                else {
                    throw std::invalid_argument("Invalid layer");
                }
            }
            return seq;
        }))
        .def_property_readonly("layers", &Sequential::get_layers, py::return_value_policy::reference_internal)
        .def("__call__", &Sequential::forward, py::arg("x"))
        .def("parameters", &Sequential::parameters, py::return_value_policy::reference_internal);
    
    py::class_<MultiHeadAttention>(t, "MultiHeadAttention")
        .def(py::init<const size_t, const size_t>(), py::arg("dim"), py::arg("num_heads"))
        .def("__call__", &MultiHeadAttention::forward, py::arg("x"))
        .def("parameters", &MultiHeadAttention::parameters, py::return_value_policy::reference_internal);
    
    py::class_<LayerNorm>(t, "LayerNorm")
        .def(py::init<const size_t>(), py::arg("dim"))
        .def("__call__", &LayerNorm::forward, py::arg("x"))
        .def("parameters", &LayerNorm::parameters, py::return_value_policy::reference_internal);
    
    py::class_<TransformerBlock>(t, "TransformerBlock")
        .def(py::init<const size_t, const size_t>(), py::arg("dim"), py::arg("num_heads"))
        .def("__call__", &TransformerBlock::forward, py::arg("x"))
        .def("parameters", &TransformerBlock::parameters, py::return_value_policy::reference_internal);
    
    py::class_<Embedding>(t, "Embedding")
        .def(py::init<const size_t, const size_t>(), py::arg("vocab_size"), py::arg("dim"))
        .def("__call__", &Embedding::forward, py::arg("x"))
        .def("parameters", &Embedding::parameters, py::return_value_policy::reference_internal);
    
    py::class_<PositionalEmbedding>(t, "PositionalEmbedding")
        .def(py::init<const size_t, const size_t>(), py::arg("max_length"), py::arg("dim"))
        .def("__call__", &PositionalEmbedding::forward, py::arg("x"))
        .def("parameters", &PositionalEmbedding::parameters, py::return_value_policy::reference_internal);
    
    py::class_<GPT>(t, "GPT")
        .def(py::init<const size_t, const size_t, const size_t, const size_t, const size_t>(), py::arg("vocab_size"), py::arg("dim"), py::arg("num_heads"), py::arg("num_blocks"), py::arg("max_length"))
        .def("__call__", &GPT::forward, py::arg("x"))
        .def("parameters", &GPT::parameters, py::return_value_policy::reference_internal);

    py::class_<Adam>(t, "Adam")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, float>(), py::arg("parameters"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9, py::arg("beta2") = 0.999, py::arg("epsilon") = 1e-8, py::arg("max_norm") = 1.0)
        .def("step", &Adam::step, py::call_guard<py::gil_scoped_release>())
        .def_property("lr", &Adam::get_lr, &Adam::set_lr)
        .def("get_m", &Adam::get_m)
        .def("get_v", &Adam::get_v)
        .def_property_readonly("t", &Adam::get_t)
        .def("load_state", &Adam::load_state);

    py::class_<DataLoaderIterator>(t, "DataLoaderIterator")
        .def("__iter__", &DataLoaderIterator::iter, py::return_value_policy::reference_internal)
        .def("__next__", &DataLoaderIterator::next);

    py::class_<DataLoaderCSV>(t, "DataLoaderCSV")
        .def(py::init<const std::string&, size_t, size_t, size_t, bool>(), py::arg("filepath"), py::arg("num_examples"), py::arg("example_size"), py::arg("mb_size") = 128, py::arg("shuffle") = true)
        .def("size", &DataLoaderCSV::size)
        .def("num_batches", &DataLoaderCSV::num_batches)
        .def("__iter__", &DataLoaderCSV::iter, py::keep_alive<0, 1>());
}
