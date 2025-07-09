//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <array>
#include <numeric>
#include <stdexcept>

namespace utec::algebra
{
    template <typename T, size_t Rank>
    class Tensor
    {

    public:
        std::array<std::size_t, Rank> shape_;
        std::vector<T> data_; // almacena los datos del tensor en un vector lineal

        // iteradores

        auto cbegin() const { return data_.cbegin(); }
        auto cend() const { return data_.cend(); }

        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }

        // constructores

        Tensor(const std::array<std::size_t, Rank> &shape)
        {
            this->shape_ = shape;
            std::size_t size_data = 1;
            for (std::size_t i = 0; i < Rank; ++i)
            {
                size_data *= shape[i]; // calculamos el tamaño total del tensor (producto de las dimensiones)
            }
            data_.resize(size_data); // redimensionamos el vector de datos al tamaño total
        }

        Tensor()
        {
            std::array<size_t, Rank> shape;
            for (std::size_t i = 0; i < Rank; ++i)
            {
                shape[i] = 1; // inicializamos las dimensiones con 1
            }
            *this = Tensor(shape);
        }

        template <typename... Dims>
        Tensor(Dims... dims)
        {
            constexpr std::size_t r = sizeof...(Dims);

            std::array<size_t, Rank> shape;

            if constexpr (r == Rank)
            {
                shape = {static_cast<std::size_t>(dims)...};
                *this = Tensor(shape);
            }
            else
                throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
        }

        Tensor(std::initializer_list<size_t> shape)
        {
            for (std::size_t i = 0; i < Rank; ++i)
            {
                shape_[i] = shape.begin()[i];
            }
            std::size_t size_data = 1;
            for (std::size_t i = 0; i < Rank; ++i)
            {
                size_data *= shape_[i]; // calculamos el tamaño total del tensor (producto de las dimensiones)
            }
            data_.resize(size_data); // redimensionamos el vector de datos al tamaño total
        }

        // acceso variádico (único):
        template <typename... Idxs>
        T &operator()(Idxs... idxs)
        {
            // verificamos que el número de índices es igual al rango

            if (sizeof...(Idxs) != Rank)
                throw std::invalid_argument("Number of indexes does not match with rank");

            // verificamos que los índices están dentro de los límites
            std::array<std::size_t, Rank> index = {static_cast<std::size_t>(idxs)...};
            /*
            for(std::size_t i = 0; i < Rank; ++i) {
                if(index[i]>=shape_[i]) {
                    throw std::out_of_range("Index is out of range");
                }
            }
            */
            // calculamos el índice lineal en el vector de datos
            std::size_t linear_index = 0;
            std::size_t multiplier = 1;
            for (int i = Rank - 1; i >= 0; --i)
            {
                linear_index += index[i] * multiplier;
                multiplier *= shape_[i];
            }
            return data_[linear_index]; // devolvemos la referencia al dato en el vector
        }

        template<typename TensorType, typename Func>
        auto apply(const TensorType& tensor, Func&& f) {
            TensorType result = tensor;
            for (auto& v : result.data_) {
                v = f(v);
            }
            return result;
        }


        template <typename... Idxs>
        const T &operator()(Idxs... idxs) const
        {
            // este se usa para asignar
            // verificamos que el número de índices es igual al rango

            if (sizeof...(Idxs) != Rank)
                throw std::invalid_argument("Number of indexes does not match rank");

            std::array<std::size_t, Rank> index = {static_cast<std::size_t>(idxs)...};

            // verificamos que los índices están dentro de los límites
            /*
            for(std::size_t i = 0; i < Rank; ++i) {
            if(index[i] >= shape_[i]) throw std::out_of_range("Index is out of range");
            }
            */
            std::size_t linear_index = 0;
            std::size_t multiplier = 1;
            for (int i = Rank - 1; i >= 0; --i)
            {
                linear_index += index[i] * multiplier;
                multiplier *= shape_[i];
            }
            return data_[linear_index]; // devolvemos la referencia al dato en el vector
        }

        // información de dimensiones:
        std::size_t size() const noexcept
        {
            return data_.size(); // devolvemos el tamaño del vector de datos
        }

        const std::array<std::size_t, Rank> &shape() const noexcept
        {
            return shape_; // devolvemos la referencia al array de dimensiones
        }

        template <typename... Args>
        void reshape(Args... args)
        {
            constexpr std::size_t r = sizeof...(Args);

            std::array<std::size_t, Rank> new_shape;

            if constexpr (r == Rank)
            {
                new_shape = {static_cast<std::size_t>(args)...};
                this->reshape(new_shape);
            }
            else
                throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
        }

        void reshape(const std::array<std::size_t, Rank> &new_shape)
        {
            std::size_t total_elems = data_.size();
            std::size_t nuevos_elems = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>{});
            this->shape_ = new_shape;
            data_.resize(nuevos_elems);
        }

        // modificación masiva:
        void fill(const T &value) noexcept
        {
            std::fill(data_.begin(), data_.end(), value);
        }

        // broadcasting (funciones)

        std::array<std::size_t, Rank> broadcasting(const Tensor &other) const
        {
            std::array<std::size_t, Rank> newShape;

            for (std::size_t i = 0; i < Rank; ++i)
            {
                if (shape_[i] == other.shape_[i])
                    newShape[i] = shape_[i];
                else if (shape_[i] == 1 && other.shape_[i] != 1)
                    newShape[i] = other.shape_[i];
                else if (shape_[i] != 1 && other.shape_[i] == 1)
                    newShape[i] = shape_[i];
                else
                    throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }
            return newShape;
        }

        std::vector<T> broadcast_data(const std::vector<T> &data, const std::array<std::size_t, Rank> &from_shape, const std::array<std::size_t, Rank> &to_shape) const
        {

            if (from_shape == to_shape)
                return data;

            std::vector<T> res;
            std::size_t total_size = 1;
            for (auto dims : to_shape)
                total_size *= dims;
            res.resize(total_size);

            std::array<std::size_t, Rank> from_strides;
            std::size_t stride = 1;
            for (int i = Rank - 1; i >= 0; --i)
            {
                from_strides[i] = (from_shape[i] == 1) ? 0 : stride;
                stride *= from_shape[i];
            }

            for (std::size_t i = 0; i < total_size; ++i)
            {
                std::size_t idx = i;
                std::size_t from_index = 0;

                for (int dim = Rank - 1; dim >= 0; --dim)
                {
                    std::size_t coord = idx % to_shape[dim];
                    idx /= to_shape[dim];

                    std::size_t from_coord = (from_shape[dim] == 1) ? 0 : coord;
                    from_index += from_coord * from_strides[dim];
                }
                res[i] = data[from_index];
            }
            return res;
        }

        // operadores

        Tensor &operator=(std::initializer_list<T> data)
        {
            if (data_.size() != data.size())
                throw std::invalid_argument("Data size does not match tensor size");

            std::copy(data.begin(), data.end(), data_.begin());
            return *this;
        }

        Tensor operator+(const Tensor &other) const
        {

            auto shape = broadcasting(other);
            auto data1 = broadcast_data(data_, shape_, shape);
            auto data2 = broadcast_data(other.data_, other.shape_, shape);

            Tensor<T, Rank> res(shape);
            for (std::size_t i = 0; i < data1.size(); ++i)
            {
                res.data_[i] = data1[i] + data2[i];
            }
            return res;
        }

        Tensor operator+(const T &scalar) const
        {
            Tensor<T, Rank> res(this->shape_);
            for (std::size_t i = 0; i < data_.size(); ++i)
            {
                res.data_[i] = data_[i] + scalar;
            }
            return res;
        }

        Tensor operator-(const Tensor &other) const
        {
            auto shape = broadcasting(other);
            auto data1 = broadcast_data(data_, shape_, shape);
            auto data2 = broadcast_data(other.data_, other.shape_, shape);

            Tensor<T, Rank> res(shape);
            for (std::size_t i = 0; i < data1.size(); ++i)
            {
                res.data_[i] = data1[i] - data2[i];
            }
            return res;
        }

        Tensor operator-(const T &scalar) const
        {
            Tensor<T, Rank> res(this->shape_);
            for (std::size_t i = 0; i < data_.size(); ++i)
            {
                res.data_[i] = data_[i] - scalar;
            }
            return res;
        }

        Tensor operator*(const Tensor &other) const
        { // broadcasting en dim de tamaño 1

            // comprobar que se pueden multiplicar dos matrices de diferentes dimensiones
            if (Rank == 2)
            {
                if (shape_[1] != other.shape_[0])
                    throw std::invalid_argument("Matrices not compatible for multiplication");
                Tensor<T, 2> res(shape_[0], other.shape_[1]);
                for (std::size_t i = 0; i < res.shape_[0]; ++i)
                    for (std::size_t j = 0; j < res.shape_[1]; ++j)
                    {
                        for (std::size_t k = 0; k < shape_[1]; ++k)
                        {
                            res(i, j) += (*this)(i, k) * other(k, j);
                        }
                    }
                return res;
            }

            // comprobando que se puede broadcastear
            if (shape_.size() != other.shape_.size())
            {
                throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }

            // el tamaño del tensor resultado
            auto shape = broadcasting(other);
            Tensor<T, Rank> res(shape);

            auto data1 = broadcast_data(data_, shape_, shape);
            auto data2 = broadcast_data(other.data_, other.shape_, shape);

            for (std::size_t i = 0; i < data1.size(); ++i)
            {
                res.data_[i] = data1[i] * data2[i];
            }
            return res;
        }

        Tensor operator*(const T &scalar) const
        {
            Tensor<T, Rank> res(shape_);
            for (size_t i = 0; i < data_.size(); ++i)
            {
                res.data_[i] = data_[i] * scalar;
            }
            return res;
        }

        Tensor operator/(const T &scalar) const
        {
            Tensor<T, Rank> res(shape_);
            for (size_t i = 0; i < data_.size(); ++i)
            {
                res.data_[i] = data_[i] / scalar;
            }
            return res;
        }

        T &at(const std::array<std::size_t, Rank> &index)
        {
            std::size_t linear_index = 0;
            std::size_t multiplier = 1;
            for (int i = Rank - 1; i >= 0; --i)
            {
                linear_index += index[i] * multiplier;
                multiplier *= shape_[i];
            }
            return data_[linear_index];
        }

        const T &at(const std::array<std::size_t, Rank> &index) const
        {
            std::size_t linear_index = 0;
            std::size_t multiplier = 1;
            for (int i = Rank - 1; i >= 0; --i)
            {
                linear_index += index[i] * multiplier;
                multiplier *= shape_[i];
            }
            return data_[linear_index];
        }
    };

    // utilizades (solo para rank>=2)_
    template <typename T, size_t Rank>
    Tensor<T, Rank> transpose_2d(const Tensor<T, Rank> &tensor)
    {
        if (Rank < 2)
            throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
        auto shape = tensor.shape_;
        auto shape2 = shape;
        std::swap(shape2[Rank - 1], shape2[Rank - 2]);

        Tensor<T, Rank> res(shape2);

        std::array<std::size_t, Rank> idx = {};
        std::array<std::size_t, Rank> idx_t = {};

        std::size_t total = 1;
        for (auto s : shape)
            total *= s;

        for (std::size_t flat_idx = 0; flat_idx < total; ++flat_idx)
        {
            std::size_t rem = flat_idx;
            for (int i = Rank - 1; i >= 0; --i)
            {
                idx[i] = rem % shape[i];
                rem /= shape[i];
            }
            idx_t = idx;
            std::swap(idx_t[Rank - 1], idx_t[Rank - 2]);

            std::apply(
                [&](auto &&...args_t)
                {
                    std::apply(
                        [&](auto &&...args)
                        {
                            res(args_t...) = tensor(args...);
                        },
                        idx);
                },
                idx_t);
        }
        return res;
    }

    template <typename T, size_t Rank>
    void print_tensor(std::ostream &os, const Tensor<T, Rank> &tensor, std::size_t dim = 0, std::size_t offset = 0)
    {
        if (dim < Rank - 1)
            os << "{\n";

        std::size_t block_size = 1;
        for (std::size_t i = dim + 1; i < Rank; ++i)
            block_size *= tensor.shape_[i];

        for (std::size_t i = 0; i < tensor.shape_[dim]; ++i)
        {
            if (dim == Rank - 1)
            {
                for (std::size_t j = 0; j < tensor.shape_[dim]; ++j)
                    os << tensor.data_[offset + j] << " ";
                os << "\n";
                break;
            }
            else
            {
                print_tensor(os, tensor, dim + 1, offset + i * block_size);
            }
        }

        if (dim < Rank - 1)
            os << "}\n";
    }

    template <typename T, size_t Rank>
    std::ostream &operator<<(std::ostream &os, const Tensor<T, Rank> &tensor)
    {
        print_tensor(os, tensor, 0, 0); // iniciamos la impresión desde la primera dimensión
        return os;
    }

    template <typename T, size_t Rank>
    Tensor<T, Rank> operator+(const T &scalar, const Tensor<T, Rank> &tensor)
    {
        return tensor + scalar;
    }

    template <typename T, size_t Rank>
    Tensor<T, Rank> matrix_product(const Tensor<T, Rank> &A, const Tensor<T, Rank> &B)
    {
        // verificar que las dimensiones sean compatibles
        if (A.shape_[Rank - 1] != B.shape_[Rank - 2])
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        // verificar que las dimensiones del batch matcheen
        for (int i = 0; i < Rank - 2; ++i)
        {
            if (A.shape_[i] != B.shape_[i])
                throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
        }
        std::array<size_t, Rank> shape_result = A.shape_;
        shape_result[Rank - 1] = B.shape_[Rank - 1]; // el nuevo N

        Tensor<T, Rank> result(shape_result);

        std::array<size_t, Rank> idxA = {};
        std::array<size_t, Rank> idxB = {};
        std::array<size_t, Rank> idxR = {};

        const size_t K = A.shape_[Rank - 1]; // tamaño común

        // Recorremos todos los índices del resultado
        std::array<size_t, Rank> shape_loop = shape_result;

        std::vector<size_t> counters(Rank, 0);
        while (true)
        {
            // armamos los índices
            for (size_t i = 0; i < Rank; ++i)
                idxR[i] = counters[i];

            T sum = T{};
            for (size_t k = 0; k < K; ++k)
            {
                for (size_t i = 0; i < Rank; ++i)
                {
                    idxA[i] = idxR[i];
                    idxB[i] = idxR[i];
                }
                idxA[Rank - 1] = k;
                idxB[Rank - 2] = k;

                T a_val = std::apply([&](auto... args)
                                     { return A(args...); }, idxA);
                T b_val = std::apply([&](auto... args)
                                     { return B(args...); }, idxB);
                sum += a_val * b_val;
            }

            std::apply([&](auto... args)
                       { result(args...) = sum; }, idxR);

            size_t i = Rank;
            while (i-- > 0)
            {
                if (++counters[i] < shape_loop[i])
                    break;
                counters[i] = 0;
            }
            if (i == static_cast<size_t>(-1))
                break;
        }

        return result;
    }

    template<typename TensorType, typename Func>
    TensorType apply(const TensorType& tensor, Func&& f) {
    TensorType result = tensor;
    for (auto& v : result.data_) {
        v = f(v);
    }
    return result;
}

} // namespace utec::algebra

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
