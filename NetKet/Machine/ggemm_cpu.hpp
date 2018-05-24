/**
 * A helper function to calculate the output dimension's size give the original
 * size of the image, padding, patch size and stride.
 * @param  image_size The size of the dimension in the original image
 * @param  pad_size   The amount of padding to apply to the original image
 * @param  patch_size The size of the dimension in the patch taken from the image
 * @param  stride     The patch's stride over the original image
 * @param  round_down Whether to round down or up when calculating the size
 * @return            The output size of the patch image
 * @remarks round_down can be used to control pooling/conv style im2col method.
 */
inline int dimension_out_size(const int image_size, const int pad_size,
                              const int patch_size,
                              const int stride, const bool round_down) {
    if (round_down) {
        return (image_size + 2 * pad_size - patch_size) / stride + 1;
    } else {
        return static_cast<int>(std::ceil(
                static_cast<float>(image_size + 2 * pad_size - patch_size) /
                stride)) + 1;
    }
}

template<typename T>
void channels_first_im2col_cpu(const T *data_im,
                               const int channels, const int height,
                               const int width,
                               const int patch_h, const int patch_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w,
                               T *data_col, const bool round_down,
                               T out_of_bounds_value) {

    const int height_col = dimension_out_size(height, pad_h, patch_h, stride_h,
                                              round_down);
    const int width_col = dimension_out_size(width, pad_w, patch_w, stride_w,
                                             round_down);
    const int patch_c = channels;
    const int patch_col = patch_c * patch_h * patch_w;
    for (int p = 0; p < patch_col; ++p) {
        const int c_offset = p % patch_c;
        const int w_offset = (p / patch_c) % patch_w;
        const int h_offset = (p / patch_c) / patch_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                const int h_pad = h * stride_h - pad_h + h_offset;
                const int w_pad = w * stride_w - pad_w + w_offset;
                const int c_pad = c_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
                    && c_pad >= 0 && c_pad < channels) {
                    data_col[(p * height_col + h) * width_col + w] =
                            data_im[(c_pad * height + h_pad) * width + w_pad];
                } else {
                    data_col[(p * height_col + h) * width_col +
                             w] = out_of_bounds_value;
                }
            }
        }
    }
}

template<typename T>
void simnets_tf_set(const int N, const T alpha, T *Y) {
    if (alpha == T{0}) {
        memset(Y, 0, sizeof(T) * N);
        return;
    }
    for (int i = 0; i < N; ++i) {
        Y[i] = alpha;
    }
}

template<typename T>
void channels_first_col2im_cpu(const T *data_col,
                               const int channels, const int height, const int width,
                               const int patch_h, const int patch_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w,
                               T *data_im, const bool round_down) {
    simnets_tf_set(height * width * channels, T(0), data_im);
    const int height_col = dimension_out_size(height, pad_h, patch_h, stride_h, round_down);
    const int width_col = dimension_out_size(width, pad_w, patch_w, stride_w, round_down);
    const int patch_c = channels;
    const int patch_col = patch_c * patch_h * patch_w;
    for (int p = 0; p < patch_col; ++p) {
        const int c_offset = p % patch_c;
        const int w_offset = (p / patch_c) % patch_w;
        const int h_offset = p / patch_c / patch_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                const int h_pad = h * stride_h - pad_h + h_offset;
                const int w_pad = w * stride_w - pad_w + w_offset;
                const int c_pad = c_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
                    && c_pad >= 0 && c_pad < channels) {
                    data_im[(c_pad * height + h_pad) * width + w_pad] +=
                            data_col[(p * height_col + h) * width_col + w];
                }
            }
        }
    }
}


template<typename Dtype>
inline Dtype ceiled_div(const Dtype a, const Dtype b) {
    return (a / b) + ((a % b) > 0);
}

enum {
    BLOCK_WIDTH = 16,
    BLOCK_HEIGHT = 16
};

inline int ggemm_padded_output_size(const int M, const int N) {
    int newN = ceiled_div<int>(N, BLOCK_WIDTH) * BLOCK_WIDTH;
    int newM = ceiled_div<int>(M, BLOCK_HEIGHT) * BLOCK_HEIGHT;
    return newN * newM - M * N;
}

template<typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F1)(Atype, Btype, Ptype), Ctype (*ACC_F1)(Ctype, Ctype),
        Ctype (*COMB_F2)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F2)(Ctype,
                                                                      Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_2ops_cpu(const int M, const int N, const int K,
                    const Atype *A, const Btype *B, Ctype *C,
                    const Ctype Cinit1, const Ctype Cinit2,
                    const Ptype extra_params, const int batch_size = 1,
                    int A_batch_stride = -1, int B_batch_stride = -1,
                    int C_batch_stride = -1) {
    if (BATCH_A_ACTIVE) {
        if (A_batch_stride < 0) {
            A_batch_stride = M * K;
        }
    }
    if (BATCH_B_ACTIVE) {
        if (B_batch_stride < 0) {
            B_batch_stride = N * K;
        }
    }
    if (BATCH_C_ACTIVE) {
        if (C_batch_stride < 0) {
            C_batch_stride = M * N;
        }
    }
    for (int r = 0; r < batch_size; ++r) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                Ctype sum1 = Cinit1;
                for (int k = 0; k < K; ++k) {
                    Atype a = A[i * K + k];
                    Btype b = B[k * N + j];
                    Ctype temp = COMB_F1(a, b, extra_params);
                    sum1 = ACC_F1(sum1, temp);
                }
                Ctype sum2 = Cinit2;
                for (int k = 0; k < K; ++k) {
                    Atype a = A[i * K + k];
                    Btype b = B[k * N + j];
                    Ctype temp = COMB_F2(a, b, sum1, extra_params);
                    sum2 = ACC_F2(sum2, temp);
                }
                Ctype final_value;
                if (APPLY_ON_C) {
                    final_value = APPLY_F(sum1, sum2, extra_params);
                } else {
                    final_value = sum2;
                }
                if (ADD_TO_C) {
                    C[i * N + j] = ACC_F2(C[i * N + j], final_value);
                } else {
                    C[i * N + j] = final_value;
                }
            }
        }
        if (BATCH_A_ACTIVE) {
            A += A_batch_stride;
        }
        if (BATCH_B_ACTIVE) {
            B += B_batch_stride;
        }
        if (BATCH_C_ACTIVE) {
            C += C_batch_stride;
        }
    }
}

template <bool TRANSPOSE_A, bool TRANSPOSE_B,
        typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F)(Ctype, Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_readc_cpu(const int M, const int N, const int K,
                     const Atype* A, const Btype* B, const Ctype* Cin, Ctype* Cout,
                     const Ctype Cinit, const Ptype extra_params, const int batch_size = 1,
                     int A_batch_stride = -1, int B_batch_stride = -1, int C_batch_stride = -1) {
    if (BATCH_A_ACTIVE) {
        if (A_batch_stride < 0) {
            A_batch_stride = M * K;
        }
    }
    if (BATCH_B_ACTIVE) {
        if (B_batch_stride < 0) {
            B_batch_stride = N * K;
        }
    }
    if (BATCH_C_ACTIVE) {
        if (C_batch_stride < 0) {
            C_batch_stride = M * N;
        }
    }
    for (int r = 0; r < batch_size; ++r) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                Ctype sum = Cinit;
                Ctype c = Cin[i * N + j];
                for (int k = 0; k < K; ++k) {
                    Atype a = A[TRANSPOSE_A ? k * M + i : i * K + k];
                    Btype b = B[TRANSPOSE_B ? j * K + k : k * N + j];
                    sum = ACC_F(sum, COMB_F(a, b, c, extra_params));
                }
                if (APPLY_ON_C) {
                    sum = APPLY_F(sum, extra_params);
                }
                if (ADD_TO_C) {
                    Cout[i * N + j] = ACC_F(Cout[i * N + j], sum);
                } else {
                    Cout[i * N + j] = sum;
                }
            }
        }
        if (BATCH_A_ACTIVE) {
            A += A_batch_stride;
        }
        if (BATCH_B_ACTIVE) {
            B += B_batch_stride;
        }
        if (BATCH_C_ACTIVE) {
            Cin += C_batch_stride;
            Cout += C_batch_stride;
        }
    }
}

template<typename Dtype>
Dtype ggemm_mul(Dtype a, Dtype b) {
    return a * b;
}

template<typename Dtype, typename Ntype>
Dtype ggemm_add(Dtype a, Dtype b, Ntype nothing) {
    return a + b;
}

template<typename Dtype>
Dtype ggemm_add(Dtype a, Dtype b) {
    return a + b;
}

template<typename Dtype>
Dtype ggemm_max(Dtype a, Dtype b) {
    return std::max(a, b);
}

template<>
std::complex<double> ggemm_max(std::complex<double> a, std::complex<double> b) {
    return std::complex<double>(std::max(a.real(), b.real()),
                                std::max(a.imag(), b.imag()));
}

template<typename Dtype>
Dtype softmax(Dtype offset, Dtype data, Dtype max, uint8_t nothing) {
    return std::exp(data + offset - max);
}

template<typename Dtype>
Dtype softmax_activation(Dtype max, Dtype input, uint8_t nothing) {
    return std::log(input) + max;
}

template<typename T>
T mex_backward_bottom_finite(T offset, Eigen::Matrix<T, 2, 1> top_data, T data,
                             Eigen::Matrix<T, 2, 1> extra) {
    return top_data(1, 0) * std::exp(extra(0, 0) * (data + offset - top_data(0, 0)) + extra(1, 0));
}

template<typename T>
T mex_backward_offsets_finite(Eigen::Matrix<T, 2, 1> top_data, T data, T offset,
                                  Eigen::Matrix<T, 2, 1> extra) {
    return mex_backward_bottom_finite(offset, top_data, data, extra);
}

template<typename T, typename P>
T no_op(T a, P nothing) {
    return a;
}

template <typename T> void interlace_cpu(const int N, const T* a, const T* b,
                                         typename Eigen::Matrix<T, 2, 1>* o) {
    for (int i = 0; i < N; ++i)
    {
        o[i] = Eigen::Matrix<T, 2, 1>();
        o[i] << a[i], b[i];
    }
}