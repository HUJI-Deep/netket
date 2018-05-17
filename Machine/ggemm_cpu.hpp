template <typename T>
void channels_first_im2col_cpu(const T* data_im,
                               const int channels, const int height, const int width,
                               const int patch_h, const int patch_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w,
                               T* data_col, const bool round_down, T out_of_bounds_value);

template<typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F1)(Atype, Btype, Ptype), Ctype (*ACC_F1)(Ctype, Ctype),
        Ctype (*COMB_F2)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F2)(Ctype, Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_2ops_cpu(const int M, const int N, const int K,
                    const Atype *A, const Btype *B, Ctype *C,
                    const Ctype Cinit1, const Ctype Cinit2, const Ptype extra_params, const int batch_size = 1,
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

template<typename Dtype>
Dtype softmax(Dtype offset, Dtype data, Dtype max, uint8_t nothing) {
    return std::exp(data + offset - max);
}

template<typename Dtype>
Dtype softmax_activation(Dtype max, Dtype input, uint8_t nothing) {
    return std::log(input) + max;
}