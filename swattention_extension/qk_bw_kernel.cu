#include <torch/extension.h>
#include <cmath>

template <typename scalar_t>
__global__ void qk_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> keys,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_queries,
    int seq_len,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (keys.size(0)* keys.size(1))){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < keys.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < keys.size(3)){
                const int b = x / keys.size(1);
                const int h = x - b * keys.size(1);
                const int pos = y;
                const int start_pos = pos - (kernel_size-1)/2;
                scalar_t updt = scalar_t(0);
                int offset=0;

                #pragma unroll
                for (int current_pos=start_pos; current_pos<(start_pos+kernel_size); ++current_pos){
                    if ((current_pos >= 0) && (current_pos < seq_len)){
                        updt += d_attn_weight[b][h][pos][offset] * keys[b][h][current_pos][z]; 
                    }
                    ++offset;
                }
                d_queries[b][h][pos][z]=updt; 

            }

        }
    }
}


template <typename scalar_t>
__global__ void qk_inverse_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> queries,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_keys,
    int seq_len,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (d_keys.size(0)* d_keys.size(1))){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_keys.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < d_keys.size(3)){
                const int b = x / d_keys.size(1);
                const int h = x - b * d_keys.size(1);
                const int pos = y;
                const int q_start_pos = pos - kernel_size/2;
                const int q_end_pos = pos + 1 + (kernel_size-1)/2;
                scalar_t updt = scalar_t(0);
                int offset=kernel_size;
                #pragma unroll
                for (int current_pos=q_start_pos; current_pos<q_end_pos; ++current_pos){
                    --offset;
                    if ((current_pos >= 0) && (current_pos < seq_len)){
                        updt += d_attn_weight[b][h][current_pos][offset] * queries[b][h][current_pos][z]; 
                    }            
                }
                d_keys[b][h][pos][z]=updt; 

            }

        }
    }
}


std::vector<torch::Tensor> qk_bw_cu(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int seq_len,
    int kernel_size,
    int cuda_threads
){
    TORCH_CHECK((cuda_threads>0)&&(cuda_threads<=1024),"The value of CUDA_NUM_THREADS should between 1 and 1024");
    TORCH_CHECK(queries.size(0) == keys.size(0), "Query and Key should have same Batch_Size");
    TORCH_CHECK(queries.size(1) == keys.size(1), "Query and Key should have same Head Nums");
    TORCH_CHECK(queries.size(2) == keys.size(2), "Query and Key should have same Token Nums");
    TORCH_CHECK(queries.size(3) == keys.size(3), "Query and Key should have same Head Dims");
    const int B= queries.size(0), N = queries.size(1), L = queries.size(2), C = queries.size(3);

    const int attention_span = kernel_size;
    const int DIMTHREADS = min(cuda_threads, C);
    const int PIXELTHREADS = min(int(cuda_threads / DIMTHREADS), L);
    const int BATCHTHREADS = max(1, cuda_threads / (PIXELTHREADS * DIMTHREADS));

    torch::Tensor d_queries = torch::empty({B, N, L, C}, queries.options());
    torch::Tensor d_keys = torch::empty({B, N, L, C}, keys.options());

    const dim3 threads(BATCHTHREADS, PIXELTHREADS, DIMTHREADS);
    const dim3 blocks(((B*N)+threads.x-1)/threads.x, (L+threads.y-1)/threads.y, (C+threads.z-1)/threads.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(queries.scalar_type(), "qk_bw_cu", 
    ([&] {
        qk_bw_kernel<scalar_t><<<blocks, threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),        
            seq_len,
            kernel_size
        );
        qk_inverse_bw_kernel<scalar_t><<<blocks, threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),        
            seq_len,
            kernel_size
        );
    }));

    return {d_queries,d_keys};
}


