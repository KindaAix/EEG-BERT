#include <torch/extension.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void rpb_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight,//B,H,L,span
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> d_rpb,
    int seq_len,
    int kernel_size,
    int d_rpb_numel
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x <  d_attn_weight.size(1)){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_attn_weight.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < d_attn_weight.size(3)){
                const int pos = y;
                const int start_pos = pos - (kernel_size-1)/2;
                const int current_pos = start_pos + z;
                scalar_t updt = scalar_t(0);
                if ((current_pos >= 0) && (current_pos < seq_len)){
                    #pragma unroll
                    for (int b=0; b < d_attn_weight.size(0); ++b)
                        updt += d_attn_weight[b][x][pos][z];
                }
                const int index=x*d_rpb.size(1)+z;
                at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(updt), true);
            }

        }
    }
}

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


std::vector<torch::Tensor> qk_rpb_bw_cu(
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
    const int QK_DIMTHREADS = min(cuda_threads, C);
    const int QK_PIXELTHREADS = min(int(cuda_threads / QK_DIMTHREADS), L);
    const int QK_BATCHTHREADS = max(1, cuda_threads / (QK_PIXELTHREADS * QK_DIMTHREADS));

    const int RPB_KERNELTHREADS = min(cuda_threads, attention_span);
    const int RPB_PIXELTHREADS = min(int(cuda_threads / RPB_KERNELTHREADS), L);
    const int RPB_HEADTHREADS = max(1, cuda_threads / (RPB_PIXELTHREADS * RPB_KERNELTHREADS));

    torch::Tensor d_queries = torch::empty({B, N, L, C}, queries.options());
    torch::Tensor d_keys = torch::empty({B, N, L, C}, keys.options());
    torch::Tensor d_rpb = torch::zeros({N, attention_span}, keys.options());
    const int d_rpb_numel=N*attention_span;

    const dim3 rpb_threads(RPB_HEADTHREADS, RPB_PIXELTHREADS, RPB_KERNELTHREADS);
    const dim3 rpb_blocks((N+rpb_threads.x-1)/rpb_threads.x,(L+rpb_threads.y-1)/rpb_threads.y, (attention_span+rpb_threads.z-1)/rpb_threads.z);  

    const dim3 qk_threads(QK_BATCHTHREADS, QK_PIXELTHREADS, QK_DIMTHREADS);
    const dim3 qk_blocks(((B*N)+qk_threads.x-1)/qk_threads.x, (L+qk_threads.y-1)/qk_threads.y, (C+qk_threads.z-1)/qk_threads.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(queries.scalar_type(), "qk_bw_cu", 
    ([&] {
        rpb_bw_kernel<scalar_t><<<rpb_blocks, rpb_threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_rpb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),        
            seq_len,
            kernel_size,
            d_rpb_numel
        );
        qk_bw_kernel<scalar_t><<<qk_blocks, qk_threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),        
            seq_len,
            kernel_size
        );
        qk_inverse_bw_kernel<scalar_t><<<qk_blocks, qk_threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            seq_len,
            kernel_size
        );
    }));

    return {d_queries,d_keys,d_rpb};
}


