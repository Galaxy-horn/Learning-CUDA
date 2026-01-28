#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
// 并行归约
template <typename T>
__global__ void traceKernel(const T* input, T* output, size_t rows, size_t cols, size_t diag_len) {
  extern __shared__ char shared_mem[];
  T* sdata = reinterpret_cast<T*>(shared_mem);
  
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  sdata[tid] = (i < diag_len) ? input[i * cols + i] : T(0);
  __syncthreads();
  
  // 共享内存归约
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  
  if (tid == 0) output[blockIdx.x] = sdata[0];
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  size_t diag_len = (rows < cols) ? rows : cols;
  
  T* d_input;
  RUNTIME_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice));
  
  int threads = 256;
  int blocks = (diag_len + threads - 1) / threads;
  
  T* d_output;
  RUNTIME_CHECK(cudaMalloc(&d_output, blocks * sizeof(T)));
  
  traceKernel<<<blocks, threads, threads * sizeof(T)>>>(d_input, d_output, rows, cols, diag_len);
  
  std::vector<T> h_output(blocks);
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, blocks * sizeof(T), cudaMemcpyDeviceToHost));
  
  T result = T(0);
  for (int i = 0; i < blocks; i++) result += h_output[i];
  
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_output));
  
  return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
// Flash Attention核函数
template <typename T>
__global__ void flashAttentionKernel(const T* Q, const T* K, const T* V, T* O,
                                      int batch_size, int tgt_len, int src_len,
                                      int q_heads, int kv_heads, int head_dim, bool is_causal) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i >= tgt_len) return;
  
  int kv_h = h * kv_heads / q_heads;
  float scale = 1.0f / sqrtf((float)head_dim);
  
  int q_offset = ((b * tgt_len + i) * q_heads + h) * head_dim;
  int k_base = ((b * src_len + 0) * kv_heads + kv_h) * head_dim;
  int o_offset = q_offset;
  
  int max_j = is_causal ? (i + 1) : src_len;
  
  float max_val = -INFINITY;
  for (int j = 0; j < max_j; j++) {
    float score = 0.0f;
    int kv_offset = ((b * src_len + j) * kv_heads + kv_h) * head_dim;
    for (int d = 0; d < head_dim; d++) {
      score += (float)Q[q_offset + d] * (float)K[kv_offset + d];
    }
    score *= scale;
    max_val = fmaxf(max_val, score);
  }
  
  float sum_exp = 0.0f;
  for (int j = 0; j < max_j; j++) {
    float score = 0.0f;
    int kv_offset = ((b * src_len + j) * kv_heads + kv_h) * head_dim;
    for (int d = 0; d < head_dim; d++) {
      score += (float)Q[q_offset + d] * (float)K[kv_offset + d];
    }
    score *= scale;
    sum_exp += expf(score - max_val);
  }
  
  for (int d = 0; d < head_dim; d++) {
    float out_val = 0.0f;
    for (int j = 0; j < max_j; j++) {
      float score = 0.0f;
      int kv_offset = ((b * src_len + j) * kv_heads + kv_h) * head_dim;
      for (int d2 = 0; d2 < head_dim; d2++) {
        score += (float)Q[q_offset + d2] * (float)K[kv_offset + d2];
      }
      score *= scale;
      float attn_weight = expf(score - max_val) / sum_exp;
      out_val += attn_weight * (float)V[kv_offset + d];
    }
    O[o_offset + d] = (T)out_val;
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
  size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
  size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
  
  T *d_q, *d_k, *d_v, *d_o;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, q_size * sizeof(T)));
  
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
  
  int threads = 256;
  int blocks_x = (target_seq_len + threads - 1) / threads;
  dim3 blocks(blocks_x, query_heads, batch_size);
  
  flashAttentionKernel<<<blocks, threads>>>(d_q, d_k, d_v, d_o,
                                             batch_size, target_seq_len, src_seq_len,
                                             query_heads, kv_heads, head_dim, is_causal);
  
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_size * sizeof(T), cudaMemcpyDeviceToHost));
  
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
