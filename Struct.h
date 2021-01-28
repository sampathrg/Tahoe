#ifndef STRUCT_H
#define STRUCT_H
#include "cuda_base.h"
#include <time.h>
#include <string>
#include <math.h> 
#include "simhash.h"

int adaptive_format_number = 0;

int selected_algorithm = 0;

__host__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__host__ __device__ __forceinline__ int tree_num_nodes(int depth) {
	  return (1 << (depth + 1)) - 1;
}

__host__ __device__ __forceinline__ int forest_num_nodes(int num_trees, int depth) {
  return num_trees * tree_num_nodes(depth);
}

enum algo_t {
  NAIVE,
  TREE_REORG,
  BATCH_TREE_REORG
};

enum strategy_t {
  SHARED_DATA,
  SHARED_FOREST,
  SPLIT_FOREST,
  SPLIT_FOREST_SHARED_DATA
};


enum output_t {
  RAW = 0x0,
  AVG = 0x1,
  SIGMOID = 0x10,
  THRESHOLD = 0x100,
};

struct dense_node_t {
  float weight;
  float val;
  int bits;
};

struct sparse_node_t {
  float val;
  int bits;
  int left_idx;
};


static const int FID_MASK = (1 << 30) - 1;
static const int DEF_LEFT_MASK = 1 << 30;
static const int IS_LEAF_MASK = 1 << 31;

static const int FID_MASK_INT = (1 << 29) - 1;
static const int DEF_LEFT_MASK_INT = 1 << 29;
static const int IS_LEAF_MASK_INT = 1 << 30;
static const int EXCHANGE_MASK_INT = 1 << 31;

static const int FID_MASK_SHORT = (1 << 13) - 1;
static const int DEF_LEFT_MASK_SHORT = 1 << 13;
static const int IS_LEAF_MASK_SHORT = 1 << 14;
static const int EXCHANGE_MASK_SHORT = 1 << 15;

static const int FID_MASK_CHAR = (1 << 5) - 1;
static const int DEF_LEFT_MASK_CHAR = 1 << 5;
static const int IS_LEAF_MASK_CHAR = 1 << 6;
static const int EXCHANGE_MASK_CHAR = 1 << 7;


void encode_node_adaptive(std::vector<float> &values_reorder_h, std::vector<int> &fids_reorder_h, bool* defaults_reorder_h, bool* is_leafs_reorder_h, bool* exchanges_reorder_h, float* bits_values_h, char* bits_char_h, short int* bits_short_h, int* bits_int_h, int bits_length, int length)
{
	for(int i=0; i<length; i++)
	{
		bits_values_h[i] = values_reorder_h[i];
		if(bits_length == 1)
		{
			bits_char_h[i] = (fids_reorder_h[i] & FID_MASK_CHAR) | (defaults_reorder_h[i] ? DEF_LEFT_MASK_CHAR : 0) | (is_leafs_reorder_h[i] ? IS_LEAF_MASK_CHAR : 0)
							| (exchanges_reorder_h[i] ? EXCHANGE_MASK_CHAR : 0);
		}
		if(bits_length == 2)
		{
			bits_short_h[i] = (fids_reorder_h[i] & FID_MASK_SHORT) | (defaults_reorder_h[i] ? DEF_LEFT_MASK_SHORT : 0) 
			                | (is_leafs_reorder_h[i] ? IS_LEAF_MASK_SHORT : 0) | (exchanges_reorder_h[i] ? EXCHANGE_MASK_SHORT : 0);
		}
		if(bits_length == 4)
		{
			bits_int_h[i] = (fids_reorder_h[i] & FID_MASK_INT) | (defaults_reorder_h[i] ? DEF_LEFT_MASK_INT : 0) | (is_leafs_reorder_h[i] ? IS_LEAF_MASK_INT : 0)
							| (exchanges_reorder_h[i] ? EXCHANGE_MASK_INT : 0);
		}
	}
}




void encode_node(dense_node_t* node, int fid, float value, bool def_left, float weight, bool is_leaf)
{
	node->weight = weight;
	node->val = value;
	node->bits = (fid & FID_MASK) | (def_left ? DEF_LEFT_MASK : 0) | (is_leaf ? IS_LEAF_MASK : 0);
}

void dense_node_decode(const dense_node_t* n, float* value, float* weight,
                       int* fid, bool* def_left, bool* is_leaf) {
  *value = n->val;
  *weight = n->weight;
  *fid = n->bits & FID_MASK;
  *def_left = n->bits & DEF_LEFT_MASK;
  *is_leaf = n->bits & IS_LEAF_MASK;
}


struct TahoeTestParams {
  // input data parameters
  int num_rows;
  int num_cols;
  float nan_prob;
  // forest parameters
  int depth;
  int num_trees;
  float leaf_prob;
  // output parameters
  output_t output;
  float threshold;
  float global_bias;
  // runtime parameters
  algo_t algo;
  int seed;
  float tolerance;
  strategy_t strategy;

  char input_model_file[1024];
  char input_data_file[1024];
  float missing;
};


// predict_params are parameters for prediction
struct predict_params {
  // Model parameters.
  int num_cols;
  algo_t algo;
  strategy_t strategy;
  int max_items;  // only set and used by infer()

  // Data parameters.
  float* preds;
  const float* data;
  size_t num_rows;

  // Other parameters.
  int max_shm;

  float missing;
};


/** forest_params_t are the trees to initialize the predictor */
struct forest_params_t {
  // total number of nodes; ignored for dense forests
  int num_nodes;
  // maximum depth; ignored for sparse forests
  int depth;
  // ntrees is the number of trees
  int num_trees;
  // num_cols is the number of columns in the data
  int num_cols;
  // algo is the inference algorithm;
  // sparse forests do not distinguish between NAIVE and TREE_REORG
  algo_t algo;
  // output is the desired output type
  output_t output;
  // threshold is used to for classification if output == OUTPUT_CLASS,
  // and is ignored otherwise
  float threshold;
  // global_bias is added to the sum of tree predictions
  // (after averaging, if it is used, but before any further transformations)
  float global_bias;
  strategy_t strategy;

  float missing;
};


/** performs additional transformations on the array of forest predictions
    (preds) of size n; the transformations are defined by output, and include
    averaging (multiplying by inv_num_trees), adding global_bias (always done),
    sigmoid and applying threshold */
__global__ void transform_k(float* preds, size_t n, output_t output,
                            float inv_num_trees, float threshold,
                            float global_bias) {
  size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (i >= n) return;
  float result = preds[i];
  if ((output & output_t::AVG) != 0) result *= inv_num_trees;
  result += global_bias;
  if ((output & output_t::SIGMOID) != 0) result = sigmoid(result);
  if ((output & output_t::THRESHOLD) != 0) {
    result = result > threshold ? 1.0f : 0.0f;
  }
  preds[i] = result;
}


struct forest {


   void init_max_shm() {
    int device = 0;
	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev, device);
	//printf( "Shared mem per mp: %d\n", (int)dev.sharedMemPerBlock );
	max_shm_ = (int)dev.sharedMemPerBlock * 0.8;
	/*
    // TODO(canonizer): use cumlHandle for this
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shm_, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    // TODO(canonizer): use >48KiB shared memory if available
	*/
  } 

  void init_common(const forest_params_t* params) {
    depth_ = params->depth;
    num_trees_ = params->num_trees;
    num_cols_ = params->num_cols;
    algo_ = params->algo;
    strategy_ = params->strategy;
    output_ = params->output;
    threshold_ = params->threshold;
    global_bias_ = params->global_bias;
    missing_ = params->missing;
    init_max_shm();
  }

  virtual void infer(predict_params params, cudaStream_t stream) = 0;

  void predict(cudaStream_t stream, float* preds, const float* data,
               size_t num_rows) {
    // Initialize prediction parameters.
    predict_params params;
    params.num_cols = num_cols_;
    params.algo = algo_;
    params.strategy = strategy_;
    params.preds = preds;
    params.data = data;
    params.num_rows = num_rows;
    params.max_shm = max_shm_;
    params.missing = missing_;

    // Predict using the forest.
    infer(params, stream);
	//printf("infer\n");

    // Transform the output if necessary.
    if (output_ != output_t::RAW || global_bias_ != 0.0f) {
      transform_k<<<ceildiv(int(num_rows), FIL_TPB), FIL_TPB, 0, stream>>>(
        preds, num_rows, output_, num_trees_ > 0 ? (1.0f / num_trees_) : 1.0f,
        threshold_, global_bias_);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  } 

  int num_trees_ = 0;
  int depth_ = 0;
  int num_cols_ = 0;
  algo_t algo_ = algo_t::NAIVE;
  strategy_t strategy_ = strategy_t::SHARED_DATA;
  int max_shm_ = 0;
  output_t output_ = output_t::RAW;
  float threshold_ = 0.5;
  float global_bias_ = 0;
  float missing_ = 0.0;
};


/** forest_t is the predictor handle */
typedef forest* forest_t;

// vec wraps float[N] for cub::BlockReduce
template <int N>
struct vec {
  float data[N];
  __host__ __device__ float& operator[](int i) { return data[i]; }
  __host__ __device__ float operator[](int i) const { return data[i]; }
  friend __host__ __device__ vec<N> operator+(const vec<N>& a,
                                              const vec<N>& b) {
    vec<N> r;
#pragma unroll
    for (int i = 0; i < N; ++i) r[i] = a[i] + b[i];
    return r;
  }
};


/** dense_tree represents a dense tree */
struct dense_tree {
  __host__ __device__ dense_tree(dense_node_t* nodes, int node_pitch)
    : nodes_(nodes), node_pitch_(node_pitch) {}
  __host__ __device__ const dense_node_t& operator[](int i) const {
    return nodes_[i * node_pitch_];
  }
  dense_node_t* nodes_ = nullptr;
  int node_pitch_ = 0;
};


/** dense_storage stores the forest as a collection of dense nodes */
struct dense_storage {
  __host__ __device__ dense_storage(dense_node_t* nodes, int num_trees,
                                    int tree_stride, int node_pitch)
    : nodes_(nodes),
      num_trees_(num_trees),
      tree_stride_(tree_stride),
      node_pitch_(node_pitch) {}
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ dense_tree operator[](int i) const {
    return dense_tree(nodes_ + i * tree_stride_, node_pitch_);
  }
  dense_node_t* nodes_ = nullptr;
  int num_trees_ = 0;
  int tree_stride_ = 0;
  int node_pitch_ = 0;
};

/** sparse_tree is a sparse tree */
struct sparse_tree {
  __host__ __device__ sparse_tree(sparse_node_t* nodes) : nodes_(nodes) {}
  __host__ __device__ const sparse_node_t& operator[](int i) const {
    return nodes_[i];
  }
  sparse_node_t* nodes_ = nullptr;
};

/** sparse_storage stores the forest as a collection of sparse nodes */
struct sparse_storage {
  int* trees_ = nullptr;
  sparse_node_t* nodes_ = nullptr;
  int num_trees_ = 0;
  __host__ __device__ sparse_storage(int* trees, sparse_node_t* nodes,
                                     int num_trees)
    : trees_(trees), nodes_(nodes), num_trees_(num_trees) {}
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ sparse_tree operator[](int i) const {
    return sparse_tree(&nodes_[trees_[i]]);
  }
};




template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data(dense_tree tree, float* sdata,
                                               int cols, vec<NITEMS>& out, algo_t algo_, int num_trees, float missing) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = tree.nodes_[curr[j]].val;
      int n_bits = tree.nodes_[curr[j]].bits;
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 30) - 1);
	  bool n_def_left = n_bits & (1 << 30);
	  bool n_is_leaf = n_bits & (1 << 31);
	  //printf("%d\n", &tree.nodes_[curr[j]]);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(algo_ == algo_t::NAIVE)
	  {
		  curr[j] = (curr[j]<<1)+1+cond;
	  }
	  if(algo_ == algo_t::TREE_REORG)
	  {
		  unsigned int temp = ((curr[j]/num_trees)*2+1)*num_trees + (cond ? num_trees : 0);////////////////////////////////////////////////////
		  curr[j] = temp ;/////////////////////////////////////////////////////////////////
		  //curr[j] = (((curr[j]/num_trees)*2)+1)*num_trees + curr[j]%2 + cond;
	  }
	  if(algo_ == algo_t::BATCH_TREE_REORG)
	  {
		  unsigned int temp = ((curr[j]/num_trees)*2+1)*num_trees + (cond ? num_trees : 0);////////////////////////////////////////////////////
		  curr[j] = temp ;/////////////////////////////////////////////////////////////////
		  //curr[j] = (((curr[j]/num_trees)*2)+1)*num_trees + curr[j]%2 + cond;
	  }

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += tree.nodes_[curr[j]].val;
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data(dense_storage forest, predict_params params) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;
  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] =
        row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  // one block works on a single row and the whole forest
  for (int j = threadIdx.x; j < forest.num_trees(); j += blockDim.x) {
    //for(int loop=0;loop<50;loop++)
    infer_one_tree_dense_shared_data<NITEMS>(forest[j], sdata, params.num_cols, out, params.algo, forest.num_trees(), params.missing);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }

}


void infer_dense_shared_data(dense_storage forest, predict_params params, cudaStream_t stream) {
  //printf("shared memory is %d\n", params.max_shm);
  int num_items = params.max_shm / (sizeof(float) * params.num_cols);
  //printf("shared memory is %d\n", params.max_shm);
  if (num_items == 0) {
    //int max_cols = params.max_shm / sizeof(float);
    assert(false && "too many features");
  }
  num_items = params.algo == algo_t::BATCH_TREE_REORG ? 1 : 1;
  //printf("MAX_BATCH_ITEMS: %d\n", num_items);
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;
  //printf("start infer_k: %d, num_blocks: %d, FIL_TPB: %d, shm_sz: %d, stream: %d\n", num_items, num_blocks, FIL_TPB, shm_sz, stream);

  switch (num_items) {
    case 1:
      //infer_k_shared_data<1><<<1, 1, shm_sz, stream>>>(forest, params);
      infer_k_shared_data<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 2:
      infer_k_shared_data<2><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 3:
      infer_k_shared_data<3><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 4:
      infer_k_shared_data<4><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 16:
      infer_k_shared_data<16><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 24:
      infer_k_shared_data<24><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    default:
      assert(false && "internal error: nitems > 4");
  }
  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}





__device__ __forceinline__ void infer_one_tree_dense_shared_forest(dense_node_t* tree, const float* sdata, int cols, float& out, int num_trees, int num_nodes) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  dense_node_t* curr_tree = tree;
  do {
      float n_val = curr_tree[curr].val;
      int n_bits = curr_tree[curr].bits;
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 30) - 1);
	  bool n_def_left = n_bits & (1 << 30);
	  bool n_is_leaf = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
      bool cond = isnan(val) ? !n_def_left : val >= n_thresh;
	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_shared_forest(dense_storage forest, predict_params params) {
  extern __shared__ char smem[];
  dense_node_t* stree = (dense_node_t*)smem;
  for (int i = 0; i < forest.num_trees(); i++) {
	for(int j= threadIdx.x; j<forest.tree_stride_; j+=blockDim.x)
	{
		stree[i*forest.tree_stride_+j].val = forest[i].nodes_[j].val;
		stree[i*forest.tree_stride_+j].bits = forest[i].nodes_[j].bits;
	}
  }
  __syncthreads();

  float out = 0.0;

  int idx = blockIdx.x * FIL_TPB + threadIdx.x;
  if (idx < params.num_rows)
  {
    infer_one_tree_dense_shared_forest(stree, &params.data[idx*params.num_cols], params.num_cols, out, forest.num_trees(), forest.tree_stride_);
	params.preds[idx] = out;
  }

}

void infer_dense_shared_forest(dense_storage forest, predict_params params, cudaStream_t stream) {
  int shm_sz = forest.num_trees() * sizeof(struct dense_node_t) * forest.tree_stride_;
  //printf("shared memory is %d\n", params.max_shm);
  if (shm_sz > params.max_shm) {
    assert(false && "forest is too large to save in shared memory");
  }

  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);
  //printf("%d - %d\n", num_blocks, FIL_TPB);
  infer_k_shared_forest<<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}









__device__ __forceinline__ void infer_one_tree_dense_split_forest(dense_node_t* tree, const float* sdata, int cols, float& out, int num_trees, int num_nodes) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  dense_node_t* curr_tree = tree;
  do {
      float n_val = curr_tree[curr].val;
      int n_bits = curr_tree[curr].bits;
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 30) - 1);
	  bool n_def_left = n_bits & (1 << 30);
	  bool n_is_leaf = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
      bool cond = isnan(val) ? !n_def_left : val >= n_thresh;
	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}



__global__ void infer_k_split_forest(dense_storage forest, predict_params params, int trees_per_sm, float* temp_out) {
  extern __shared__ char smem[];
  dense_node_t* stree = (dense_node_t*)smem;
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
  //for (int i = trees_per_sm*(blockIdx.x); i < trees_per_sm*(blockIdx.x+1) && i < forest.num_trees(); i++) {
	if((trees_per_sm * (blockIdx.x) + i) < forest.num_trees())
	{
		for(int j= threadIdx.x; j<forest.tree_stride_; j+=blockDim.x)
		{
		stree[i*forest.tree_stride_+j].val = forest[trees_per_sm * (blockIdx.x) + i].nodes_[j].val;
		stree[i*forest.tree_stride_+j].bits = forest[trees_per_sm * (blockIdx.x) + i].nodes_[j].bits;
		}
		trees++;
	}
  }
  __syncthreads();

  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest(stree, &params.data[idx*params.num_cols], params.num_cols, out, trees, forest.tree_stride_);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }

}

void infer_dense_split_forest(dense_storage forest, predict_params params, cudaStream_t stream) {
  int trees_per_sm = params.max_shm / (sizeof(struct dense_node_t) * forest.tree_stride_);
  //printf("shared memory is %d\n", params.max_shm);
  if (trees_per_sm == 0) {
    assert(false && "single tree is too big to save in shared memory");
  }
  int num_blocks = ceildiv(forest.num_trees(), trees_per_sm);
  int shm_sz = trees_per_sm * (sizeof(struct dense_node_t)) * forest.tree_stride_;////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!
  //printf("shared memory is %d\n", shm_sz);
  //printf("num trees is %d, tree_stride_ is %d\n", forest.num_trees(), forest.tree_stride_);
  //printf("start infer_k: %d, num_blocks: %d, FIL_TPB: %d, shm_sz: %d, stream: %d\n", num_items, num_blocks, FIL_TPB, shm_sz, stream);
  //printf("trees_per_sm: %d, num_blocks: %d\n", trees_per_sm, num_blocks);

  float* temp_out = NULL;
  allocate(temp_out, num_blocks*params.num_rows);

  infer_k_split_forest<<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params, trees_per_sm, temp_out);

  int* h_offsets=(int*)malloc(sizeof(int)*(params.num_rows+1));
  for(int i=0;i<=params.num_rows;i++)
	  h_offsets[i]=i*num_blocks;

  int* d_offsets;
  allocate(d_offsets, (params.num_rows+1));
  updateDevice(d_offsets, h_offsets, (params.num_rows+1), stream);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sum-reduction
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);

  CUDA_CHECK(cudaFree(d_temp_storage));

  free(h_offsets);
  CUDA_CHECK(cudaFree(d_offsets));

  CUDA_CHECK(cudaFree(temp_out));

  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}












__global__ void infer_k_split_forest_shared_data(dense_storage forest, predict_params params, int trees_per_sm, int data_per_sm, float* temp_out) {
  extern __shared__ char smem[];
  dense_node_t* stree = (dense_node_t*)smem;
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
	if((trees_per_sm * (blockIdx.x) + i) < forest.num_trees())
	{
		for(int j= threadIdx.x; j<forest.tree_stride_; j+=blockDim.x)
		{
		stree[i*forest.tree_stride_+j].val = forest[trees_per_sm * (blockIdx.x) + i].nodes_[j].val;
		stree[i*forest.tree_stride_+j].bits = forest[trees_per_sm * (blockIdx.x) + i].nodes_[j].bits;
		}
		trees++;
	}
  }

  __syncthreads();


  int loops = ceildiv_dev(params.num_rows, data_per_sm);
  float* sdata = (float*)&(smem[trees_per_sm * (sizeof(struct dense_node_t)) * forest.tree_stride_]);
  for(int loop=0; loop<loops; loop++)
  {

   for (int j = 0; j < data_per_sm && (loop*data_per_sm+j < params.num_rows); ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      sdata[j * params.num_cols + i] = params.data[(loop*data_per_sm+j) * params.num_cols + i];
    }
   __syncthreads();

  for(int idx= threadIdx.x; idx<data_per_sm; idx+=blockDim.x)
  {
    float out = 0.0;
    infer_one_tree_dense_split_forest(stree, &sdata[idx*params.num_cols], params.num_cols, out, trees, forest.tree_stride_);
    temp_out[(idx+loop*data_per_sm)*gridDim.x + blockIdx.x] = out;
  }

  }

  }


  /*
  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest(stree, &params.data[idx*params.num_cols], params.num_cols, out, trees, forest.tree_stride_);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }
  */

}



void infer_dense_split_forest_shared_data(dense_storage forest, predict_params params, cudaStream_t stream) {
  int trees_per_sm = (params.max_shm / 2.0) / (sizeof(struct dense_node_t) * forest.tree_stride_);
  //printf("shared memory is %d\n", params.max_shm);
  if (trees_per_sm == 0) {
    assert(false && "single tree is too big to save in shared memory");
  }
  int num_blocks = ceildiv(forest.num_trees(), trees_per_sm);
  //printf("shared memory is %d\n", shm_sz);
  //printf("num trees is %d, tree_stride_ is %d\n", forest.num_trees(), forest.tree_stride_);
  //printf("start infer_k: %d, num_blocks: %d, FIL_TPB: %d, shm_sz: %d, stream: %d\n", num_items, num_blocks, FIL_TPB, shm_sz, stream);
  //printf("trees_per_sm: %d, num_blocks: %d\n", trees_per_sm, num_blocks);


  //printf("shared memory is %d\n", params.max_shm);
  int data_per_sm = (params.max_shm / 2.0) / (sizeof(float) * params.num_cols);
  //printf("shared memory is %d\n", params.max_shm);
  if (data_per_sm == 0) {
    //int max_cols = params.max_shm / sizeof(float);
    assert(false && "too many features");
  }

  int shm_sz = trees_per_sm * (sizeof(struct dense_node_t)) * forest.tree_stride_ + data_per_sm * sizeof(float) * params.num_cols;////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!

  //printf("trees_per_sm: %d, data_per_sm: %d\n", trees_per_sm, data_per_sm);

  float* temp_out = NULL;
  allocate(temp_out, num_blocks*params.num_rows);

  infer_k_split_forest_shared_data<<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params, trees_per_sm, data_per_sm, temp_out);

  int* h_offsets=(int*)malloc(sizeof(int)*(params.num_rows+1));
  for(int i=0;i<=params.num_rows;i++)
	  h_offsets[i]=i*num_blocks;

  int* d_offsets;
  allocate(d_offsets, (params.num_rows+1));
  updateDevice(d_offsets, h_offsets, (params.num_rows+1), stream);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sum-reduction
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);

  CUDA_CHECK(cudaFree(d_temp_storage));

  free(h_offsets);
  CUDA_CHECK(cudaFree(d_offsets));

  CUDA_CHECK(cudaFree(temp_out));

  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}





struct dense_forest : forest {

  void transform_trees(const dense_node_t* nodes) {
    // populate node information
    for (int i = 0, gid = 0; i < num_trees_; ++i) {
      for (int j = 0, nid = 0; j <= depth_; ++j) {
        for (int k = 0; k < 1 << j; ++k, ++nid, ++gid) {
          h_nodes_[nid * num_trees_ + i] = nodes[gid];
        }
      }
    }
  } 

  void init(cudaStream_t stream, const dense_node_t* nodes,
            const forest_params_t* params) {
    init_common(params);
    int num_nodes = forest_num_nodes(num_trees_, depth_);
    allocate(nodes_, num_nodes);
    h_nodes_.resize(num_nodes);
    if (algo_ == algo_t::NAIVE) {
      std::copy(nodes, nodes + num_nodes, h_nodes_.begin());
    } else {
      transform_trees(nodes);
    }

    CUDA_CHECK(cudaMemcpyAsync(nodes_, h_nodes_.data(), num_nodes * sizeof(dense_node_t), cudaMemcpyHostToDevice, stream));
    // copy must be finished before freeing the host data
    CUDA_CHECK(cudaStreamSynchronize(stream));
    h_nodes_.clear();
    h_nodes_.shrink_to_fit();

  }

  virtual void infer(predict_params params, cudaStream_t stream) override {
    dense_storage forest(nodes_, num_trees_,
                         algo_ == algo_t::NAIVE ? tree_num_nodes(depth_) : 1,
                         algo_ == algo_t::NAIVE ? 1 : num_trees_);
	if(params.strategy == strategy_t::SHARED_DATA)
	{
		infer_dense_shared_data(forest, params, stream);
	}
	else if(params.strategy == strategy_t::SHARED_FOREST)
	{
		infer_dense_shared_forest(forest, params, stream);
	}
	else if(params.strategy == strategy_t::SPLIT_FOREST)
	{
		infer_dense_split_forest(forest, params, stream);
	}
	else if(params.strategy == strategy_t::SPLIT_FOREST_SHARED_DATA)
	{
		infer_dense_split_forest_shared_data(forest, params, stream);
	}

  } 

  dense_node_t* nodes_ = nullptr;
  thrust::host_vector<dense_node_t> h_nodes_;

};






/////shared_data_adaptive without rearrangement
template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data_adaptive(int tree, float* sdata, int cols, vec<NITEMS>& out, 
								algo_t algo_, int num_trees, float missing, int num_nodes_per_tree, float* bits_values_d, char* bits_char_d) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = bits_values_d[tree*num_nodes_per_tree + curr[j]];
      char n_bits = bits_char_d[tree*num_nodes_per_tree + curr[j]];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 5) - 1);
	  bool n_def_left = n_bits & (1 << 5);
	  bool n_is_leaf = n_bits & (1 << 6);
	  bool n_is_exchange = n_bits & (1 << 7);
	  //printf("%d\n", curr[j]);
	  //printf("n_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d, n_is_exchange: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf, n_is_exchange);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(n_is_exchange) cond = !cond;

	  curr[j] = (curr[j]<<1)+1+cond;

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += bits_values_d[tree*num_nodes_per_tree + curr[j]];
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data_adaptive(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_d, char* bits_char_d) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;

  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = threadIdx.x; j < num_trees_; j += blockDim.x) {
    infer_one_tree_dense_shared_data_adaptive<NITEMS>(j, sdata, params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_d, bits_char_d);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }
}







/////shared_data_adaptive with rearrangement for int
template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data_adaptive_reorg_int(int tree, float* sdata, int cols, vec<NITEMS>& out, 
								algo_t algo_, int num_trees, float missing, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = bits_values_reorg_d[tree + curr[j]*num_trees];
      int n_bits = bits_int_reorg_d[tree + curr[j]*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 29) - 1);
	  bool n_def_left = n_bits & (1 << 29);
	  bool n_is_leaf = n_bits & (1 << 30);
	  bool n_is_exchange = n_bits & (1 << 31);
	  //printf("%d\n", curr[j]);
	  //printf("n_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d, n_is_exchange: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf, n_is_exchange);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(n_is_exchange) cond = !cond;

	  curr[j] = (curr[j]<<1)+1+cond;

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += bits_values_reorg_d[tree + curr[j]*num_trees];
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data_adaptive_reorg_int(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;

  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = threadIdx.x; j < num_trees_; j += blockDim.x) {
    infer_one_tree_dense_shared_data_adaptive_reorg_int<NITEMS>(j, sdata, params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_int_reorg_d);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }
}





/////shared_data_adaptive with rearrangement for short
template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data_adaptive_reorg_short(int tree, float* sdata, int cols, vec<NITEMS>& out, 
								algo_t algo_, int num_trees, float missing, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = bits_values_reorg_d[tree + curr[j]*num_trees];
      short n_bits = bits_short_reorg_d[tree + curr[j]*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 13) - 1);
	  bool n_def_left = n_bits & (1 << 13);
	  bool n_is_leaf = n_bits & (1 << 14);
	  bool n_is_exchange = n_bits & (1 << 15);
	  //printf("%d\n", curr[j]);
	  //printf("n_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d, n_is_exchange: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf, n_is_exchange);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(n_is_exchange) cond = !cond;

	  curr[j] = (curr[j]<<1)+1+cond;

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += bits_values_reorg_d[tree + curr[j]*num_trees];
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data_adaptive_reorg_short(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;

  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = threadIdx.x; j < num_trees_; j += blockDim.x) {
    infer_one_tree_dense_shared_data_adaptive_reorg_short<NITEMS>(j, sdata, params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_short_reorg_d);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }
}




/////shared_data_adaptive with rearrangement for char
template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_dense_shared_data_adaptive_reorg_char(int tree, float* sdata, int cols, vec<NITEMS>& out, 
								algo_t algo_, int num_trees, float missing, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = bits_values_reorg_d[tree + curr[j]*num_trees];
      char n_bits = bits_char_reorg_d[tree + curr[j]*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 5) - 1);
	  bool n_def_left = n_bits & (1 << 5);
	  bool n_is_leaf = n_bits & (1 << 6);
	  bool n_is_exchange = n_bits & (1 << 7);
	  //printf("%d\n", curr[j]);
	  //printf("n_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d, n_is_exchange: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf, n_is_exchange);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;

	  if(n_is_exchange) cond = !cond;

	  curr[j] = (curr[j]<<1)+1+cond;

    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += bits_values_reorg_d[tree + curr[j]*num_trees];
  //printf("\n");
}


template <int NITEMS>
__global__ void infer_k_shared_data_adaptive_reorg_char(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;

  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = threadIdx.x; j < num_trees_; j += blockDim.x) {
    infer_one_tree_dense_shared_data_adaptive_reorg_char<NITEMS>(j, sdata, params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_char_reorg_d);
  }
  __syncthreads();

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }
}




template <int NITEMS>
__global__ void infer_adaptive_reorg_char(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  int index = blockIdx.x * FIL_TPB + threadIdx.x;
  if(index < params.num_rows)
  {
  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = 0; j < num_trees_; j++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_char<NITEMS>(j, const_cast<float*>(&params.data[index * params.num_cols]), params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_char_reorg_d);
  }
  params.preds[index] = out[0];
  }

}
template <int NITEMS>
__global__ void infer_adaptive_reorg_short(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  int index = blockIdx.x * FIL_TPB + threadIdx.x;
  if(index < params.num_rows)
  {
  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = 0; j < num_trees_; j++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_short<NITEMS>(j, const_cast<float*>(&params.data[index * params.num_cols]), params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_short_reorg_d);
  }
  params.preds[index] = out[0];
  }

}
template <int NITEMS>
__global__ void infer_adaptive_reorg_int(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  int index = blockIdx.x * FIL_TPB + threadIdx.x;
  if(index < params.num_rows)
  {
  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int j = 0; j < num_trees_; j++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_int<NITEMS>(j, const_cast<float*>(&params.data[index * params.num_cols]), params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_int_reorg_d);
  }
  params.preds[index] = out[0];
  }

}




__device__ __forceinline__ void infer_one_tree_dense_shared_forest_adaptive_reorg_char(char* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  do {
      float n_val = stree_vals[num_tree_curr + curr*num_trees];
      char n_bits = stree_bits[num_tree_curr + curr*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 5) - 1);
	  bool n_def_left = n_bits & (1 << 5);
	  bool n_is_leaf = n_bits & (1 << 6);
	  bool n_is_exchange = n_bits & (1 << 7);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}

__global__ void infer_k_shared_forest_adaptive_reorg_char(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  char* stree_bits = smem + sizeof(float)*num_trees_*num_nodes_per_tree;
  for (int i = 0; i < num_trees_; i++) {
	for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
	{
		stree_bits[i*num_nodes_per_tree+j] = bits_char_reorg_d[i*num_nodes_per_tree+j];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[i*num_nodes_per_tree+j];
	}
  }
  __syncthreads();

  float out = 0.0;

  int idx = blockIdx.x * FIL_TPB + threadIdx.x;
  if (idx < params.num_rows)
  {
    infer_one_tree_dense_shared_forest_adaptive_reorg_char(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, num_trees_, num_nodes_per_tree, params.missing);
	params.preds[idx] = out;
  }

}


__device__ __forceinline__ void infer_one_tree_dense_shared_forest_adaptive_reorg_int(int* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  do {
      float n_val = stree_vals[num_tree_curr + curr*num_trees];
      int n_bits = stree_bits[num_tree_curr + curr*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 29) - 1);
	  bool n_def_left = n_bits & (1 << 29);
	  bool n_is_leaf = n_bits & (1 << 30);
	  bool n_is_exchange = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_shared_forest_adaptive_reorg_int(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  int index = sizeof(float)*num_trees_*num_nodes_per_tree;
  while(index%4 != 0) index++;
  int* stree_bits = (int*) (smem + index);
  for (int i = 0; i < num_trees_; i++) {
	for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
	{
		stree_bits[i*num_nodes_per_tree+j] = bits_int_reorg_d[i*num_nodes_per_tree+j];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[i*num_nodes_per_tree+j];
	}
  }
  __syncthreads();

  float out = 0.0;

  int idx = blockIdx.x * FIL_TPB + threadIdx.x;
  if (idx < params.num_rows)
  {
    infer_one_tree_dense_shared_forest_adaptive_reorg_int(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, num_trees_, num_nodes_per_tree, params.missing);
	params.preds[idx] = out;
  }

}




__device__ __forceinline__ void infer_one_tree_dense_shared_forest_adaptive_reorg_short(short* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  do {
      float n_val = stree_vals[num_tree_curr + curr*num_trees];
      short n_bits = stree_bits[num_tree_curr + curr*num_trees];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 13) - 1);
	  bool n_def_left = n_bits & (1 << 13);
	  bool n_is_leaf = n_bits & (1 << 14);
	  bool n_is_exchange = n_bits & (1 << 15);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_shared_forest_adaptive_reorg_short(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  int index = sizeof(float)*num_trees_*num_nodes_per_tree;
  while(index%2 != 0) index++;
  short* stree_bits = (short*) (smem + index);
  for (int i = 0; i < num_trees_; i++) {
	for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
	{
		stree_bits[i*num_nodes_per_tree+j] = bits_short_reorg_d[i*num_nodes_per_tree+j];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[i*num_nodes_per_tree+j];
	}
  }
  __syncthreads();

  float out = 0.0;

  int idx = blockIdx.x * FIL_TPB + threadIdx.x;
  if (idx < params.num_rows)
  {
    infer_one_tree_dense_shared_forest_adaptive_reorg_short(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, num_trees_, num_nodes_per_tree, params.missing);
	params.preds[idx] = out;
  }

}




__device__ __forceinline__ void infer_one_tree_dense_split_forest_adaptive_reorg_int(int* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  int* curr_tree_bits = stree_bits;
  float* curr_tree_vals = stree_vals;
  do {
      float n_val = curr_tree_vals[curr];
      int n_bits = curr_tree_bits[curr];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 29) - 1);
	  bool n_def_left = n_bits & (1 << 29);
	  bool n_is_leaf = n_bits & (1 << 30);
	  bool n_is_exchange = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree_bits+=num_nodes;
		  curr_tree_vals+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_split_forest_adaptive_reorg_int(predict_params params, int trees_per_sm, float* temp_out, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  int* stree_bits = (int*)(smem + sizeof(float)*trees_per_sm*num_nodes_per_tree);
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
	if((trees_per_sm * (blockIdx.x) + i) < num_trees_)
	{
		for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
		{
		int tree_index = i+trees_per_sm*(blockIdx.x);
		stree_bits[i*num_nodes_per_tree+j] = bits_int_reorg_d[tree_index + j*num_trees_];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[tree_index + j*num_trees_];
		}
		trees++;
	}
  }
  __syncthreads();

  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest_adaptive_reorg_int(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, trees, num_nodes_per_tree, params.missing);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }

}




__device__ __forceinline__ void infer_one_tree_dense_split_forest_adaptive_reorg_short(short* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  short* curr_tree_bits = stree_bits;
  float* curr_tree_vals = stree_vals;
  do {
      float n_val = curr_tree_vals[curr];
      short n_bits = curr_tree_bits[curr];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 13) - 1);
	  bool n_def_left = n_bits & (1 << 13);
	  bool n_is_leaf = n_bits & (1 << 14);
	  bool n_is_exchange = n_bits & (1 << 15);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree_bits+=num_nodes;
		  curr_tree_vals+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_split_forest_adaptive_reorg_short(predict_params params, int trees_per_sm, float* temp_out, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  short* stree_bits = (short*)(smem + sizeof(float)*trees_per_sm*num_nodes_per_tree);
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
	if((trees_per_sm * (blockIdx.x) + i) < num_trees_)
	{
		for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
		{
		int tree_index = i+trees_per_sm*(blockIdx.x);
		stree_bits[i*num_nodes_per_tree+j] = bits_short_reorg_d[tree_index + j*num_trees_];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[tree_index + j*num_trees_];
		}
		trees++;
	}
  }
  __syncthreads();

  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest_adaptive_reorg_short(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, trees, num_nodes_per_tree, params.missing);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }

}



__device__ __forceinline__ void infer_one_tree_dense_split_forest_adaptive_reorg_char(char* stree_bits, float* stree_vals, const float* sdata, int cols, float& out, int num_trees, int num_nodes, float missing) {
  unsigned int curr;
  curr = 0;
  int num_tree_curr = 0;
  char* curr_tree_bits = stree_bits;
  float* curr_tree_vals = stree_vals;
  do {
      float n_val = curr_tree_vals[curr];
      char n_bits = curr_tree_bits[curr];
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 5) - 1);
	  bool n_def_left = n_bits & (1 << 5);
	  bool n_is_leaf = n_bits & (1 << 6);
	  bool n_is_exchange = n_bits & (1 << 7);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
		  out+=n_thresh;
		  num_tree_curr++;
		  curr_tree_bits+=num_nodes;
		  curr_tree_vals+=num_nodes;
		  curr = 0;
          continue;
      }
      float val = sdata[n_fid];
	  const float eps=1.0e-6;
	  bool cond = (fabs(val-missing)<=eps) ? !n_def_left : val >= n_thresh;
	  if(n_is_exchange) cond = !cond;

	  curr = (curr<<1)+1+cond;
  } while (num_tree_curr<num_trees);
}


__global__ void infer_k_split_forest_adaptive_reorg_char(predict_params params, int trees_per_sm, float* temp_out, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d) {
  extern __shared__ char smem[];
  float* stree_vals = (float*)smem;
  char* stree_bits = smem + sizeof(float)*trees_per_sm*num_nodes_per_tree;
  int trees = 0;
  for (int i = 0; i < trees_per_sm; i++) {
	if((trees_per_sm * (blockIdx.x) + i) < num_trees_)
	{
		for(int j= threadIdx.x; j<num_nodes_per_tree; j+=blockDim.x)
		{
		int tree_index = i+trees_per_sm*(blockIdx.x);
		stree_bits[i*num_nodes_per_tree+j] = bits_char_reorg_d[tree_index + j*num_trees_];
		stree_vals[i*num_nodes_per_tree+j] = bits_values_reorg_d[tree_index + j*num_trees_];
		}
		trees++;
	}
  }
  __syncthreads();

  float out=0.0;

  for(int idx= threadIdx.x; idx<params.num_rows; idx+=blockDim.x)
  {
	out = 0.0;
    infer_one_tree_dense_split_forest_adaptive_reorg_char(stree_bits, stree_vals, &params.data[idx*params.num_cols], params.num_cols, out, trees, num_nodes_per_tree, params.missing);
	temp_out[idx*gridDim.x + blockIdx.x] = out;
  }

}



template <int NITEMS>
__global__ void infer_k_shared_data_wo_adaptive_reorg_int(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, int* bits_int_reorg_d, int num) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * num;

  for (int j = threadIdx.x; (j < num) && ((j+rid) < params.num_rows); j += blockDim.x) {
    for (int i = 0; i < params.num_cols; i ++) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = params.data[row * params.num_cols + i] ;
    }

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int k = 0; k < num_trees_; k ++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_int<NITEMS>(k, &sdata[j * params.num_cols], params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_int_reorg_d);
  }

      int idx = blockIdx.x * num + j;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[0];
	  }

}

}




template <int NITEMS>
__global__ void infer_k_shared_data_wo_adaptive_reorg_short(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, short* bits_short_reorg_d, int num) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * num;

  for (int j = threadIdx.x; (j < num) && ((j+rid) < params.num_rows); j += blockDim.x) {
    for (int i = 0; i < params.num_cols; i ++) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = params.data[row * params.num_cols + i] ;
    }

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int k = 0; k < num_trees_; k ++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_short<NITEMS>(k, &sdata[j * params.num_cols], params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_short_reorg_d);
  }

      int idx = blockIdx.x * num + j;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[0];
	  }

}

}




template <int NITEMS>
__global__ void infer_k_shared_data_wo_adaptive_reorg_char(predict_params params, int num_trees_, int num_nodes_per_tree, float* bits_values_reorg_d, char* bits_char_reorg_d, int num) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * num;

  for (int j = threadIdx.x; (j < num) && ((j+rid) < params.num_rows); j += blockDim.x) {
    for (int i = 0; i < params.num_cols; i ++) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] = params.data[row * params.num_cols + i] ;
    }

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;

  for (int k = 0; k < num_trees_; k ++) {
    infer_one_tree_dense_shared_data_adaptive_reorg_char<NITEMS>(k, &sdata[j * params.num_cols], params.num_cols, out, params.algo, num_trees_, params.missing, num_nodes_per_tree, bits_values_reorg_d, bits_char_reorg_d);
  }

      int idx = blockIdx.x * num + j;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[0];
	  }

}

}





struct dense_adaptive_forest : forest {

void swap_child(int k, int depth, std::vector<float> &values_h, std::vector<float> &weights_h, std::vector<int> &fids_h, bool* defaults_h, bool* is_leafs_h, bool* exchanges_h, int tree_offset)
{

	int offset = pow(2,depth);
	float temp_value, temp_weight;
	int temp_fid;
	bool temp_default, temp_is_leaf, temp_exchange;

	temp_value = values_h[k + offset+tree_offset];
	values_h[k + offset+tree_offset] = values_h[k+tree_offset];
	values_h[k+tree_offset] = temp_value;

	temp_weight = weights_h[k + offset+tree_offset];
	weights_h[k + offset+tree_offset] = weights_h[k+tree_offset];
	weights_h[k+tree_offset] = temp_weight;

	temp_fid = fids_h[k + offset+tree_offset];
	fids_h[k + offset+tree_offset] = fids_h[k+tree_offset];
	fids_h[k+tree_offset] = temp_fid;

	temp_default = defaults_h[k + offset+tree_offset];
	defaults_h[k + offset+tree_offset] = defaults_h[k+tree_offset];
	defaults_h[k+tree_offset] = temp_default;

	temp_is_leaf = is_leafs_h[k + offset+tree_offset];
	is_leafs_h[k + offset+tree_offset] = is_leafs_h[k+tree_offset];
	is_leafs_h[k+tree_offset] = temp_is_leaf;
	
	temp_exchange = exchanges_h[k + offset+tree_offset];
	exchanges_h[k + offset+tree_offset] = exchanges_h[k+tree_offset];
	exchanges_h[k+tree_offset] = temp_exchange;

	if((k*2+1)<tree_num_nodes(depth_))
	{
	swap_child((k*2+1), depth+1, values_h, weights_h, fids_h, defaults_h, is_leafs_h, exchanges_h, tree_offset);
	swap_child((k*2+2), depth+1, values_h, weights_h, fids_h, defaults_h, is_leafs_h, exchanges_h, tree_offset);
	}

}





  void init(cudaStream_t stream, const dense_node_t* nodes,
            const forest_params_t* params) {
    init_common(params);
    int num_nodes = forest_num_nodes(num_trees_, depth_);
    int num_nodes_per_tree = tree_num_nodes(depth_);
    std::vector<float> values_h(num_nodes), weights_h(num_nodes);
    std::vector<int> fids_h(num_nodes);
    bool* defaults_h = new bool[num_nodes];
    bool* is_leafs_h = new bool[num_nodes];
    bool* exchanges_h = new bool[num_nodes];

    for (size_t i = 0; i < num_trees_; ++i) {
        for (size_t j = 0; j < num_nodes_per_tree; ++j) {
		dense_node_decode(&nodes[i*num_nodes_per_tree + j], &values_h[i*num_nodes_per_tree + j], &weights_h[i*num_nodes_per_tree + j],
				&fids_h[i*num_nodes_per_tree + j], &defaults_h[i*num_nodes_per_tree + j], &is_leafs_h[i*num_nodes_per_tree + j]);
		exchanges_h[i*num_nodes_per_tree + j] = 0;
	}
    }

    for (size_t i = 0; i < num_trees_; ++i) {
		int tree_offset = i*num_nodes_per_tree;
        for (int j = (depth_-1); j >= 0; --j) {
	    for(int k = (pow(2,j)-1); k < (pow(2,j+1)-1); k++) {
			if(is_leafs_h[k+tree_offset]==0)
			{
               float left_weight = weights_h[k*2+1+tree_offset];
               float right_weight = weights_h[k*2+2+tree_offset];
			   if(left_weight<right_weight)
			   {
				    exchanges_h[k+tree_offset] = 1;

					float temp_value, temp_weight;
					int temp_fid;
					bool temp_default, temp_is_leaf, temp_exchange;

					temp_value = values_h[k*2+1+tree_offset];
					values_h[k*2+1+tree_offset] = values_h[k*2+2+tree_offset];
					values_h[k*2+2+tree_offset] = temp_value;

					temp_weight = weights_h[k*2+1+tree_offset];
					weights_h[k*2+1+tree_offset] = weights_h[k*2+2+tree_offset];
					weights_h[k*2+2+tree_offset] = temp_weight;

					temp_fid = fids_h[k*2+1+tree_offset];
					fids_h[k*2+1+tree_offset] = fids_h[k*2+2+tree_offset];
					fids_h[k*2+2+tree_offset] = temp_fid;
	
					temp_default = defaults_h[k*2+1+tree_offset];
					defaults_h[k*2+1+tree_offset] = defaults_h[k*2+2+tree_offset];
					defaults_h[k*2+2+tree_offset] = temp_default;
	
					temp_is_leaf = is_leafs_h[k*2+1+tree_offset];
					is_leafs_h[k*2+1+tree_offset] = is_leafs_h[k*2+2+tree_offset];
					is_leafs_h[k*2+2+tree_offset] = temp_is_leaf;

					temp_exchange = exchanges_h[k*2+1+tree_offset];
					exchanges_h[k*2+1+tree_offset] = exchanges_h[k*2+2+tree_offset];
					exchanges_h[k*2+2+tree_offset] = temp_exchange;


					if(((k*2+1)*2+1)<tree_num_nodes(depth_))
					{
						swap_child((k*2+1)*2+1, 1, values_h, weights_h, fids_h, defaults_h, is_leafs_h, exchanges_h, tree_offset);
						swap_child((k*2+1)*2+2, 1, values_h, weights_h, fids_h, defaults_h, is_leafs_h, exchanges_h, tree_offset);
					}
			   }
			}
	}
    }
    }

	int max_fid = 0;
	for(int i=0; i<num_nodes; i++)
		max_fid = (max_fid < fids_h[i])?fids_h[i]:max_fid;

	float fid_len = (log(max_fid)/log(2) + 3 )/8; //+3 is for other bits

	if(fid_len < 1.0 && fid_len > 0.0)
	{
		bits_length = 1;
		adaptive_format_number = 1;
	}
	else if(fid_len < 2.0 && fid_len >= 1.0)
	{
		bits_length = 2;
		adaptive_format_number = 2;
	}
	else if(fid_len < 4.0 && fid_len >= 2.0)
	{
		bits_length = 4;
		adaptive_format_number = 4;
	}
	else
	{
		bits_length = -1;
		adaptive_format_number = -1;
	}

	char ***test = new char**[num_trees_];

	for(int i=0; i<num_trees_; i++)
	{
		test[i] = new char*[num_nodes_per_tree-2];
		for(int j=0; j<num_nodes_per_tree-2; j++)
		{
			test[i][j] = new char[10];
		}
	}


    std::vector<std::pair<ul_int,int>> hash(num_trees_);
	for(int i=0; i<num_trees_; i++)
	{
		hash[i] = std::make_pair(sh_simhash(test[i], num_nodes_per_tree-2), i);
	}

	std::sort(hash.begin(),hash.end());

    std::vector<float> values_reorder_h(num_nodes), weights_reorder_h(num_nodes);
    std::vector<int> fids_reorder_h(num_nodes);
    bool* defaults_reorder_h = new bool[num_nodes];
    bool* is_leafs_reorder_h = new bool[num_nodes];
    bool* exchanges_reorder_h = new bool[num_nodes];

	for(int i=0; i<num_trees_; i++)
	{
		for(int j=0; j<num_nodes_per_tree; j++)
		{
			values_reorder_h[i*num_nodes_per_tree+j] = values_h[(hash[i].second)*num_nodes_per_tree+j];
			weights_reorder_h[i*num_nodes_per_tree+j] = weights_h[(hash[i].second)*num_nodes_per_tree+j];
			fids_reorder_h[i*num_nodes_per_tree+j] = fids_h[(hash[i].second)*num_nodes_per_tree+j];
			defaults_reorder_h[i*num_nodes_per_tree+j] = defaults_h[(hash[i].second)*num_nodes_per_tree+j];
			is_leafs_reorder_h[i*num_nodes_per_tree+j] = is_leafs_h[(hash[i].second)*num_nodes_per_tree+j];
			exchanges_reorder_h[i*num_nodes_per_tree+j] = exchanges_h[(hash[i].second)*num_nodes_per_tree+j];
		}
	}
	
	bits_values_h = new float[num_nodes];
	if(bits_length == 1)
		bits_char_h = new char[num_nodes];
	if(bits_length == 2)
		bits_short_h = new short int[num_nodes];
	if(bits_length == 4)
		bits_int_h = new int[num_nodes];

	encode_node_adaptive(values_reorder_h, fids_reorder_h, defaults_reorder_h, is_leafs_reorder_h, exchanges_reorder_h, bits_values_h, bits_char_h, bits_short_h, bits_int_h, bits_length, num_nodes);

	bits_values_reorg_h = new float[num_nodes];
	if(bits_length == 1)
		bits_char_reorg_h = new char[num_nodes];
	if(bits_length == 2)
		bits_short_reorg_h = new short int[num_nodes];
	if(bits_length == 4)
		bits_int_reorg_h = new int[num_nodes];

	for(int j=0; j<num_nodes_per_tree; j++)
	{
		for(int i=0; i<num_trees_; i++)
		{
			bits_values_reorg_h[j*num_trees_ + i] = bits_values_h[i*num_nodes_per_tree + j];
			if(bits_length == 1)
				bits_char_reorg_h[j*num_trees_ + i] = bits_char_h[i*num_nodes_per_tree + j];
			if(bits_length == 2)
				bits_short_reorg_h[j*num_trees_ + i] = bits_short_h[i*num_nodes_per_tree + j];
			if(bits_length == 4)
				bits_int_reorg_h[j*num_trees_ + i] = bits_int_h[i*num_nodes_per_tree + j];
		}
	}

	//printf_float_CPU(bits_values_h, num_nodes);
	//printf_float_CPU(bits_values_org_h, num_nodes);

	allocate(bits_values_d, num_nodes);
	if(bits_length == 1)
		allocate(bits_char_d, num_nodes);
	if(bits_length == 2)
		allocate(bits_short_d, num_nodes);
	if(bits_length == 4)
		allocate(bits_int_d, num_nodes);

	updateDevice(bits_values_d, bits_values_h, num_nodes, stream);	
	if(bits_length == 1)
		updateDevice(bits_char_d, bits_char_h, num_nodes, stream);	
	if(bits_length == 2)
		updateDevice(bits_short_d, bits_short_h, num_nodes, stream);	
	if(bits_length == 4)
		updateDevice(bits_int_d, bits_int_h, num_nodes, stream);	



	allocate(bits_values_reorg_d, num_nodes);
	if(bits_length == 1)
		allocate(bits_char_reorg_d, num_nodes);
	if(bits_length == 2)
		allocate(bits_short_reorg_d, num_nodes);
	if(bits_length == 4)
		allocate(bits_int_reorg_d, num_nodes);

	updateDevice(bits_values_reorg_d, bits_values_reorg_h, num_nodes, stream);	
	if(bits_length == 1)
		updateDevice(bits_char_reorg_d, bits_char_reorg_h, num_nodes, stream);	
	if(bits_length == 2)
		updateDevice(bits_short_reorg_d, bits_short_reorg_h, num_nodes, stream);	
	if(bits_length == 4)
		updateDevice(bits_int_reorg_d, bits_int_reorg_h, num_nodes, stream);	


    values_h.clear();
    weights_h.clear();
    fids_h.clear();
    delete[] defaults_h;
    delete[] is_leafs_h;
    delete[] exchanges_h;

    values_reorder_h.clear();
    weights_reorder_h.clear();
    fids_reorder_h.clear();
    delete[] defaults_reorder_h;
    delete[] is_leafs_reorder_h;
    delete[] exchanges_reorder_h;

	for(int i=0; i<num_trees_; i++)
	{
		for(int j=0; j<num_nodes_per_tree-2; j++)
			delete test[i][j];
		delete test[i];
	}
	delete test;


  }





  void infer_adaptive(predict_params params, cudaStream_t stream) {
  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);

  if(adaptive_format_number == 1)
	infer_adaptive_reorg_char<1><<<num_blocks, FIL_TPB, 0, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d);
  if(adaptive_format_number == 2)
	infer_adaptive_reorg_short<1><<<num_blocks, FIL_TPB, 0, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d);
  if(adaptive_format_number == 4)
	infer_adaptive_reorg_int<1><<<num_blocks, FIL_TPB, 0, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d);
  CUDA_CHECK(cudaPeekAtLastError());
}






  void infer_dense_shared_data_adaptive(predict_params params, cudaStream_t stream) {
  int num_items = params.max_shm / (sizeof(float) * params.num_cols) ;
  if (num_items == 0) {
    assert(false && "too many features");
  }
  num_items = 1;
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;

  if(adaptive_format_number == 1)
	infer_k_shared_data_adaptive_reorg_char<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d);
  if(adaptive_format_number == 2)
	infer_k_shared_data_adaptive_reorg_short<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d);
  if(adaptive_format_number == 4)
	infer_k_shared_data_adaptive_reorg_int<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d);

  CUDA_CHECK(cudaPeekAtLastError());
}



  void infer_dense_shared_data_wo_adaptive(predict_params params, cudaStream_t stream) {
  int num_items = params.max_shm / (sizeof(float) * params.num_cols) ;
  if (num_items == 0) {
    assert(false && "too many features");
  }
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;
  //printf("num_items %d, num_blocks %d\n", num_items, num_blocks);

  if(adaptive_format_number == 1)
	infer_k_shared_data_wo_adaptive_reorg_char<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d, num_items);
  if(adaptive_format_number == 2)
	infer_k_shared_data_wo_adaptive_reorg_short<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d, num_items);
  if(adaptive_format_number == 4)
	infer_k_shared_data_wo_adaptive_reorg_int<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d, num_items);

  CUDA_CHECK(cudaPeekAtLastError());
}



  void infer_dense_shared_forest_adaptive(predict_params params, cudaStream_t stream) {
  if(adaptive_format_number == 1)
  {
	  int shm_sz = num_trees_ * (sizeof(char)+sizeof(float)) * tree_num_nodes(depth_);
	  //printf("shared memory is %d\n", params.max_shm);
	  if (shm_sz > params.max_shm) {
	    assert(false && "forest is too large to save in shared memory");
	  }
	
	  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);
	  //printf("%d - %d\n", num_blocks, FIL_TPB);
	  infer_k_shared_forest_adaptive_reorg_char<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d);
	  //printf("end infer_k\n");
	  CUDA_CHECK(cudaPeekAtLastError());
  }
  if(adaptive_format_number == 2)
  {
	  int shm_sz = num_trees_ * (sizeof(short)+sizeof(float)) * tree_num_nodes(depth_);
	  //printf("shared memory is %d\n", params.max_shm);
	  //printf("shm_sz is %d\n", shm_sz);
	  //printf("num_trees_ %d tree_num_nodes %d\n", num_trees_, tree_num_nodes(depth_));
	  if (shm_sz > params.max_shm) {
	    assert(false && "forest is too large to save in shared memory");
	  }
	
	  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);
	  //printf("%d - %d\n", num_blocks, FIL_TPB);
	  infer_k_shared_forest_adaptive_reorg_short<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d);
	  //printf("end infer_k\n");
	  CUDA_CHECK(cudaPeekAtLastError());
  }
  if(adaptive_format_number == 4)
  {
	  int shm_sz = num_trees_ * (sizeof(int)+sizeof(float)) * tree_num_nodes(depth_);
	  //printf("shared memory is %d\n", params.max_shm);
	  if (shm_sz > params.max_shm) {
	    assert(false && "forest is too large to save in shared memory");
	  }
	
	  int num_blocks = ceildiv(int(params.num_rows), FIL_TPB);
	  //printf("%d - %d\n", num_blocks, FIL_TPB);
	  infer_k_shared_forest_adaptive_reorg_int<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d);
	  //printf("end infer_k\n");
	  CUDA_CHECK(cudaPeekAtLastError());
  }


}




void infer_dense_split_forest_adaptive(predict_params params, cudaStream_t stream) {
  int trees_per_sm = 0;
  if(adaptive_format_number == 1)
	trees_per_sm = params.max_shm / ( (sizeof(char)+sizeof(float)) * tree_num_nodes(depth_) );
  if(adaptive_format_number == 2)
	trees_per_sm = params.max_shm / ( (sizeof(short)+sizeof(float)) * tree_num_nodes(depth_) );
  if(adaptive_format_number == 4)
	trees_per_sm = params.max_shm / ( (sizeof(int)+sizeof(float)) * tree_num_nodes(depth_) );
  //printf("shared memory is %d\n", params.max_shm);
  //printf("adaptive_format_number is %d\n", adaptive_format_number);
  if (trees_per_sm == 0) {
    assert(false && "single tree is too big to save in shared memory");
    //printf("single tree is too big to save in shared memory\n");
    //return;
  }
  int num_blocks = ceildiv(num_trees_, trees_per_sm);
  int shm_sz = 0;
  if(adaptive_format_number == 1)
	shm_sz = trees_per_sm * ( (sizeof(char)+sizeof(float)) * tree_num_nodes(depth_) );////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!
  if(adaptive_format_number == 2)
	shm_sz = trees_per_sm * ( (sizeof(short)+sizeof(float)) * tree_num_nodes(depth_) );////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!
  if(adaptive_format_number == 4)
	shm_sz = trees_per_sm * ( (sizeof(int)+sizeof(float)) * tree_num_nodes(depth_) );////////////////////////////DO NOT KNOW WHY!!!!!!!!!!!!!

  float* temp_out = NULL;
  allocate(temp_out, num_blocks*params.num_rows);

  if(adaptive_format_number == 1)
  infer_k_split_forest_adaptive_reorg_char<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, trees_per_sm, temp_out, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_char_reorg_d);
  if(adaptive_format_number == 2)
  infer_k_split_forest_adaptive_reorg_short<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, trees_per_sm, temp_out, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_short_reorg_d);
  if(adaptive_format_number == 4)
  infer_k_split_forest_adaptive_reorg_int<<<num_blocks, FIL_TPB, shm_sz, stream>>>(params, trees_per_sm, temp_out, num_trees_, tree_num_nodes(depth_), bits_values_reorg_d, bits_int_reorg_d);


  int* h_offsets=(int*)malloc(sizeof(int)*(params.num_rows+1));
  for(int i=0;i<=params.num_rows;i++)
	  h_offsets[i]=i*num_blocks;

  int* d_offsets;
  allocate(d_offsets, (params.num_rows+1));
  updateDevice(d_offsets, h_offsets, (params.num_rows+1), stream);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sum-reduction
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, temp_out, params.preds, params.num_rows, d_offsets, d_offsets + 1);

  CUDA_CHECK(cudaFree(d_temp_storage));

  free(h_offsets);
  CUDA_CHECK(cudaFree(d_offsets));

  CUDA_CHECK(cudaFree(temp_out));

  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}



  virtual void infer(predict_params params, cudaStream_t stream) override {
		if(selected_algorithm==0)
			infer_adaptive(params, stream);
		if(selected_algorithm==1)
			infer_dense_shared_data_wo_adaptive(params, stream);
		if(selected_algorithm==2)
			infer_dense_shared_data_adaptive(params, stream);
		if(selected_algorithm==3)
			infer_dense_shared_forest_adaptive(params, stream);
		if(selected_algorithm==4)
			infer_dense_split_forest_adaptive(params, stream);
  } 






  float* bits_values_d = nullptr;
  int* bits_int_d = nullptr;
  short int* bits_short_d = nullptr;
  char* bits_char_d = nullptr;

  float* bits_values_h = nullptr;
  int* bits_int_h = nullptr;
  short int* bits_short_h = nullptr;
  char* bits_char_h = nullptr;


  float* bits_values_reorg_d = nullptr;
  int* bits_int_reorg_d = nullptr;
  short int* bits_short_reorg_d = nullptr;
  char* bits_char_reorg_d = nullptr;

  float* bits_values_reorg_h = nullptr;
  int* bits_int_reorg_h = nullptr;
  short int* bits_short_reorg_h = nullptr;
  char* bits_char_reorg_h = nullptr;


  int bits_length = 0;

};






template <int NITEMS>
__device__ __forceinline__ void infer_one_tree_sparse(sparse_tree tree, float* sdata,
                                               int cols, vec<NITEMS>& out, algo_t algo_, int num_trees) {
  unsigned int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      float n_val = tree.nodes_[curr[j]].val;
      int n_bits = tree.nodes_[curr[j]].bits;
	  //float n_output = n_val;
	  float n_thresh = n_val;
	  int n_fid = n_bits & ((1 << 30) - 1);
	  bool n_def_left = n_bits & (1 << 30);
	  bool n_is_leaf = n_bits & (1 << 31);
	  //printf("\nn_val: %f, n_bits: %d, n_fid: %d, n_def_left: %d, n_is_leaf: %d\n", n_val, n_bits, n_fid, n_def_left, n_is_leaf);
      if (n_is_leaf) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n_fid];
      bool cond = isnan(val) ? !n_def_left : val >= n_thresh;
	  if(algo_ == algo_t::NAIVE)
	  {
		  //curr[j] = curr[j] + 1 + cond;
		  curr[j] = tree.nodes_[curr[j]].left_idx + cond;
	  }
    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += tree.nodes_[curr[j]].val;
}


template <int NITEMS>
__global__ void infer_k(sparse_storage forest, predict_params params) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;
  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] =
        row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;
  // one block works on a single row and the whole forest
  for (int j = threadIdx.x; j < forest.num_trees(); j += blockDim.x) {
    infer_one_tree_sparse<NITEMS>(forest[j], sdata, params.num_cols, out, params.algo, forest.num_trees());
  }

  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);

  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < params.num_rows)
	  {
		  params.preds[idx] = out[i];
	  }
    }
  }

}


void infer_sparse(sparse_storage forest, predict_params params, cudaStream_t stream) {
  const int MAX_BATCH_ITEMS = 4;
  params.max_items =
    params.algo == algo_t::BATCH_TREE_REORG ? MAX_BATCH_ITEMS : 1;
  int num_items = params.max_shm / (sizeof(float) * params.num_cols);
  //printf("shared memory is %d\n", params.max_shm);
  if (num_items == 0) {
    //int max_cols = params.max_shm / sizeof(float);
    assert(false && "too many features");
  }
  num_items = std::min(num_items, params.max_items);
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;
  //printf("start infer_k: %d, num_blocks: %d, FIL_TPB: %d, shm_sz: %d, stream: %d\n", num_items, num_blocks, FIL_TPB, shm_sz, stream);
  switch (num_items) {
    case 1:
      infer_k<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 2:
      infer_k<2><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 3:
      infer_k<3><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 4:
      infer_k<4><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    default:
      assert(false && "internal error: nitems > 4");
  }
  //printf("end infer_k\n");
  CUDA_CHECK(cudaPeekAtLastError());
}


struct sparse_forest : forest {

  void init(const cudaStream_t stream, const int* trees, const sparse_node_t* nodes,
            const forest_params_t* params) {
    init_common(params);
    depth_ = 0;  // a placeholder value
    num_nodes_ = params->num_nodes;

    // trees
    allocate(trees_, num_trees_);
    CUDA_CHECK(cudaMemcpyAsync(trees_, trees, sizeof(int) * num_trees_, cudaMemcpyHostToDevice, stream));

    // nodes
	allocate(nodes_, num_nodes_);
    CUDA_CHECK(cudaMemcpyAsync(nodes_, nodes, sizeof(sparse_node_t) * num_nodes_, cudaMemcpyHostToDevice, stream));
  } 

  virtual void infer(predict_params params, cudaStream_t stream) override {
    sparse_storage forest(trees_, nodes_, num_trees_);
    infer_sparse(forest, params, stream);
  } 
 
  int num_nodes_ = 0;
  int* trees_ = nullptr;
  sparse_node_t* nodes_ = nullptr;

};


#endif
