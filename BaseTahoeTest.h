#ifndef BASEFILTEST_H
#define BASEFILTEST_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cassert>
#include "Struct.h"
#include <sys/time.h>


void printf_dense_node(std::vector<dense_node_t> nodes)
{
	for(int i=0; i<nodes.size(); i++)
		printf("%f, %f, %d, ", nodes[i].val, nodes[i].weight, nodes[i].bits);
	printf("\n");
}

__global__ void printf_single_float(float* in)
{
	printf("%f, ", in[0]);
}

__global__ void printf_single_int(int* in)
{
	printf("%d, ", in[0]);
}

void printf_dense_forest_GPU(dense_forest* f)
{
	printf("forest num_trees_:%d, depth_=%d, num_cols_=%d \n", f->num_trees_, f->depth_, f->num_cols_);
	int num_nodes = forest_num_nodes(f->num_trees_, f->depth_);
	for(int i=0; i<num_nodes; i++)
	{
		printf_single_float<<<1,1>>>(&f->nodes_[i].val);
		printf_single_int<<<1,1>>>(&f->nodes_[i].bits);
	}
	printf("\n");
}


class BaseTahoeTest
{

	public:
	BaseTahoeTest(std::string input_model_file, std::string input_data_file, int algorithm, int num_rows=1000, int num_cols=500, float nan_prob=0.0, int depth=20, int num_trees=10, float leaf_prob=0.0, output_t output=output_t::RAW, float threshold=0.0, float global_bias=0.0, algo_t algo=algo_t::NAIVE, int seed=0, float tolerance=1e-3f, strategy_t strategy=strategy_t::SHARED_DATA)
	{
    	ps.num_rows = num_rows;
    	ps.num_cols = num_cols;
    	ps.nan_prob = nan_prob;
    	ps.depth = depth;
    	ps.num_trees = num_trees;
    	ps.leaf_prob = leaf_prob;
    	ps.output = output;
    	ps.threshold = threshold;
    	ps.global_bias = global_bias;
    	ps.algo = algo;
    	ps.seed = seed;
    	ps.tolerance = tolerance;
    	ps.strategy = strategy;
	input_model_file.copy(ps.input_model_file, input_model_file.length(), 0);
	*(ps.input_model_file+input_model_file.length())='\0';
	input_data_file.copy(ps.input_data_file, input_data_file.length(), 0);
	*(ps.input_data_file+input_data_file.length())='\0';
	selected_algorithm = algorithm;
	}

  int SetUp(float &speedup){

    float acc[6];
    // setup

	//ps = testing::TestWithParam<FilTestParams>::GetParam();

    CUDA_CHECK(cudaStreamCreate(&stream));

	printf("Load forest\n");
    //generate_forest();
    generate_forest_from_file();
	printf("Load data\n");
    //generate_data();
    generate_data_from_file();
	printf("Predict on CPU and get standard results...\n");
    predict_on_cpu();
	printf("Test on GPU...\n");
    float baseline = predict_on_gpu_dense();
    predict_on_gpu_dense_adaptive(acc);
    //predict_on_gpu_sparse();
    
    int algorithm = 0;
    float time = FLT_MAX;
    if( acc[0] < time ){
	algorithm = 1;
	time = acc[0];}
    if( acc[1] < time ){
	algorithm = 1;
	time = acc[1];}
    if( acc[2] < time ){
	algorithm = 2;
	time = acc[2];}
    if( acc[3] < time ){
	algorithm = 3;
	time = acc[3];}
    if( acc[4] < time ){
	algorithm = 4;
	time = acc[4];}

    speedup = baseline/time;

    return algorithm;

  }

  void TearDown(){
    CUDA_CHECK(cudaFree(preds_d));
    CUDA_CHECK(cudaFree(want_preds_d));
    CUDA_CHECK(cudaFree(data_d));
  }

  void generate_float_vec(std::vector<float> &data, int num){
	for(int i=0;i<num;i++)
	{
		float x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		x = x*2.0 -1.0;
		data[i] = x;
	}
  }

  void dense_node_init(dense_node_t* n, int fid, float value, bool def_left,
                     float weight, bool is_leaf) {
	encode_node(n, fid, value, def_left, weight, is_leaf);
  }
  
  void generate_int_vec(std::vector<int> &data, int num, int max){
	for(int i=0;i<num;i++)
	{
		int x = std::rand()%max;
		data[i] = x;
	}
  }

  void generate_bool(bool* data, int num, float prob){
	for(int i=0;i<num;i++)
	{
		data[i]=(std::rand()<(RAND_MAX*prob))?true:false;
	}
  }

  void generate_forest() {
    size_t num_nodes = forest_num_nodes();

    // helper data
    float* weights_d = nullptr;
    float* thresholds_d = nullptr;
    int* fids_d = nullptr;
    bool* def_lefts_d = nullptr;
    bool* is_leafs_d = nullptr;
    bool* def_lefts_h = nullptr;
    bool* is_leafs_h = nullptr;

    // allocate GPU data
    allocate(weights_d, num_nodes);
    allocate(thresholds_d, num_nodes);
    allocate(fids_d, num_nodes);
    allocate(def_lefts_d, num_nodes);
    allocate(is_leafs_d, num_nodes);

    // copy data to host
    std::vector<float> weights_h(num_nodes), thresholds_h(num_nodes);
    std::vector<int> fids_h(num_nodes);
    def_lefts_h = new bool[num_nodes];
    is_leafs_h = new bool[num_nodes];

	std::srand(ps.seed);
	generate_float_vec(weights_h, num_nodes);
	generate_float_vec(thresholds_h, num_nodes);
	generate_int_vec(fids_h, num_nodes, ps.num_cols);
	generate_bool(def_lefts_h, num_nodes, 0.5);
	generate_bool(is_leafs_h, num_nodes, ps.leaf_prob);

/*
	for(int i=0;i<ps.num_trees;i++)
	{
		for(int j=0;j<tree_num_nodes();j++)
		{
			weights_h[i*tree_num_nodes()+j]=weights_h[j];
			thresholds_h[i*tree_num_nodes()+j]=thresholds_h[j];
			fids_h[i*tree_num_nodes()+j]=fids_h[j];
			def_lefts_h[i*tree_num_nodes()+j]=def_lefts_h[j];
			is_leafs_h[i*tree_num_nodes()+j]=is_leafs_h[j];
		}
	}
*/

	//for (std::vector<float>::iterator it=weights_h.begin(); it != weights_h.end(); it++)
	//	    std::cout << *it<< " ";

	updateDevice(weights_d, weights_h.data(), num_nodes, stream);	
	updateDevice(thresholds_d, thresholds_h.data(), num_nodes, stream);	
	updateDevice(fids_d, fids_h.data(), num_nodes, stream);	
	updateDevice(def_lefts_d, def_lefts_h, num_nodes, stream);	
	updateDevice(is_leafs_d, is_leafs_h, num_nodes, stream);	

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// mark leaves
    for (size_t i = 0; i < ps.num_trees; ++i) {
      int num_tree_nodes = tree_num_nodes();
      size_t leaf_start = num_tree_nodes * i + num_tree_nodes / 2;
      size_t leaf_end = num_tree_nodes * (i + 1);
      for (size_t j = leaf_start; j < leaf_end; ++j) {
        is_leafs_h[j] = true;
      }
    }

	// initialize nodes
    nodes.resize(num_nodes);
    for(int i=0;i<ps.num_trees;i++)
    {
      for(int j=0;j<tree_num_nodes();j++)
      {	
      //for (int i = 0; i < num_nodes; ++i) {
        //dense_node_init(&nodes[i*tree_num_nodes()+j], weights_h[j], thresholds_h[j], fids_h[j],
        //                   def_lefts_h[j], is_leafs_h[j]);
        dense_node_init(&nodes[i*tree_num_nodes()+j], weights_h[i*tree_num_nodes()+j], thresholds_h[i*tree_num_nodes()+j], fids_h[i*tree_num_nodes()+j],
                           def_lefts_h[i*tree_num_nodes()+j], is_leafs_h[i*tree_num_nodes()+j]);
      }
    }

#if _DEBUG_
	printf("dense_weight on CPU:\n");
	printf_float_CPU(weights_h.data(), num_nodes);
	printf("dense_threshold on CPU:\n");
	printf_float_CPU(thresholds_h.data(), num_nodes);
	printf("dense_fid on CPU:\n");
	printf_int_CPU(fids_h.data(), num_nodes);
	printf("dense_def on CPU:\n");
	printf_bool_CPU(def_lefts_h, num_nodes);
	printf("dense_is_leaf on CPU:\n");
	printf_bool_CPU(is_leafs_h, num_nodes);

	printf("dense_node on CPU:\n");
	printf_dense_node(nodes);
#endif

	// clean up
    delete[] def_lefts_h;
    delete[] is_leafs_h;
    CUDA_CHECK(cudaFree(is_leafs_d));
    CUDA_CHECK(cudaFree(def_lefts_d));
    CUDA_CHECK(cudaFree(fids_d));
    CUDA_CHECK(cudaFree(thresholds_d));
    CUDA_CHECK(cudaFree(weights_d));

	weights_h.clear();
	thresholds_h.clear();
	fids_h.clear();

  }




void generate_forest_from_file() {

int MAX_LINE=1024;
char buf[MAX_LINE];
FILE *fp;

if((fp = fopen(ps.input_model_file,"r")) == NULL)
{
	perror("fail to read");
	exit (1) ;
}

if(fgets(buf,MAX_LINE,fp))
    ps.num_trees = atoi(buf);
if(fgets(buf,MAX_LINE,fp))
    ps.depth = atoi(buf)-1;

    //printf("%d, %d\n", ps.num_trees, ps.depth);

    size_t num_nodes = forest_num_nodes();

    // copy data to host
    std::vector<float> values_h(num_nodes), weights_h(num_nodes);
    std::vector<int> fids_h(num_nodes);
    bool* defaults_h = new bool[num_nodes];
    bool* is_leafs_h = new bool[num_nodes];

    //printf("%d, %d\n", ps.num_trees, tree_num_nodes());

    for (size_t i = 0; i < ps.num_trees; ++i) {
    	for (size_t j = 0; j < tree_num_nodes(); ++j) {
	    fgets(buf,MAX_LINE,fp);
	    int fid_temp = atoi(buf);
	    fgets(buf,MAX_LINE,fp);
	    float value_temp = atof(buf);
	    fgets(buf,MAX_LINE,fp);
	    bool default_temp = atoi(buf);
	    fgets(buf,MAX_LINE,fp);
	    float weight_temp = atof(buf);
	    fgets(buf,MAX_LINE,fp);
	    bool is_leaf_temp = atoi(buf);
	    //printf("%d, %f, %d, %f, %d\n", fid_temp, value_temp, default_temp, weight_temp, is_leaf_temp);
	    fids_h[i*tree_num_nodes()+j]=fid_temp;
	    values_h[i*tree_num_nodes()+j]=value_temp;
	    defaults_h[i*tree_num_nodes()+j]=default_temp;
	    weights_h[i*tree_num_nodes()+j]=weight_temp;
	    is_leafs_h[i*tree_num_nodes()+j]=is_leaf_temp;
    	}
    }

    fclose(fp);

    // initialize nodes
    nodes.resize(num_nodes);
    for(int i=0;i<ps.num_trees;i++)
    {
      for(int j=0;j<tree_num_nodes();j++)
      {	
        dense_node_init(&nodes[i*tree_num_nodes()+j], fids_h[i*tree_num_nodes()+j], values_h[i*tree_num_nodes()+j],
                           defaults_h[i*tree_num_nodes()+j], weights_h[i*tree_num_nodes()+j], is_leafs_h[i*tree_num_nodes()+j]);
      }
    }

#if _DEBUG_
	printf("dense_value on CPU:\n");
	printf_float_CPU(values_h.data(), num_nodes);
	printf("dense_weight on CPU:\n");
	printf_float_CPU(weights_h.data(), num_nodes);
	printf("dense_fid on CPU:\n");
	printf_int_CPU(fids_h.data(), num_nodes);
	printf("dense_def on CPU:\n");
	printf_bool_CPU(defaults_h, num_nodes);
	printf("dense_is_leaf on CPU:\n");
	printf_bool_CPU(is_leafs_h, num_nodes);

	printf("dense_node on CPU:\n");
	printf_dense_node(nodes);
#endif

    // clean up
    delete[] defaults_h;
    delete[] is_leafs_h;
    weights_h.clear();
    values_h.clear();
    fids_h.clear();
}

void generate_data_from_file() {

int MAX_LINE=1024;
char buf[MAX_LINE];
FILE *fp;

if((fp = fopen(ps.input_data_file,"r")) == NULL)
{
	perror("fail to read");
	exit (1) ;
}

if(fgets(buf,MAX_LINE,fp))
    ps.num_rows = atoi(buf);
if(fgets(buf,MAX_LINE,fp))
    ps.num_cols = atoi(buf);
if(fgets(buf,MAX_LINE,fp))
    ps.missing = atof(buf);

ps.num_rows = ps.num_rows/128;///////////////////////////////////////////////////////////////MUST BE DELETED!!!!!
    //printf("%d, %d, %f\n", ps.num_rows, ps.num_cols, ps.missing);

    // allocate arrays
    size_t num_data = ps.num_rows * ps.num_cols;
    allocate(data_d, num_data);

    data_h.resize(num_data);

    for (size_t i = 0; i < ps.num_rows; ++i) {
    	for (size_t j = 0; j < ps.num_cols; ++j) {
	    fgets(buf,MAX_LINE,fp);
	    float value_temp = atof(buf);
	    data_h[i*ps.num_cols+j]=value_temp;
    	}
    }

    fclose(fp);

    updateDevice(data_d, data_h.data(), num_data, stream);	
    CUDA_CHECK(cudaPeekAtLastError());

#if _DEBUG_
	printf("data on CPU:\n");
	printf_float_CPU(data_h.data(), data_h.size());

	printf("data on GPU:\n");
	printf_float_GPU<<<1,1>>>(data_d, num_data);
#endif
}

void generate_data() {

    // allocate arrays
    size_t num_data = ps.num_rows * ps.num_cols;
    allocate(data_d, num_data);
    bool* mask_d = nullptr;
    allocate(mask_d, num_data);

    data_h.resize(num_data);
	bool* mask_h = new bool[num_data];
	generate_float_vec(data_h, num_data);
	generate_bool(mask_h, num_data, 1.0-ps.nan_prob);
	updateDevice(data_d, data_h.data(), num_data, stream);	
	updateDevice(mask_d, mask_h, num_data, stream);	
    // generate random data
    int tpb = 256;
    nan_kernel<<<ceildiv(int(num_data), tpb), tpb, 0, stream>>>(data_d, mask_d, num_data, std::numeric_limits<float>::quiet_NaN());
    CUDA_CHECK(cudaPeekAtLastError());

    // copy to host
    updateHost(data_h.data(), data_d, num_data, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

#if _DEBUG_
	printf("data on CPU:\n");
	printf_float_CPU(data_h.data(), data_h.size());

	printf("data on GPU:\n");
	printf_float_GPU<<<1,1>>>(data_d, num_data);
#endif

    // clean up
	delete mask_h;
    CUDA_CHECK(cudaFree(mask_d));
  }

  float infer_one_tree(dense_node_t* root, float* data) {
    int curr = 0;
    float value = 0.0f;
    int fid = 0;
    bool def_left = false, is_leaf = false;
    float weight = 0.0f;
    for (;;) {
      dense_node_decode(&root[curr], &value, &weight, &fid, &def_left,
                             &is_leaf);
      if (is_leaf) break;
      float val = data[fid];
      const float eps=1.0e-6;
      bool cond = (fabs(val-ps.missing)<=eps) ? !def_left : val >= value;
      curr = (curr << 1) + 1 + (cond ? 1 : 0);
    }
    return value;
  }

  void predict_on_cpu() {
    // predict on host
    std::vector<float> want_preds_h(ps.num_rows);
    int num_nodes = tree_num_nodes();
    for (int i = 0; i < ps.num_rows; ++i) {
      float pred = 0.0f;
      for (int j = 0; j < ps.num_trees; ++j) {
        pred += infer_one_tree(&nodes[j * num_nodes], &data_h[i * ps.num_cols]);
      }
      if ((ps.output & output_t::AVG) != 0) pred = pred / ps.num_trees;
      pred += ps.global_bias;
      if ((ps.output & output_t::SIGMOID) != 0) pred = sigmoid(pred);
      if ((ps.output & output_t::THRESHOLD) != 0) {
        pred = pred > ps.threshold ? 1.0f : 0.0f;
      }
      want_preds_h[i] = pred;
    }

    // copy to GPU
    allocate(want_preds_d, ps.num_rows);
    updateDevice(want_preds_d, want_preds_h.data(), ps.num_rows, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

#if _DEBUG_
	printf("Predict on CPU:\n");
	printf_float_CPU(want_preds_h.data(), want_preds_h.size());
#endif
    cudaStreamSynchronize(stream);

  }


  void check_params(const forest_params_t* params, bool dense) {
  if (dense) {
    assert(params->depth >= 0 && "depth must be non-negative for dense forests");
  } else {
    assert(params->num_nodes >= 0 &&
           "num_nodes must be non-negative for sparse forests");
    assert(params->algo == algo_t::NAIVE &&
           "only NAIVE algorithm is supported for sparse forests");
  }
  assert(params->num_trees >= 0 && "num_trees must be non-negative");
  assert(params->num_cols >= 0 && "num_cols must be non-negative");
  switch (params->algo) {
    case algo_t::NAIVE:
    case algo_t::TREE_REORG:
    case algo_t::BATCH_TREE_REORG:
      break;
    default:
      assert(false && "aglo should be NAIVE, TREE_REORG or BATCH_TREE_REORG");
  }
  // output_t::RAW == 0, and doesn't have a separate flag
  output_t all_set =
    output_t(output_t::AVG | output_t::SIGMOID | output_t::THRESHOLD);
  if ((params->output & ~all_set) != 0) {
    assert(false &&
           "output should be a combination of RAW, AVG, SIGMOID and THRESHOLD");
  }
  }


  void init_dense(const cudaStream_t stream, dense_forest** pf, const dense_node_t* nodes,
                const forest_params_t* params) {
  check_params(params, true);
  dense_forest* f = new dense_forest;
  f->init(stream, nodes, params);
  *pf = f;
  }

  void init_forest_dense(dense_forest** pforest){
    // init FIL model
    forest_params_t fil_ps;
    fil_ps.depth = ps.depth;
    fil_ps.num_trees = ps.num_trees;
    fil_ps.num_cols = ps.num_cols;
    fil_ps.algo = ps.algo;
    fil_ps.output = ps.output;
    fil_ps.threshold = ps.threshold;
    fil_ps.global_bias = ps.global_bias;
    fil_ps.strategy = ps.strategy;

    fil_ps.missing = ps.missing;
	//printf("parameters: %d, %d, %d, %d, %d, %f, %f\n", fil_ps.depth, fil_ps.num_trees, fil_ps.num_cols, fil_ps.algo, fil_ps.output, fil_ps.threshold, fil_ps.global_bias);
    init_dense(stream, pforest, nodes.data(), &fil_ps);
  }

  void predict_dense(cudaStream_t stream, dense_forest* f, float* preds, const float* data,
             size_t num_rows) {
  	f->predict(stream, preds, data, num_rows);
  } 

  float predict_on_gpu_dense() {
    dense_forest* forest = nullptr;
    init_forest_dense(&forest);

    // predict
    allocate(preds_d, ps.num_rows);
    CUDA_CHECK(cudaPeekAtLastError());

	/*
    predict_dense(stream, forest, preds_d, data_d, ps.num_rows);
	*/

	for(int i=0;i<5;i++)
    predict_dense(stream, forest, preds_d, data_d, ps.num_rows);


	struct timeval start;
	struct timeval end;
	gettimeofday(&start,NULL);
	cudaDeviceSynchronize();
	for(int i=0;i<5;i++)
    predict_dense(stream, forest, preds_d, data_d, ps.num_rows);
	cudaDeviceSynchronize();
	gettimeofday(&end,NULL);

	float time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	printf("time_dense is %f us\n",time_use/ps.num_rows/5);

    CUDA_CHECK(cudaPeekAtLastError());

#if _DEBUG_
	float* output = new float[ps.num_rows];
	updateHost(output, preds_d, ps.num_rows, stream);
	printf("\nPredict on GPU:\n");
	printf_float_CPU(output, ps.num_rows);
	delete output;
#endif

	cudaDeviceSynchronize();
	compare_GPU<<<1,1,0,stream>>>(preds_d, want_preds_d, ps.num_rows);
	cudaDeviceSynchronize();

    // cleanup
    delete forest;
    return time_use/ps.num_rows/5;
  }


  void predict_dense_adaptive(cudaStream_t stream, dense_adaptive_forest* f, float* preds, const float* data,
             size_t num_rows) {
  	f->predict(stream, preds, data, num_rows);
  } 


  void init_dense_adaptive(const cudaStream_t stream, dense_adaptive_forest** pf, const dense_node_t* nodes,
                const forest_params_t* params) {
  check_params(params, true);
  dense_adaptive_forest* f = new dense_adaptive_forest;
  f->init(stream, nodes, params);
  *pf = f;
  }

  void init_forest_dense_adaptive(dense_adaptive_forest** pforest){
    // init FIL model
    forest_params_t fil_ps;
    fil_ps.depth = ps.depth;
    fil_ps.num_trees = ps.num_trees;
    fil_ps.num_cols = ps.num_cols;
    fil_ps.algo = ps.algo;
    fil_ps.output = ps.output;
    fil_ps.threshold = ps.threshold;
    fil_ps.global_bias = ps.global_bias;
    fil_ps.strategy = ps.strategy;

    fil_ps.missing = ps.missing;
    //printf("parameters: %d, %d, %d, %d, %d, %f, %f\n", fil_ps.depth, fil_ps.num_trees, fil_ps.num_cols, fil_ps.algo, fil_ps.output, fil_ps.threshold, fil_ps.global_bias);
    init_dense_adaptive(stream, pforest, nodes.data(), &fil_ps);
  }




  void predict_on_gpu_dense_adaptive(float* acc) {
    dense_adaptive_forest* forest = nullptr;
    init_forest_dense_adaptive(&forest);

    // predict
    allocate(preds_d, ps.num_rows);
    CUDA_CHECK(cudaPeekAtLastError());

/*
    predict_dense_adaptive(stream, forest, preds_d, data_d, ps.num_rows);
*/


        int device = 0;
	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev, device);
	//printf( "Shared mem per mp: %d\n", (int)dev.sharedMemPerBlock );
	int max_shm = (int)dev.sharedMemPerBlock * 0.8;


	for(int loop=0; loop<=4; loop++)
	{

	selected_algorithm = loop;

	if(loop==3)
	{
	  int shm_sz = ps.num_trees * (sizeof(char)+sizeof(float)) * tree_num_nodes();
	  if (shm_sz > max_shm) {
		acc[loop] = FLT_MAX;
		continue;
	  }
	
	}

	int strategy=loop;

	if(strategy==0 || strategy==1)
	{
		strategy=0;
	}
	else
	{
		strategy--;
	}
	printf("Using strategy %d\n", strategy);

	for(int i=0;i<5;i++)
    	predict_dense_adaptive(stream, forest, preds_d, data_d, ps.num_rows);

	struct timeval start;
	struct timeval end;
	gettimeofday(&start,NULL);
	cudaDeviceSynchronize();
	for(int i=0;i<5;i++)
    predict_dense_adaptive(stream, forest, preds_d, data_d, ps.num_rows);
	cudaDeviceSynchronize();
	gettimeofday(&end,NULL);


	float time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	printf("time_dense is %f us\n",time_use/ps.num_rows/5);

	cudaDeviceSynchronize();
	compare_GPU<<<1,1,0,stream>>>(preds_d, want_preds_d, ps.num_rows);
	cudaDeviceSynchronize();

	acc[loop] = time_use/ps.num_rows/5;

	}

    // cleanup
    delete forest;
  }








  void sparse_node_init(sparse_node_t* node, float output, float thresh, int fid,
                      bool def_left, bool is_leaf, int left_index) {
  node->val = is_leaf ? output : thresh;
  node->bits = (fid & FID_MASK) | (def_left ? DEF_LEFT_MASK : 0) | (is_leaf ? IS_LEAF_MASK : 0);
  node->left_idx = left_index;
  }


/*
  void dense2sparse_node(const dense_node_t* dense_root, int i_dense,
                         int i_sparse_root, int i_sparse) {
    float output, threshold;
    int feature;
    bool def_left, is_leaf;
    dense_node_decode(&dense_root[i_dense], &output, &threshold, &feature,
                      &def_left, &is_leaf);
    if (is_leaf) {
      // leaf sparse node
      sparse_node_init(&sparse_nodes[i_sparse], output, threshold, feature,
                       def_left, is_leaf, 0);
      return;
    }
    // inner sparse node
    // reserve space for children
    int left_index = sparse_nodes.size();
    sparse_nodes.push_back(sparse_node_t());
    sparse_nodes.push_back(sparse_node_t());
    sparse_node_init(&sparse_nodes[i_sparse], output, threshold, feature,
                     def_left, is_leaf, left_index - i_sparse_root);
    dense2sparse_node(dense_root, 2 * i_dense + 1, i_sparse_root, left_index);
    dense2sparse_node(dense_root, 2 * i_dense + 2, i_sparse_root,
                      left_index + 1);
  }

  void dense2sparse_tree(dense_node_t* dense_root) {
    int i_sparse_root = sparse_nodes.size();
    sparse_nodes.push_back(sparse_node_t());
    dense2sparse_node(dense_root, 0, i_sparse_root, i_sparse_root);
    trees.push_back(i_sparse_root);
  }

  void dense2sparse() {
    for (int tree = 0; tree < ps.num_trees; ++tree) {
      dense2sparse_tree(&nodes[tree * tree_num_nodes()]);
    }
  }

  void printf_sparse()
  {
	  for (int i = 0; i < sparse_nodes.size(); i++) {
		  printf("ID: %d, val: %f, bits: %d, left_idx: %d\n", i, sparse_nodes[i].val, sparse_nodes[i].bits, sparse_nodes[i].left_idx);
	  }
	  for (int i = 0; i < ps.num_trees; i++) {
		  printf("Tree: %d, trees: %d\n", i, trees[i]);
	  }
  }

  void init_sparse(const cudaStream_t& h, sparse_forest** pf, const int* trees,
                 const sparse_node_t* nodes, const forest_params_t* params) {
  check_params(params, false);
  sparse_forest* f = new sparse_forest;
  f->init(h, trees, nodes, params);
  *pf = f;
  }

  void init_forest_sparse(sparse_forest** pforest){
    // init FIL model
    forest_params_t fil_params;
    fil_params.num_trees = ps.num_trees;
    fil_params.num_cols = ps.num_cols;
    fil_params.algo = algo_t::NAIVE;
    fil_params.output = ps.output;
    fil_params.threshold = ps.threshold;
    fil_params.global_bias = ps.global_bias;
    dense2sparse();
	//printf_sparse();
    fil_params.num_nodes = sparse_nodes.size();
    init_sparse(stream, pforest, trees.data(), sparse_nodes.data(), &fil_params);
  }

  void predict_sparse(const cudaStream_t& h, sparse_forest* f, float* preds, const float* data,
             size_t num_rows) {
  f->predict(h, preds, data, num_rows);
  } 

  void predict_on_gpu_sparse() {
    sparse_forest* forest = nullptr;
    init_forest_sparse(&forest);

    // predict
    allocate(preds_d, ps.num_rows);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    predict_sparse(stream, forest, preds_d, data_d, ps.num_rows);

	for(int i=0;i<50;i++)
    predict_sparse(stream, forest, preds_d, data_d, ps.num_rows);

	cudaProfilerStart();

	struct timeval start;
	struct timeval end;
	gettimeofday(&start,NULL);
	for(int i=0;i<5000;i++)
    predict_sparse(stream, forest, preds_d, data_d, ps.num_rows);
	gettimeofday(&end,NULL);

	cudaProfilerStop();

	float time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
	printf("time_sparse is %f us\n",time_use/5000);

    CUDA_CHECK(cudaStreamSynchronize(stream));

#if _DEBUG_
	float* output = new float[ps.num_rows];
	updateHost(output, preds_d, ps.num_rows, stream);
	printf("\npredicted sparse output:\n");
	printf_float_CPU(output, ps.num_rows);
	delete output;
#endif

	cudaDeviceSynchronize();
	compare_GPU<<<1,1,0,stream>>>(preds_d, want_preds_d, ps.num_rows);
	cudaDeviceSynchronize();

    delete forest;
  }
*/


/*
  void compare() {
    ASSERT_TRUE(devArrMatch(want_preds_d, preds_d, ps.num_rows,
                            CompareApprox<float>(ps.tolerance), stream));
  }

  float infer_one_tree(fil::dense_node_t* root, float* data) {
    int curr = 0;
    float output = 0.0f, threshold = 0.0f;
    int fid = 0;
    bool def_left = false, is_leaf = false;
    for (;;) {
      fil::dense_node_decode(&root[curr], &output, &threshold, &fid, &def_left,
                             &is_leaf);
      if (is_leaf) break;
      float val = data[fid];
      bool cond = isnan(val) ? !def_left : val >= threshold;
      curr = (curr << 1) + 1 + (cond ? 1 : 0);
    }
    return output;
  }

*/

  void Free()
  {
	  CUDA_CHECK(cudaFree(preds_d));
	  CUDA_CHECK(cudaFree(want_preds_d));
	  CUDA_CHECK(cudaFree(data_d));
	  data_h.clear();
	  nodes.clear();
	  sparse_nodes.clear();
	  trees.clear();
  }

  int tree_num_nodes() { return (1 << (ps.depth + 1)) - 1; }

  int forest_num_nodes() { return tree_num_nodes() * ps.num_trees; }


  // predictions
  float* preds_d = nullptr;
  float* want_preds_d = nullptr;

  // input data
  float* data_d = nullptr;
  std::vector<float> data_h;

  // forest data
  std::vector<dense_node_t> nodes;

  // sprase forest
  std::vector<sparse_node_t> sparse_nodes;
  std::vector<int> trees;

  // parameters
  cudaStream_t stream;
  TahoeTestParams ps;
};

#endif
