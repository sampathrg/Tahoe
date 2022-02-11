#include "iostream"
#include <cmath>
#include "BaseTahoeTest.h"
#include "bandwidthTest.h"
using namespace std;

int main(int argc,char *argv[])
{

	if(argc!=3)
	{
	    printf("Please use proper inputs: ./Tahoe [Model_Path] [Data_Path];");
	}

	printf("Model: %s , Data: %s\n", argv[1], argv[2]);
	//BaseTahoeTest* pTest = new BaseTahoeTest(argv[1], argv[2], 10000, 500, (float)0.0, 8, 8, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	BaseTahoeTest* pTest = new BaseTahoeTest(argv[1], argv[2]);

	// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
	// global_bias, algo, seed, tolerance

	int N_batch = 1;                                 // Sample: the number of samples in a batch
	int S_sample = 18;                               // Sample: the size of a sample
	int D_tree = pTest->ps.depth;                    // Forest: the depth of a tree
	int N_nodes = (int)pow(2, D_tree + 1) - 1;       // Forest: the number of nodes in a tree
	int N_tree = pTest->ps.num_trees;                // Forest: the number of trees in a forest
	int S_att = 2;                                   // Forest: the size of an attribute
	int S_node = 24;                                 // Forest: the size of a decision node
	int BW_R_COA_GMEM = runTest(NULL, NULL);         // Hardware: read bandwidth of global memory for coalesced data
	int BW_W_SMEM = 4*BW_R_COA_GMEM;                 // Hardware: write bandwidth of shared memory
	int BW_R_SMEM = 10*BW_R_COA_GMEM;                // Hardware: read bandwidth of shared memory
	int BW_R_NCOA_GMEM = (int)(BW_R_COA_GMEM/4);     // Hardware: read bandwidth of global memory for uncoalesced data

	// ALGORITHM_0: Shared data (using adaptive forest format)
	float T_S_MEM =                                               // execution time to access shared memory
	                 S_sample / BW_W_SMEM +                             //   - the time of writing sample into shared memory after loading it from global memory
	                 N_tree * D_tree * S_att / BW_R_SMEM;               //   - the time of reading attributes of the sample from shared memory to meet the needs of traversing
	float T_G_MEM =                                               // execution time to access global memory
	                 S_sample / BW_R_COA_GMEM +                         //   - the time of loading the sample from global memory
					 N_tree * D_tree * S_node * 2.0 / BW_R_COA_GMEM;    //   - the time of traversing trees in global memory with improved memory coalescence 
					                                                    //     using half of bandwidth
	float algorithm0 = T_S_MEM + T_G_MEM;

	// Algorithm_1: Shared forest. This algorithm is reduction free.
	T_S_MEM = N_tree * D_tree * S_node / BW_R_SMEM;           // is the time of reading the forest in shared memory for inference;
	T_G_MEM = N_tree * D_tree * S_att / BW_R_NCOA_GMEM;       // is the time of reading attributes in the sample in global memory using uncoalesced memory accesses.
	float algorithm1 = T_S_MEM + T_G_MEM;
	
	// Algorithm_2: Direct method. This algorithm is reduction free.
	T_S_MEM = 0.0;                                                     // this strategy does not access shared memory
	T_G_MEM =                                                          // execution time to access global memory
				N_tree * N_nodes * S_node * 2.0 / BW_R_COA_GMEM +      //   - the time of loading the forest from global memory with improved memory coalescence
				N_tree * D_tree * S_att / BW_R_NCOA_GMEM;              //   - the time of reading attributes of the sample from global memory using uncoalesced accesses
	float algorithm2 = T_S_MEM + T_G_MEM;

	// Algorithm_3: Splitting shared forest
	T_S_MEM = 
				N_tree * N_nodes * S_node / BW_W_SMEM / N_batch +  // the time of writing the forest to shared memory after reading it from global memory
				N_tree * D_tree * S_node / BW_R_SMEM;              // the time of reading the forest in shared memory for inference
	T_G_MEM = 
				N_tree * N_nodes * S_node / BW_R_COA_GMEM / N_batch +    // the time of reading global memory to load the forest using coalesced memory accesses
				N_tree * D_tree * S_att / BW_R_NCOA_GMEM;                // the time of reading attributes of the sample from global memory using uncoalesced memory accesses
	float algorithm3 = T_S_MEM + T_G_MEM;


	int algorithm = 0;

	float time = FLT_MAX;
	if( algorithm0 < time ){
		algorithm = 2;
		time = algorithm0;}
	if( algorithm1 < time ){
		algorithm = 3;
		time = algorithm1;}
	if( algorithm2 < time ){
		algorithm = 4;
		time = algorithm2;}
	if( algorithm3 < time ){
		algorithm = 5;
		time = algorithm3;}

	cout << "Performance model choose #" << algorithm << " strategy." << endl;


	float speedup = 0.0;
	int best_by_run = pTest->SetUp(speedup);
	if(algorithm==best_by_run)
		cout<<"Performance model predicts correctly"<<endl;
	else
		cout<<"Performance model predicts incorrectly"<<endl;

	cout<<"Tahoe brings "<<speedup<<"x speedup."<<endl;

	pTest->Free();

	return 0;
}
