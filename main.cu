#include "iostream"
#include "BaseTahoeTest.h"
using namespace std;

int main(int argc,char *argv[])
{
	// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
	// global_bias, algo, seed, tolerance

	/*
	int N_batch = 1;
	int S_sample = 18;
	int D_tree = 8;
	int N_tree = 2000;
	int S_att = 2;
	int S_node = 24;
	int BW_W_SMEM = 2000;
	int BW_R_SMEM = 5000;
	int BW_R_COA_GMEM = 512;
	int BW_R_NCOA_GMEM = 128;

	float algorithm0 = S_sample/BW_W_SMEM + D_tree*N_tree*S_att/BW_R_SMEM + S_sample/BW_R_COA_GMEM + D_tree*N_tree*S_node*2.0/BW_R_COA_GMEM;

	float algorithm1 = D_tree*N_tree*S_node/BW_R_SMEM + D_tree*N_tree*S_node/BW_R_COA_GMEM;

	float algorithm2 = D_tree*N_tree*S_node*2.0/BW_R_COA_GMEM + D_tree*N_tree*S_att/BW_R_NCOA_GMEM;

	float algorithm3 = D_tree*N_tree*S_node/BW_W_SMEM/N_batch + D_tree*N_tree*S_node/BW_R_SMEM + D_tree*N_tree*S_node/BW_R_COA_GMEM/N_batch + D_tree*N_tree*S_att/BW_R_NCOA_GMEM;

	int algorithm = 0;

	float time = FLT_MAX;
	if( algorithm0 < time ){
		algorithm = 1;
		time = algorithm0;}
	if( algorithm1 < time ){
		algorithm = 2;
		time = algorithm1;}
	if( algorithm2 < time ){
		algorithm = 3;
		time = algorithm2;}
	if( algorithm3 < time ){
		algorithm = 4;
		time = algorithm3;}

	cout << "Performance model choose #" << algorithm << " strategy." << endl;

	*/

	if(argc!=3)
	{
	    printf("Please use proper inputs: ./Tahoe [Model_Path] [Data_Path];");
	}

	printf("Model: %s , Data: %s\n", argv[1], argv[2]);
	//BaseTahoeTest* pTest = new BaseTahoeTest(argv[1], argv[2], 10000, 500, (float)0.0, 8, 8, (float)0.0, output_t::RAW, (float)0.0, (float)0.0, algo_t::NAIVE, 0, (float)1e-3f, strategy_t::SHARED_DATA);
	BaseTahoeTest* pTest = new BaseTahoeTest(argv[1], argv[2]);
	float speedup = 0.0;
	int best_by_run = pTest->SetUp(speedup);
	/*
	if(algorithm==best_by_run)
		cout<<"Performance model predicts correctly"<<endl;
	else
		cout<<"Performance model predicts incorrectly"<<endl;
	*/

	cout<<"Tahoe brings "<<speedup<<"x speedup."<<endl;

	pTest->Free();

	return 0;
}
