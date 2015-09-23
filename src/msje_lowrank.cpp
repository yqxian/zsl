// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2007- Leon Bottou

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <string.h>

#include "assert.h"
#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include "loss.h"
#include "data_mult.h"

using namespace std;

// ---- Loss function
#ifndef LOSS
# define LOSS HingeLoss
#endif

// ---- Bias term
#ifndef BIAS
# define BIAS 1
#endif
#ifndef REGULARIZED_BIAS
# define REGULARIZED_BIAS 0
#endif


// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
const char *att1_file_train = 0;
const char *att1_file_test = 0;
const char *att2_file_train = 0;
const char *att2_file_test = 0;
const char *embfile = 0;

bool normalize = true;
bool isval = false;
double lambda = 1e-63;
double eta = 1e-5;

int epochs = 5;
int maxtrain = -1;
int rank = 0;

void usage(const char *progname)
{
  const char *s = ::strchr(progname,'/');
  progname = (s) ? s + 1 : progname;
  cerr << "Usage: " << progname << " [options] trainfile [testfile] embfile" << endl
       << "Options:" << endl;
#define NAM(n) "    " << setw(16) << left << n << setw(0) << ": "
#define DEF(v) " (default: " << v << ".)"
  cerr << NAM("-lambda x")
       << "Regularization parameter" << DEF(lambda) << endl
       << NAM("-eta")
       << "Size of the SGD steps" << DEF(eta) << endl
       << NAM("-epochs n")
       << "Number of training epochs" << DEF(epochs) << endl
       << NAM("-dontnormalize")
       << "Do not normalize the L2 norm of patterns." << endl
       << NAM("-maxtrain n")
       << "Restrict training set to n examples." << endl;
#undef NAM
#undef DEF
  ::exit(10);
}

// --multiclass svmsgd
class SvmSgdSJE
{
public:
  SvmSgdSJE(int dims, int emb_dim1, int emb_dim2, int rank, double lambda, double eta);
  ~SvmSgdSJE();
public:
  void train(int imin, int imax, int itemax, double* xp, const yvec_t &yp, double *att_mat1, double *att_mat2, int nclass, const char *prefix = "");
  double test(int imin, int imax, double* xp, const yvec_t &y,double *att_mat1, double *att_mat2, int nclass, const char *prefix = "");
	void init_tensor();
	void getA(double *x, double *A);
	void getBC(double *att1, double *att2, int nclass, double *B, double *C);
	void getB(double *att1, int nclass, double *B);
	void getC(double *att2, int nclass, double *C);
	int getOneLoss(double *x, double *A, double *B, double *C, int nclass, int label);
	int getOnePredict(double *x, double *A, double *B, double *C, int nclass);
	void copy_emb_tensor(double *tmp_x, double *tmp_y1, double *tmp_y2);
	void save_model(const char *fname);
	void load_model(const char *fname);
private:
	double *emb_tensor_x;
	double *emb_tensor_y1;
	double *emb_tensor_y2;
  int emb_dim1;	//dim of output embedding
  int emb_dim2;	//dim of output embedding
	int dims;
  double  lambda;
	int rank;
  double  eta;
  int t;
};

void SvmSgdSJE::save_model(const char *fname){

	save_emb_tensor_components(fname, rank, emb_dim1, emb_dim2, dims, emb_tensor_x, emb_tensor_y1, emb_tensor_y2);

}

void SvmSgdSJE::load_model(const char *fname){

	load_emb_tensor_components(fname, rank, emb_dim1, emb_dim2, dims, emb_tensor_x, emb_tensor_y1, emb_tensor_y2);

}

void SvmSgdSJE::copy_emb_tensor(double *tmp_x, double *tmp_y1, double *tmp_y2){
	memcpy(tmp_x, emb_tensor_x, sizeof(double)*rank*dims);
	memcpy(tmp_y1, emb_tensor_y1, sizeof(double)*rank*emb_dim1);
	memcpy(tmp_y2, emb_tensor_y2, sizeof(double)*rank*emb_dim2);

}
/// Constructor
SvmSgdSJE::SvmSgdSJE(int _dims, int _emb_dim1,int _emb_dim2, int _rank, double _lambda, double _eta)
{
	dims = _dims;
	emb_dim1 = _emb_dim1;
	emb_dim2 = _emb_dim2;

	lambda = _lambda;
	eta = _eta;

  t = 0;
	rank = _rank;

	emb_tensor_x = new double[rank*dims];
	emb_tensor_y1 = new double[rank*emb_dim1];
	emb_tensor_y2 = new double[rank*emb_dim2];
}

/// Destructor
SvmSgdSJE::~SvmSgdSJE(){
	delete[] emb_tensor_x;
	delete[] emb_tensor_y1;
	delete[] emb_tensor_y2;
}

void SvmSgdSJE::init_tensor(){
	cout << "start to initilize the tensor" << endl;
	double std_dev1 = 1.0 / sqrt(dims);  
	double std_dev2 = 1.0 / sqrt(emb_dim1);  
	double std_dev3 = 1.0 / sqrt(emb_dim2);  
	for(int l = 0; l < rank; l++){
		for(int ix = 0; ix < dims; ix++)
			emb_tensor_x[l*dims + ix] = std_dev1 * rand_gen();
					
		for(int iy1 = 0; iy1 < emb_dim1; iy1++)
			emb_tensor_y1[l*emb_dim1 + iy1] = std_dev2 * rand_gen();

		for(int iy2 = 0; iy2 < emb_dim2; iy2++)
			emb_tensor_y2[l*emb_dim2 + iy2] = std_dev3 * rand_gen();
	}
}

void SvmSgdSJE::getA(double *x, double *A){

	for(int l = 0; l < rank; l++){
		A[l] = 0.0;
	}
	
	for(int l = 0; l < rank; l++){
		for(int ix = 0; ix < dims; ix++)
			A[l] += emb_tensor_x[l*dims + ix] * x[ix];
	}
}


void SvmSgdSJE::getB(double *att1, int nclass, double *B){
	
	for(int j = 0; j < nclass; j++){
		for(int l = 0; l < rank; l++){
			B[j*rank + l] = 0.0;
		}
	}

	for(int j = 0; j < nclass; j++){
		for(int l = 0; l < rank; l++){
			for(int iy1 = 0; iy1 < emb_dim1; iy1++)
				B[j*rank + l] += emb_tensor_y1[l*emb_dim1 + iy1] * att1[j*emb_dim1 + iy1];
		}
	}
}
void SvmSgdSJE::getC(double *att2, int nclass, double *C){
	
	for(int j = 0; j < nclass; j++){
		for(int l = 0; l < rank; l++){
			C[j*rank + l] = 0.0;
		}
	}

	for(int j = 0; j < nclass; j++){
		for(int l = 0; l < rank; l++){
			for(int iy2 = 0; iy2 < emb_dim2; iy2++)
				C[j*rank + l] += emb_tensor_y2[l*emb_dim2 + iy2] * att2[j*emb_dim2 + iy2];
		}
	}
}
void SvmSgdSJE::getBC(double *att1, double *att2, int nclass, double *B, double *C){
	
	for(int j = 0; j < nclass; j++){
		for(int l = 0; l < rank; l++){
			B[j*rank + l] = 0.0;
			C[j*rank + l] = 0.0;
		}
	}

	for(int j = 0; j < nclass; j++){
		for(int l = 0; l < rank; l++){
			for(int iy1 = 0; iy1 < emb_dim1; iy1++)
				B[j*rank + l] += emb_tensor_y1[l*emb_dim1 + iy1] * att1[j*emb_dim1 + iy1];
			for(int iy2 = 0; iy2 < emb_dim2; iy2++)
				C[j*rank + l] += emb_tensor_y2[l*emb_dim2 + iy2] * att2[j*emb_dim2 + iy2];
		}
	}
}

int SvmSgdSJE::getOnePredict(double *x, double *A, double *B, double *C, int nclass){
	int best_idx = -1;
	double best_score = -1.0f;
	for(int j = 0; j < nclass; j++){
		double score = 0.0;	
		for(int l = 0; l < rank; l++){					
			score += A[l] * B[j*rank+l] * C[j*rank+l];  				
		}
		if(score > best_score){
			best_score = score;
			best_idx = j;			
		}
	}
	if(best_idx == -1)
		cout << "Error: best score is below -1" << endl;
	return best_idx;
}

int SvmSgdSJE::getOneLoss(double *x, double *A, double *B, double *C, int nclass, int label){
	int best_idx = -1;
	double best_score = -DBL_MAX/1000;
	for(int j = 0; j < nclass; j++){
		double score = 0.0;	
		for(int l = 0; l < rank; l++){					
			score += A[l] * B[j*rank+l] * C[j*rank+l];  				
		}
		//Following the paper, delta(y_n,y) = 1 if y_n != y
		if(j != label){
			score += 1;
		}
		if(score > best_score){
			best_score = score;
			best_idx = j;			
		}
	}
	if(best_idx == -1)
		cout << "Warning: max score is below -DBL_MAX/1000" << endl;
	return best_idx;
}

void SvmSgdSJE::train(int imin, int imax, int itemax, double* xp, const yvec_t &yp, double *att_mat1,	double *att_mat2, int nclass, const char *prefix){

	cout << prefix << " Training SJE for lbd = " << lambda << ", eta = " << eta << " and " << nclass << " classes" << endl;

	assert(imin <= imax);
	Timer time;
	//allocate memory for matrix A,B
	double *A = new double[rank];
	double *B = new double[rank*emb_dim1];
	double *C = new double[rank*emb_dim2];
	//loop over samples
	//int i = 0; //the index of the randomly picked sample
	getBC(att_mat1, att_mat2, nclass, B, C);
	for(int i = imin; i <= imax; i++){
		time.start();
		//cout << "Iteration " << ite;
		//i = rand() % imax + imin;
		//cout << "Picked sample " << i << ", label is " << yp.at(i) << endl;
		getA(&xp[i*dims], A);
		int ni = int(yp.at(i)-1);
		int best_idx = getOneLoss(&xp[i*dims], A, B, C, nclass, ni);
	//	cout << "Compute argmax over y..." << endl;
	//	cout << "Finish computing argmax over y!" << endl;
		//if the decision is wrong
		if(best_idx != -1 && best_idx != yp.at(i) - 1){
			cout << "Upate emb_tensor_x component.." << endl;
			//update the emb_tensor by gradient descent
			//fix emb_tensor_y1, emb_tensor_y2, update emb_tensor_x
			for(int l = 0; l < rank; l++){
				double da_saver = B[best_idx*rank+l]*C[best_idx*rank+l] - B[ni*rank+l]*C[ni*rank+l];
				for(int ix = 0; ix < dims; ix++)
					emb_tensor_x[l*dims+ix] -= eta * xp[i*dims+ix] * da_saver; 
			}
		}
		//update A
		getA(&xp[i*dims], A);
		best_idx = getOneLoss(&xp[i*dims], A, B, C, nclass, ni);
		if(best_idx != -1 && best_idx != ni){
			//fix emb_tensor_x, emb_tensor_y2, update emb_tensor_y1
			cout << "Upate emb_tensor_y1 component.." << endl;
			for(int l = 0; l < rank; l++){
				double db_saver_best = A[l] * C[best_idx*rank+l];
				double db_saver_ni = A[l] * C[ni*rank+l];
				for(int iy1 = 0; iy1 < emb_dim1; iy1++)
					emb_tensor_y1[l*emb_dim1+iy1] -= eta * (att_mat1[best_idx*emb_dim1+iy1]*db_saver_best - att_mat1[ni*emb_dim1+iy1]*db_saver_ni);
			}
		}
		//update B
		getB(att_mat1, nclass, B);
		best_idx = getOneLoss(&xp[i*dims], A, B, C, nclass, ni);
		if(best_idx != -1 && best_idx != ni){
			//fix emb_tensor_x, emb_tensor_y1, update emb_tensor_y2
			cout << "Upate emb_tensor_y2 component.." << endl;
			for(int l = 0; l < rank; l++){
				double dc_saver_best = A[l] * B[best_idx*rank+l];
				double dc_saver_ni = A[l] * B[ni*rank+l];	
				for(int iy2 = 0; iy2 < emb_dim2; iy2++)
					emb_tensor_y2[l*emb_dim2+iy2] -= eta * (att_mat2[best_idx*emb_dim2+iy2]*dc_saver_best - att_mat2[ni*emb_dim2+iy2]*dc_saver_ni); 
			}
		}
		//update C
		getC(att_mat2, nclass, C);
		time.stop();
		cout << "Total time: "  << time.elapsed() << " secs."  << endl;
		time.reset();
	}

	delete[] A;
	A = NULL;
	delete[] B;
	B = NULL;
	delete[] C;
	C = NULL;
}

double SvmSgdSJE::test(int imin, int imax,double* xp, const yvec_t &yp, double *att_mat1, double *att_mat2, int nclass, const char* prefix)
{
	cout << prefix << " Testing Multi-class for [" << imin << ", " << imax << "]." << endl;
 	assert(imin <= imax);
	
	int nsamples = imax-imin+1;

	double* scores = new double[nclass*nsamples];
	int* conf_mat = new int[nclass*nclass];
	memset(conf_mat,0,sizeof(int)*nclass*nclass);

	for(int i = 0; i < nclass*nsamples; i++) scores[i] = 0.0;
	//allocate memory for matrix A
	double *A = new double[rank];
	double *B = new double[rank*emb_dim1];
	double *C = new double[rank*emb_dim2];
 	//computing the score for each sample for each class
	getBC(att_mat1, att_mat2, nclass, B, C);
	
	for(int i = imin; i <= imax; i++)
	{
		//Timer timer;
		//timer.start();
		//cout << "Sample " << i << " with label "<< yp.at(i) <<endl;
		//initilize matrix A
		getA(&xp[i*dims], A);
		double best_score = 0.0;
		//double best_idx = -1;
		for(int j = 0; j < nclass; j++){
			//the following four loops are for computing the multi-linear form score
			//loop over output embedding y1
			double tmp_score = 0.0;
			for(int l = 0; l < rank; l++){					
					tmp_score += A[l] * B[j*rank+l] * C[j*rank+l];  				
			}
			if(tmp_score > best_score){
				best_score = tmp_score;
			//	best_idx = j;
			}
			scores[nsamples*j + i] = tmp_score;
		}
		//cout << ", predicted label is " << best_idx+1;
		//if(best_idx != yp.at(i) - 1)
		//	cout << " Wrong!!!!!!!!!!!!" << endl;
		//else
		//	cout << " Correct!"  << endl;
		//timer.stop();
		//cout << "Total time: " << timer.elapsed() << " secs." << endl;
	}
	//select the best score for each sample and compute the accuracy
 	double nerr = 0;
	for(int i = 0; i < nsamples; i++)
	{
		int true_class = int(yp.at(i)-1);

		double max_score = -1e12;
		int predicted_class = -1;

		for(int c = 0; c < nclass; c++)
		{
			if ( scores[nsamples*c+i] > max_score )
			{
				predicted_class = c;
				max_score = scores[nsamples*c+i];
			}
		}
		if(max_score < -1)
			cout << "Warning: best score= " << max_score << "is below -1" << endl;
		
		if(predicted_class != -1)	
			conf_mat[nclass*true_class+predicted_class]++;
		else
			cout << "Error: best_idx=-1" << endl;
		//cout << true_class << " " << predicted_class << endl;

		if( true_class != predicted_class )
			nerr++;		
	}

  	nerr = nerr / nsamples;
  	cout << " Per image accuracy = " << setprecision(4) << 100-(100 * nerr) << "%." << endl;

	double sum_diag_conf=0;
	double sum_each_line;
	for(int i = 0; i < nclass; i++)
	{
		sum_each_line = 0;
		for(int j = 0; j < nclass; j++)
		{
			sum_each_line = sum_each_line + conf_mat[i*nclass+j];
		}
		if(sum_each_line == 0)
			cout << "Error: class " << i << " has 0 sample" << endl;
		cout << " Class = " << i << " accuracy = " << setprecision(4) << double(conf_mat[i*nclass+i]/sum_each_line) << "%." << endl;

		sum_diag_conf = sum_diag_conf + double(double(conf_mat[i*nclass+i])/sum_each_line);
	}
	cout << " Per " << prefix << " class accuracy = " << setprecision(4) << 100 * double(sum_diag_conf / nclass) << "%." << endl;
	
	double acc = double(sum_diag_conf / nclass);

	delete[] A;
	A = NULL;
	delete[] B;
	B = NULL;
	delete[] C;
	C = NULL;
	delete[] scores;
	scores = NULL;
	delete[] conf_mat;
	conf_mat = NULL;

	return acc;
}


void parse(int argc, const char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
      {
        if (trainfile == 0)
          	trainfile = arg;
        else if (testfile == 0)
            testfile = arg;
		  	else if (att1_file_train == 0)
		    	att1_file_train = arg;
		  	else if (att1_file_test == 0)
		    	att1_file_test = arg;
		  	else if (att2_file_train == 0)
		    	att2_file_train = arg;
		  	else if (att2_file_test == 0)
		   	 	att2_file_test = arg;
		  	else if (embfile == 0){
		    	embfile = arg;
			//printf("%s\n", embfile);
		  }
	      else
	        usage(argv[0]);
      }
      else
      {
		while (arg[0] == '-') 
			arg += 1;
		string opt = arg;
		if (opt == "lambda" && i+1<argc)
		{
			lambda = atof(argv[++i]);
			//assert(lambda>0 && lambda<1e4);
		}
		else if (opt == "eta" && i+1<argc)
		{
		  	eta = atof(argv[++i]);
		  	assert(eta>0);
		}
	  	else if (opt == "epochs" && i+1<argc)
		{
			epochs = atoi(argv[++i]);
			assert(epochs>0 && epochs<1e6);
		}
	    else if (opt == "val")
		{
		    isval = true;
		}
	    else if (opt == "dontnormalize")
		{
		    normalize = false;
		}
	    else if (opt == "maxtrain" && i+1 < argc)
		{
		    maxtrain = atoi(argv[++i]);
		    assert(maxtrain > 0);
		}
	   	else if (opt == "rank" && i+1 < argc)
		{
		    rank = atoi(argv[++i]);
		    assert(rank > 0);
		}
	    else
		{
		    cerr << "Option " << argv[i] << " not recognized." << endl;
		    usage(argv[0]);
		}

        }
    }
  if (! trainfile)
    usage(argv[0]);
}

void config(const char *progname)
{
	  cout << "# Running: " << progname;
	  cout << " -lambda " << lambda;
	  cout << " -eta " << eta;
	  cout << " -epochs " << epochs;
	  if (! normalize) cout << " -dontnormalize";
	  if (maxtrain > 0) cout << " -maxtrain " << maxtrain;
	  cout << endl;
	#define NAME(x) #x
	#define NAME2(x) NAME(x)
	  cout << "# Compiled with: "
		   << " -DLOSS=" << NAME2(LOSS)
		   << " -DBIAS=" << BIAS
		   << " -DREGULARIZED_BIAS=" << REGULARIZED_BIAS
		   << endl;
}


// --- main function


double* xtrain;
yvec_t ytrain;
double* xtest;
yvec_t ytest;

double* att1_mat_train;
double* att2_mat_train;
double* att1_mat_test;
double* att2_mat_test;

int dims;
int emb_dim1, emb_dim2;
int nclass_train, nclass_test;

int main(int argc, const char **argv)
{
	parse(argc, argv);
	config(argv[0]);
	
	srand(time(NULL));
	// load the attribute vectors for training as FVector type 
	if (att1_file_train)
	att1_mat_train = load_attribute_matrix(att1_file_train, emb_dim1, nclass_train);
	if (att2_file_train)
	 att2_mat_train = load_attribute_matrix(att2_file_train, emb_dim2, nclass_train);

	cout << "# of train classes=" << nclass_train << endl;	
	xvec_t xtrain_fvec;
	// Load training images and their labels
	if (trainfile)
		loadmult_datafile(trainfile, xtrain_fvec, ytrain, dims);//, 315);
	
	xtrain = xvec_to_double(xtrain_fvec.size(), dims, xtrain_fvec);
	
	SvmSgdSJE sje(dims, emb_dim1,emb_dim2, rank, lambda, eta);
	//allocate memory for emb_tensor
	double *mean = new double[dims];
	double *variance = new double[dims];
	double max = -1;
	normalization(xtrain, dims, xtrain_fvec.size(), mean, variance, true);
  normalization2(xtrain, dims, xtrain_fvec.size(), max, true);

	cout << "rank=" << rank << endl;
	cout << "The size of the tensor is" << emb_dim1 << "*" << emb_dim2 << "*" << dims << " " << endl;

	sje.init_tensor();
	// load the attribute vectors for test as FVector type 
	if (att1_file_test)
		att1_mat_test = load_attribute_matrix(att1_file_test, emb_dim1, nclass_test);
	if (att2_file_test)
		att2_mat_test = load_attribute_matrix(att2_file_test, emb_dim2, nclass_test);
	xvec_t xtest_fvec;
	// Load test images and their labels 
	if (testfile)
		loadmult_datafile(testfile, xtest_fvec, ytest, dims);
	
	xtest = xvec_to_double(xtest_fvec.size(),dims,xtest_fvec);
	//normalization(xtest, dims, xtest_fvec.size());
	normalization(xtest, dims, xtest_fvec.size(), mean, variance, false);
  normalization2(xtest, dims, xtest_fvec.size(), max, false);
	int tmin = 0, tmax = xtest_fvec.size() - 1;
	int imin = 0, imax = xtrain_fvec.size() - 1; 
	int ite_per_epoch = xtrain_fvec.size();
	
  int best_nepoch = 0;
  double best_accuracy = 0.0;
  double cur_accuracy = 0.0;
	double *best_emb_tensor_x = new double[rank*dims];
	double *best_emb_tensor_y1 = new double[rank*emb_dim1];
	double *best_emb_tensor_y2 = new double[rank*emb_dim2];
	for(int i = 1; i <= epochs; i++){
		// Training
		cout << "Epoch " << i << "..." << endl;
		sje.train(imin, imax, ite_per_epoch, xtrain, ytrain, att1_mat_train, att2_mat_train, nclass_train);
		if(isval){
			cur_accuracy = sje.test(tmin, tmax, xtest, ytest, att1_mat_test, att2_mat_test, nclass_test, "test");
			if(cur_accuracy > best_accuracy){
        best_accuracy = cur_accuracy;
        best_nepoch = i;
				sje.copy_emb_tensor(best_emb_tensor_x, best_emb_tensor_y1, best_emb_tensor_y2);				
      }
		}
	}
	if(isval){
		save_emb_tensor_components(embfile,	rank, emb_dim1, emb_dim2, dims, best_emb_tensor_x, best_emb_tensor_y1, best_emb_tensor_y2);
    cout << best_accuracy << " " << eta << " " << best_nepoch << " " << rank << endl;
	}
	else{
		sje.save_model(embfile);
		sje.test(tmin, tmax, xtest, ytest, att1_mat_test, att2_mat_test, nclass_test, "test");
	}	
	return 0;
}
