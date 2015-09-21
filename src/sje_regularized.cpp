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
const char *att_file_train = 0;
const char *att_file_test = 0;
const char *embfile = 0;

bool normalize = true;
bool isval = false;
double lambda = 1e-63;
double eta = 1e-5;

int epochs = 5;
int maxtrain = -1;
int nclass = 0;

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
  SvmSgdSJE(int nclass, int dim, double lambda, double eta);
  ~SvmSgdSJE();
public:
  void train(int imin, int imax, double *xp, const yvec_t &yp, int dims, 
	double *att_mat, int cls_dim, 
	double *emb_mat, int emb_dim, const char *prefix = "");
  double test(int imin, int imax, double *x, const yvec_t &y, int dims,
	double *emb_mat, int emb_dim, double *att_mat, const char *prefix = "");
	void check_gradient(double *cur_w, double *g, double *xn, double *yn, double *y);
	double gradient_norm(double* g, int dim);
	void updateEta_tdecay();
	double evaluate_objective(double *xp, const yvec_t yp, double *emb_mat, double *att_mat);
private:
  int nclass;
  int dim;
  double  lambda;
  double  eta, eta0;
  int t;
};

void SvmSgdSJE::updateEta_tdecay(){
	eta = eta0 / (1 + eta0*lambda*t);
}

/// Constructor
SvmSgdSJE::SvmSgdSJE(int _nclass, int _dim, double _lambda, double _eta)
{
	nclass = _nclass;
	dim = _dim;

	lambda = _lambda;
	eta0 = _eta;
	eta = eta0;
        t = 0;
}

/// Destructor
SvmSgdSJE::~SvmSgdSJE(){}

/// Training the svms with SJE using ranking objective
void SvmSgdSJE::train(int imin, int imax, double *xp, const yvec_t &yp, int dims,
		 double *att_mat, int cls_dim, 
		 double *emb_mat, int emb_dim,
		 const char *prefix)
{
	cout << prefix << " Training SJE for lbd = " << lambda << ", eta = " << eta << " and " << nclass << " classes" << endl;

	assert(imin <= imax);

	double xproj[emb_dim];
	//int i = 0;
	//double prev_emb_mat[emb_dim*dims];
		//double g[emb_dim*dims];
	for(int i = imin; i <= imax; i++)
	{
		//i = rand()%imax + imin;
		for(int iy = 0; iy < emb_dim; iy++)
			xproj[iy] = 0.0;
		// project training images onto label embedding space
		double xproj_norm = 0;
		for(int iy = 0; iy < emb_dim; iy++)
		{
			for(int ix = 0; ix < dims; ix++)
			{	
				xproj[iy] += xp[dims*i + ix] * emb_mat[dims*iy + ix];
			}
			xproj_norm += xproj[iy] * xproj[iy];
		}
		// normalize the projected vector
		xproj_norm = sqrt(xproj_norm);
		//for(int iy = 0; iy < emb_dim; iy++)
			//xproj[iy] = xproj[iy] / xproj_norm;	
		int best_index = -1;
		double best_score = 0.0;
		for(int j = 0; j < nclass; j++)
		{
		  double score = 0.0;
			for(int iy = 0; iy < emb_dim; iy++)
				score += xproj[iy] * att_mat[emb_dim*j + iy];
			
			//delta(y_n,y)= 1 if y_n != y
			if(j != yp.at(i) - 1)	
				score += 1;
			if(score > best_score)
			{
				best_score = score;
				best_index = j;
			}
		}
		//update the embedding matrix when the decision is wrong
		if(best_index != int(yp.at(i)-1) && best_index != -1)
		{
	//		memcpy(prev_emb_mat, emb_mat, emb_dim*dims*sizeof(double));
			int ni = int(yp.at(i) - 1);
			for(int iy = 0; iy < emb_dim; iy++)
			{
				for(int ix = 0; ix < dims; ix++)
				{
					emb_mat[dims*iy + ix] -= eta * (xp[dims*i + ix] * (att_mat[emb_dim*best_index + iy] - att_mat[emb_dim*ni + iy]) + lambda*emb_mat[dims*iy + ix]); 
			//		g[dims*iy + ix] = xp[dims*i + ix] * (att_mat[emb_dim*best_index + iy] - att_mat[emb_dim*ni + iy]) + lambda*emb_mat[dims*iy + ix]; 
				}
			}
	//		check_gradient(prev_emb_mat, g, &xp[dims*i], &att_mat[emb_dim*ni], &att_mat[emb_dim*best_index]);
		}
		else{
       	for(int iy = 0; iy < emb_dim; iy++)
       	{
        	for(int ix = 0; ix < dims; ix++)
         	{
          	emb_mat[dims*iy + ix] -= eta*lambda*emb_mat[dims*iy + ix];
				//		g[dims*iy + ix] = lambda*emb_mat[dims*iy + ix];
         }
       }
		}
		t += 1;
		updateEta_tdecay();
	}
	//cout << "gradient norm=" << gradient_norm(g,dims*emb_dim) << endl;
}

/// Testing
double SvmSgdSJE::test(int imin, int imax, double *xp, const yvec_t &yp, int dims,  
		double *emb_mat, int emb_dim, double *att_mat, const char* prefix)
{
  	cout << prefix << " Testing Multi-class for [" << imin << ", " << imax << "]." << endl;

  	assert(imin <= imax);
  	int nsamples = imax-imin+1;

  	double* scores = new double[nclass*nsamples];
  	int* conf_mat = new int[nclass*nclass];
  	memset(conf_mat,0,sizeof(int)*nclass*nclass);
	for(int i = 0; i < nclass*nsamples; i++)
		scores[i] = 0.0;
	double xproj[emb_dim];
  	for(int i = 0; i < nsamples; i++)
  	{
		// project test images onto label embedding space
		for(int iy = 0; iy < emb_dim; iy++)
			xproj[iy] = 0.0;
		// project training images onto label embedding space
		double xproj_norm = 0;
		for(int iy = 0; iy < emb_dim; iy++)
		{
			for(int ix = 0; ix < dims; ix++)
			{	
				xproj[iy] += xp[dims*i + ix] * emb_mat[dims*iy + ix];
			}
			xproj_norm += xproj[iy] * xproj[iy];
		}
		// normalize the projected vector
		xproj_norm = sqrt(xproj_norm);
		if(xproj_norm != 0)
		{
			// calculate the scores using dot product similarity using classifiers
//			for(int iy = 0; iy < emb_dim; iy++)
	//			xproj[iy] = xproj[iy] / xproj_norm;	
		}
		for(int j = 0; j < nclass; j++)
  			{
				for(int iy = 0; iy < emb_dim; iy++)
					scores[nsamples*j + i] += xproj[iy] * att_mat[emb_dim*j + iy];
			}
			
  	}

  	double nerr = 0;
  	for(int i = 0; i < nsamples; i++)
  	{
		int true_class = int(yp.at(i)-1);

		double max_score = -1.0f;
		int predicted_class = -1;

		for(int c = 0; c < nclass; c++)
		{
			if ( scores[nsamples*c+i] > max_score )
			{
				predicted_class = c;
				max_score = scores[nsamples*c+i];
			}
		}
		
		conf_mat[nclass*true_class+predicted_class]++;
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
		sum_each_line=0;
		for(int j=0; j < nclass; j++)
		{
			sum_each_line = sum_each_line + conf_mat[i*nclass+j];
		}

		cout << " Class = " << i << " accuracy = " << setprecision(4) << double(conf_mat[i*nclass+i]/sum_each_line) << "%." << endl;
	
		sum_diag_conf = sum_diag_conf + double(double(conf_mat[i*nclass+i])/sum_each_line);
  	}
		
  	cout << " Per " << prefix << " class accuracy = " << setprecision(4) << 100 * double(sum_diag_conf / nclass) << "%." << endl;

 	double acc = double(sum_diag_conf / nclass);
  	delete scores;
  	delete conf_mat;
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
	  else if (att_file_train == 0)
            att_file_train = arg;
	  else if (att_file_test == 0)
            att_file_test = arg;
	  else if (embfile == 0)
            embfile = arg;
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
          else if (opt == "dontnormalize")
            {
              normalize = false;
            }
          else if (opt == "val")
            {
              isval = true;
            }
          else if (opt == "maxtrain" && i+1 < argc)
            {
              maxtrain = atoi(argv[++i]);
              assert(maxtrain > 0);
            }
          else if (opt == "nclass" && i+1 < argc)
            {
              nclass = atoi(argv[++i]);
              assert(nclass > 0);
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
  cout << " -nclass " << nclass;
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
int dims=0;
double* xtrain;
yvec_t ytrain;
double* xtest;
yvec_t ytest;
int nsamples;
int emb_dim;		//dim of output embedding
int cls_dim_train;
int cls_dim_test;

double *emb_mat, *last_emb_mat;
double eps = 1e-6;

double SvmSgdSJE::evaluate_objective(double *xp, const yvec_t yp, double *emb_mat, double *att_mat){
	double risk = 0.0;
	cout << "start to evaluate" << endl;
	for(int i = 0; i < nsamples; i++){
		double max_score = 0.0;
		int ni = yp.at(i) - 1;
		double norm_emb = 0.0;
		double xproj[emb_dim*dims];
		memset(xproj, 0, emb_dim*dims*sizeof(double));
		for(int j = 0; j < emb_dim; j++){
			for(int k = 0; k < dims; k++){
				xproj[j] += xp[dims*i+k] * emb_mat[dims*j+k];
				norm_emb += lambda * emb_mat[dims*j+k] * emb_mat[dims*j+k];
			}
		}
		for(int c = 0; c < nclass; c++){
			double score = 0.0;
			if(c != ni)
				score += 1;
			for(int j = 0; j < emb_dim; j++){
					score += xproj[j] * (att_mat[emb_dim*c+j] - att_mat[emb_dim*ni+j]); 	
			}
			if(max_score < score){
				max_score = score;
			}
		}
		if(max_score > 0){
			risk += max_score;
		}
		else cout << "The risk is below zero" << endl;
		risk += norm_emb;
	}
	return risk;
}

void SvmSgdSJE::check_gradient(double *cur_w, double *g, double *xn, double *yn, double *y){
	double cur_q = 1.0, new_q = 1.0;
	double new_w[emb_dim*dims];
	double delta_g = 0.0;
	//double lambda = 1e-8;
	for(int i = 0; i < emb_dim*dims; i++){
		new_w[i] = cur_w[i] - eps*g[i]; 
		delta_g += -eps * g[i] * g[i];
	}
	for(int iy = 0; iy < emb_dim; iy++){
		for(int ix = 0; ix < dims; ix++){
			cur_q += xn[ix] * (y[iy] - yn[iy]) * cur_w[dims*iy + ix] + lambda*cur_w[dims*iy+ix]*cur_w[dims*iy+ix];
			new_q += xn[ix] * (y[iy] - yn[iy]) * new_w[dims*iy + ix] + lambda*new_w[dims*iy+ix]*new_w[dims*iy+ix];
		}
	}
	double res = cur_q - (new_q+delta_g);
	if(res <= 1e-6)
		printf("Pass the gradient check\n");
	else
		printf("res=%lf, cur_q=%lf, new_q=%lf, Fail the gradient check\n", res, cur_q, new_q);
		
}

double SvmSgdSJE::gradient_norm(double* g, int dim){
	double norm = 0.0;
	for(int i = 0; i < dim; i++){
		norm += g[i]*g[i];
	}
	norm = sqrt(norm);
	return norm;
}


double max_residual(double* last_emb_mat, double* emb_mat){
	double max_d = -1;
	double max = -1;
	for(int i = 0; i < emb_dim*dims; i++){
		double tmp_d = abs(last_emb_mat[i] - emb_mat[i]);
		if(max_d < tmp_d)	
			max_d = tmp_d;
		if(max < emb_mat[i])
			max = emb_mat[i];
	}
	return max_d / max;
}

int main(int argc, const char **argv)
{
	parse(argc, argv);
	config(argv[0]);
	srand(time(NULL));  
	// load the attribute vectors for training as FVector type 
	double* att_mat_train; 
  	if (att_file_train)
     	att_mat_train = load_attribute_matrix(att_file_train, emb_dim, cls_dim_train);

  	SvmSgdSJE svm_ale_mult_2(cls_dim_train, emb_dim, lambda, eta);
	xvec_t xtrain_fvec;
  	// Load training images and their labels
  	if (trainfile)
    	loadmult_datafile(trainfile, xtrain_fvec, ytrain, dims);//, 315);
	//for(int i = 0; i < xtrain_fvec.size(); i++)
		//xtrain_fvec.at(i).scale(1/sqrt(dot(xtrain_fvec.at(i),xtrain_fvec.at(i))));
	xtrain = xvec_to_double(xtrain_fvec.size(), dims, xtrain_fvec);
	double mean[dims];
  double variance[dims];
	double max;
  normalization(xtrain, dims, xtrain_fvec.size(), mean, variance, true);
  normalization2(xtrain, dims, xtrain_fvec.size(), max, true);
	//allocate memory for emb_mat
    cout << "The size of the embedding matrix is " << emb_dim << "*"  << dims << " " << endl;
  	last_emb_mat = new double[emb_dim * dims];	
  	if (access(embfile, F_OK) == 0)
  	{	
		cout << "Read tensor from the file";
		printf("%s\n", embfile);
		emb_mat = load_embedding_matrix(embfile, emb_dim, dims);
  	}
  	else{	
	  	// initialize the label embedding space
	  	int i,j;
		emb_mat = new double[emb_dim * dims];
	  	double std_dev = 1.0 / sqrt(dims);  
	  	for(i = 0; i < emb_dim; i++)
	  	{
			for( j = 0; j < dims; j++)
			{
				emb_mat[dims*i + j] = std_dev * rand_gen();
			} 
			//double dot_prod = dot(*(emb_mat_fvec[i]), *(emb_mat_fvec[i]));
			//emb_mat_fvec[i]->scale(lambda/sqrt(dot_prod));
	  	}	
  	}
	// Training
  	int imin = 0, imax = xtrain_fvec.size() - 1; 
		nsamples = imax + 1;

		//save_embedding_matrix(embfile, emb_dim, dims, emb_mat);
    double *att_mat_test; 	
  	// load the attribute vectors for test as FVector type 
  	if (att_file_test)
     	att_mat_test = load_attribute_matrix(att_file_test, emb_dim, cls_dim_test);

  	SvmSgdSJE svm_ale_mult_2_test(cls_dim_test, emb_dim, lambda, eta); 

		xvec_t xtest_fvec;
  	// Load test images and their labels 
  	if(testfile)
    		loadmult_datafile(testfile, xtest_fvec, ytest, dims);
	
//	for(int i = 0; i < xtest_fvec.size(); i++)
	//	xtest_fvec.at(i).scale(1/sqrt(dot(xtest_fvec.at(i),xtest_fvec.at(i))));
		xtest = xvec_to_double(xtest_fvec.size(), dims, xtest_fvec);

	normalization(xtest, dims, xtest_fvec.size(), mean, variance, false);  	
	normalization2(xtest, dims, xtest_fvec.size(), max, false);  	
	int tmin = 0, tmax = xtest_fvec.size() - 1;
  int best_nepoch = 0;
	double best_accuracy = 0.0;
	double cur_accuracy = 0.0;
	double *best_emb_mat = new double[emb_dim*dims];
  for(int i = 0; i < epochs; i++)
  {
		cout << "--------- Epoch " << i+1 << "." << endl;
  	svm_ale_mult_2.train(imin, imax, xtrain, ytrain, dims, att_mat_train, cls_dim_train, emb_mat, emb_dim);
		if(isval){
    	cur_accuracy = svm_ale_mult_2_test.test(tmin, tmax, xtest, ytest, dims, emb_mat, emb_dim, att_mat_test,"test");
			if(cur_accuracy > best_accuracy){
      	best_accuracy = cur_accuracy;
       	best_nepoch = i + 1;
				memcpy(best_emb_mat, emb_mat, sizeof(double)*emb_dim*dims);
     	}
		}
  }
	if(isval){
		save_embedding_matrix(embfile, emb_dim, dims, best_emb_mat);
		cout << best_accuracy << " " << eta << " " << lambda << " " << best_nepoch << endl;
	}
	else{
		save_embedding_matrix(embfile, emb_dim, dims, emb_mat);
		svm_ale_mult_2_test.test(tmin, tmax, xtest, ytest, dims, emb_mat, emb_dim, att_mat_test,"test");
	}
  
	return 0;
}
