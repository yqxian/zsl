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
const char *att1_file_train = 0;
const char *att1_file_test = 0;
const char *att2_file_train = 0;
const char *att2_file_test = 0;
const char *embfile = 0;

bool normalize = true;
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
  SvmSgdSJE(int nclass, int dim1, int dim2, double lambda, double eta);
  ~SvmSgdSJE();
public:
  void train(int imin, int imax, int ite_per_epoch, double *xp, const yvec_t &yp, int dims, double *att_mat1, double *att_mat2,  int cls_dim, double* emb_mat1, double *emb_mat2, int emb_dim1, int emb_dim2, const char *prefix = "");
  double test(const int imin, const int imax, double *x, const yvec_t &y, const int dims, double *emb_mat1, double *emb_mat2, const int emb_dim1, const int emb_dim2, const double *att_mat1, const double *att_mat2, const char *prefix = "");
	double evaluate_objective(double *xp, const yvec_t yp, double *emb_mat, double *att_mat);
	void updateEta_tdecay();
	double* getXproject(double* emb_mat, double *x, int dim, int emb_dim);
private:
  int nclass;
  int dim1;
	int dim2;
  double  lambda;
  double  eta,eta0;
  int t;
};
void SvmSgdSJE::updateEta_tdecay(){
   eta = eta0 / (1 + eta0*lambda*t);
}

/// Constructor
SvmSgdSJE::SvmSgdSJE(int _nclass, int _dim1,int _dim2,  double _lambda, double _eta)
{
	nclass = _nclass;
	dim1 = _dim1;
	dim2 = _dim2;

	lambda = _lambda;
	eta0 = _eta;
	eta = eta0;
        t = 0;
}

/// Destructor
SvmSgdSJE::~SvmSgdSJE(){}

double* SvmSgdSJE::getXproject(double* emb_mat, double *x, int dim, int emb_dim){
	double *xproj = new double[emb_dim];
	double xproj_norm = 0.0;
	for(int iy = 0; iy < emb_dim; iy++)
		xproj[iy] = 0.0;
	for(int iy = 0; iy < emb_dim; iy++)
  {
    for(int ix = 0; ix < dim; ix++)
    {
      xproj[iy] += x[ix] * emb_mat[dim*iy + ix];
    }
    xproj_norm += xproj[iy] * xproj[iy];
  }
  // normalize the projected vector
  xproj_norm = sqrt(xproj_norm);
  if(xproj_norm != 0){
    for(int iy = 0; iy < emb_dim; iy++)
      xproj[iy] = xproj[iy] / xproj_norm;
  }
	return xproj;
}
/// Training the svms with SJE using ranking objective
void SvmSgdSJE::train(int imin, int imax, int ite_max, double *xp, const yvec_t &yp, int dims, 
		double *att_mat1, double *att_mat2, int cls_dim,  double* emb_mat1, double *emb_mat2, int emb_dim1, 
		 int emb_dim2, const char *prefix)
{
	cout << prefix << " Training SJE for lbd = " << lambda << ", eta = " << eta << " and " << nclass << " classes" << endl;

	assert(imin <= imax);

	double *xproj1;
	double *xproj2;
	double *d1 = new double[dims*emb_dim1];
	double *d2 = new double[dims*emb_dim2];
	//int i = 0;
	for(int i = imin; i <= imax; i++)
	{
		//i = rand()%imax + imin;
		//cout << "iteration " << ite << ", select sample" << i << endl;
		// project training images onto label embedding space
		xproj1 = getXproject(emb_mat1, &xp[dims*i], dims, emb_dim1);	
		xproj2 = getXproject(emb_mat2, &xp[dims*i], dims, emb_dim2);	

		int best_index = -1;
		double best_score = 0.0;
		for(int j = 0; j < nclass; j++)
		{
		  double score = 0.0;
			for(int iy = 0; iy < emb_dim1; iy++)
				score += xproj1[iy] * att_mat1[emb_dim1*j + iy];
			for(int iy = 0; iy < emb_dim2; iy++)
				score += xproj2[iy] * att_mat2[emb_dim2*j + iy];
			
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
			//cout << "Update W!" << endl;
			int ni = int(yp.at(i) - 1);
			for(int iy = 0; iy < emb_dim1; iy++)
			{
				for(int ix = 0; ix < dims; ix++)
				{
					emb_mat1[dims*iy + ix] -= eta * (xp[dims*i+ix] * (att_mat1[emb_dim1*best_index + iy] 
												- att_mat1[emb_dim1*ni+iy]) + lambda*emb_mat1[dims*iy + ix]); 
			//	cout << ix << "," << iy << "=" << emb_mat[dims*iy + ix] << endl;		
				}
			}
			for(int iy = 0; iy < emb_dim2; iy++)
			{
				for(int ix = 0; ix < dims; ix++)
				{
					emb_mat2[dims*iy + ix] -= eta * (xp[dims*i+ix] * (att_mat2[emb_dim2*best_index+iy] 
												- att_mat2[emb_dim2*ni+iy]) + lambda*emb_mat2[dims*iy+ix]); 
			//	cout << ix << "," << iy << "=" << emb_mat[dims*iy + ix] << endl;		
				}
			}
		}
		else{
      for(int iy = 0; iy < emb_dim1; iy++)
      {
        for(int ix = 0; ix < dims; ix++)
        {
          emb_mat1[dims*iy + ix] -= eta*lambda*emb_mat1[dims*iy + ix];
      //    g[dims*iy + ix] = lambda*emb_mat[dims*iy + ix];
       	}
       }
      for(int iy = 0; iy < emb_dim2; iy++)
      {
        for(int ix = 0; ix < dims; ix++)
        {
          emb_mat2[dims*iy + ix] -= eta*lambda*emb_mat2[dims*iy + ix];
      //    g[dims*iy + ix] = lambda*emb_mat[dims*iy + ix];
       	}
       }
    }
		t += 1;
		updateEta_tdecay();
	}
	delete d1;
	delete d2;
	delete xproj1;
	delete xproj2;
}

/// Testing
double SvmSgdSJE::test(const int imin, const int imax, double *xp, const yvec_t &yp, const int dims,  
		double *emb_mat1, double *emb_mat2, const int emb_dim1, const int emb_dim2, const double *att_mat1, const double *att_mat2, const char* prefix)
{
	cout << prefix << " Testing Multi-class for [" << imin << ", " << imax << "]." << endl;

 	assert(imin <= imax);
 	int nsamples = imax-imin+1;

  double* scores = new double[nclass*nsamples];
	for(int i = 0;i < nclass*nsamples; i++)
		scores[i] = 0.0;
  int* conf_mat = new int[nclass*nclass];
	//int* conf_mat = new int[nclass*nclass];
	for(int i = 0; i < nclass*nclass; i++)
		conf_mat[i] = 0;
	double *xproj1;
	double *xproj2;
  for(int i = 0; i < nsamples; i++)
  {
		// project test images onto label embedding space
		// project training images onto label embedding space
		xproj1 = getXproject(emb_mat1, &xp[dims*i], dims, emb_dim1);	
		xproj2 = getXproject(emb_mat2, &xp[dims*i], dims, emb_dim2);	
		// calculate the scores using dot product similarity using classifiers
		for(int j = 0; j < nclass; j++)
  	{
			for(int iy = 0; iy < emb_dim1; iy++)
				scores[nsamples*j + i] += xproj1[iy] * att_mat1[emb_dim1*j + iy];
			for(int iy = 0; iy < emb_dim2; iy++)
				scores[nsamples*j + i] += xproj2[iy] * att_mat2[emb_dim2*j + iy];
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
		//cout << "sample " << i << ", predicted_label=" << predicted_class;
		//cout << ", true_label=" << true_class << endl;
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
			for(int j = 0; j < nclass; j++)
			{
				sum_each_line = sum_each_line + conf_mat[i*nclass+j];
			}

			cout << " Class = " << i << " accuracy = " << setprecision(4) << double(conf_mat[i*nclass+i]/sum_each_line) << "%." << endl;
	
			sum_diag_conf = sum_diag_conf + double(double(conf_mat[i*nclass+i])/sum_each_line);
  	}
		double acc = double(sum_diag_conf / nclass);
  	cout << " Per " << prefix << " class accuracy = " << setprecision(4) << 100 * double(sum_diag_conf / nclass) << "%." << endl;

		delete xproj1;
		delete xproj2; 
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
	  	else if (att1_file_train == 0)
        att1_file_train = arg;
	  	else if (att1_file_test == 0)
        att1_file_test = arg;
	  	else if (att2_file_train == 0)
        att2_file_train = arg;
	  	else if (att2_file_test == 0)
        att2_file_test = arg;
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

int emb_dim1;		//dim of output embedding
int emb_dim2;
int cls_dim_train;
int cls_dim_test;
int nsamples;

double* emb_mat1;
double* emb_mat2;



int main(int argc, const char **argv)
{
	parse(argc, argv);
	config(argv[0]);
  
	// load the attribute vectors for training as FVector type 
	double* att1_mat_train; 
	double* att2_mat_train; 
  if(att1_file_train)
  	att1_mat_train = load_attribute_matrix(att1_file_train, emb_dim1, cls_dim_train);
	
  if(att2_file_train)
  	att2_mat_train = load_attribute_matrix(att2_file_train, emb_dim2, cls_dim_train);
  
	SvmSgdSJE svm_ale_mult_2(cls_dim_train, emb_dim1, emb_dim2, lambda, eta);
	xvec_t xtrain_fvec;
  // Load training images and their labels
  if (trainfile)
  	loadmult_datafile(trainfile, xtrain_fvec, ytrain, dims);//, 315);
	//for(int i = 0; i < xtrain_fvec.size(); i++)
		//xtrain_fvec.at(i).scale(1/sqrt(dot(xtrain_fvec.at(i),xtrain_fvec.at(i))));
	xtrain = xvec_to_double(xtrain_fvec.size(), dims, xtrain_fvec);
	double mean[dims];
	double variance[dims];
	normalization(xtrain, dims, xtrain_fvec.size(), mean, variance, true);	
	//allocate memory for emb_mat
  cout << "The size of the embedding matrix1 is " << emb_dim1 << "*"  << dims << " " << endl;
  cout << "The size of the embedding matrix2 is " << emb_dim2 << "*"  << dims << " " << endl;
  	
	  // initialize the label embedding space
	  int i,j;
		emb_mat1 = new double[emb_dim1*dims];
		emb_mat2 = new double[emb_dim2*dims];
	  double std_dev = 1.0 / sqrt(dims);  
	  for(i = 0; i < emb_dim1; i++)
	  {
			for( j = 0; j < dims; j++)
			{
				emb_mat1[dims*i + j] = std_dev * rand_gen();
			} 
			//double dot_prod = dot(*(emb_mat_fvec[i]), *(emb_mat_fvec[i]));
			//emb_mat_fvec[i]->scale(lambda/sqrt(dot_prod));
	  }	
	  for(i = 0; i < emb_dim2; i++)
	  {
			for( j = 0; j < dims; j++)
			{
				emb_mat2[dims*i + j] = std_dev * rand_gen();
			} 
			//double dot_prod = dot(*(emb_mat_fvec[i]), *(emb_mat_fvec[i]));
			//emb_mat_fvec[i]->scale(lambda/sqrt(dot_prod));
	  }	
  double *att1_mat_test; 	
  double *att2_mat_test; 	
	//save_embedding_matrix(embfile, emb_dim, dims, emb_mat);
  // load the attribute vectors for test as FVector type 
  if(att1_file_test)
  	att1_mat_test = load_attribute_matrix(att1_file_test, emb_dim1, cls_dim_test);
  if(att2_file_test)
  	att2_mat_test = load_attribute_matrix(att2_file_test, emb_dim2, cls_dim_test);
     	
  SvmSgdSJE svm_ale_mult_2_test(cls_dim_test, emb_dim1, emb_dim2, lambda, eta); 

	xvec_t xtest_fvec;
  	// Load test images and their labels 
  	if(testfile)
    	loadmult_datafile(testfile, xtest_fvec, ytest, dims);

	//for(int i = 0; i < xtest_fvec.size(); i++)
		//xtest_fvec.at(i).scale(1/sqrt(dot(xtest_fvec.at(i),xtest_fvec.at(i))));
	xtest = xvec_to_double(xtest_fvec.size(), dims, xtest_fvec);
	normalization(xtest, dims, xtest_fvec.size(), mean, variance, false);	
  	//double* xtest2 = xvec_to_double(xtest_fvec2.size(), dims, xtest_fvec2);
	int tmin = 0, tmax = xtest_fvec.size() - 1;

	// Training
  	int imin = 0, imax = xtrain_fvec.size() - 1; 
	Timer timer;
  	//begining phase
	cout << "The begining phase..." << endl;
	//test the accuracy of random matrix
	//svm_ale_mult_2.test(imin,imax,  xtrain, ytrain, dims, emb_mat, emb_dim, att_mat_train, "train");
    //svm_ale_mult_2_test.test(tmin, tmax, xtest, ytest, dims, emb_mat, 
			//					emb_dim, att_mat_test,"test");
	int ite_per_epoch = xtrain_fvec.size();
	nsamples = imax + 1;
	int epochs_first = xtrain_fvec.size()*epochs/ite_per_epoch;
	int best_nepoch = 0;
	double best_accuracy = 0;
	double cur_accuracy = 0.0;
	//double best_emb_mat[emb_dim*dims];
	for(int i = 0; i < epochs_first; i++)
  {
		//timer.start();
		cout << "Epoch " << i << endl; 
  	svm_ale_mult_2.train(imin, imax, ite_per_epoch, xtrain, ytrain, dims, att1_mat_train, att2_mat_train, cls_dim_train, emb_mat1, emb_mat2, emb_dim1, emb_dim2);
		//svm_ale_mult_2.test(imin, imax,  xtrain, ytrain, dims, emb_mat, emb_dim, att_mat_train, "train");
		cur_accuracy = svm_ale_mult_2_test.test(tmin, tmax, xtest, ytest, dims, emb_mat1, emb_mat2, emb_dim1, emb_dim2, att1_mat_test, att2_mat_test, "test");
		if(cur_accuracy > best_accuracy){
			best_accuracy = cur_accuracy;
			best_nepoch = i + 1; 
			//memcpy(best_emb_mat, emb_mat, emb_dim*dims*sizeof(double));
		}	
		//double risk = svm_ale_mult_2.evaluate_objective(xtrain, ytrain, emb_mat, att_mat_train);
		//cout << "Risk=" << risk << endl;
  		//timer.stop();
     	//cout << "Total training time " << setprecision(6) << timer.elapsed() << " secs." << endl;
	}
	cout << "Per class accuracy=" << best_accuracy << " ,eta=" << eta << " ,nepoch=" << best_nepoch <<endl; 
	cout << cur_accuracy << " " << eta << " " << lambda << " " << best_nepoch << endl;
	//save_embedding_matrix(embfile, emb_dim, dims, best_emb_mat);

	delete emb_mat1;
	delete emb_mat2;
  return 0;
}


