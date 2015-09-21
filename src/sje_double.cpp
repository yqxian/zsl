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
  void test(int imin, int imax, double *x, const yvec_t &y, int dims,
	double *emb_mat, int emb_dim, double *att_mat, const char *prefix = "");
private:
  int nclass;
  int dim;
  double  lambda;
  double  eta;
  int t;
};


/// Constructor
SvmSgdSJE::SvmSgdSJE(int _nclass, int _dim, double _lambda, double _eta)
{
	nclass = _nclass;
	dim = _dim;

	lambda = _lambda;
	eta = _eta;

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
		for(int iy = 0; iy < emb_dim; iy++)
			xproj[iy] = xproj[iy] / xproj_norm;	
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
			int ni = int(yp.at(i) - 1);
			for(int iy = 0; iy < emb_dim; iy++)
			{
				for(int ix = 0; ix < dims; ix++)
				{
					emb_mat[dims*iy + ix] += eta * xp[dims*i + ix] * (att_mat[emb_dim*ni + iy] 
												- att_mat[emb_dim*best_index + iy]); 
				}
			}
		}
		t += 1;
	}
}

/// Testing
void SvmSgdSJE::test(int imin, int imax, double *xp, const yvec_t &yp, int dims,  
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
			for(int j = 0; j < nclass; j++)
  			{
				for(int iy = 0; iy < emb_dim; iy++)
					scores[nsamples*j + i] += xproj[iy] * att_mat[emb_dim*j + iy];
			}
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
  	cout << " Per class accuracy = " << setprecision(4) << 100 * double(sum_diag_conf / nclass) << "%." << endl;

 
  	delete scores;
  	delete conf_mat;
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

int emb_dim;		//dim of output embedding
int cls_dim_train;
int cls_dim_test;

double* emb_mat;

int main(int argc, const char **argv)
{
	parse(argc, argv);
	config(argv[0]);
  
	// load the attribute vectors for training as FVector type 
	double* att_mat_train; 
  	if (att_file_train)
     	att_mat_train = load_attribute_matrix(att_file_train, emb_dim, cls_dim_train);

  	SvmSgdSJE svm_ale_mult_2(cls_dim_train, emb_dim, lambda, eta);
	xvec_t xtrain_fvec;
  	// Load training images and their labels
  	if (trainfile)
    	loadmult_datafile(trainfile, xtrain_fvec, ytrain, dims);//, 315);
	xtrain = xvec_to_double(xtrain_fvec.size(), dims, xtrain_fvec);
	//allocate memory for emb_mat
    cout << "The size of the embedding matrix is " << emb_dim << "*"  << dims << " " << endl;
  	
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
	Timer timer;
  	for(int i = 0; i < epochs; i++)
  	{
    	cout << "--------- Epoch " << i+1 << "." << endl;
		timer.start(); 
  		svm_ale_mult_2.train(imin, imax, xtrain, ytrain, dims, att_mat_train, cls_dim_train, emb_mat, emb_dim);
  		timer.stop();
     	cout << "Total training time " << setprecision(6) << timer.elapsed() << " secs." << endl;
  	}

  	save_embedding_matrix(embfile, emb_dim, dims, emb_mat);
    double *att_mat_test; 	
  	// load the attribute vectors for test as FVector type 
  	if (att_file_test)
     	att_mat_test = load_attribute_matrix(att_file_test, emb_dim, cls_dim_test);

  	SvmSgdSJE svm_ale_mult_2_test(cls_dim_test, emb_dim, lambda, eta); 

	xvec_t xtest_fvec;
  	// Load test images and their labels 
  	if(testfile)
    	loadmult_datafile(testfile, xtest_fvec, ytest, dims);
	xtest = xvec_to_double(xtest_fvec.size(), dims, xtest_fvec);

  	int tmin = 0, tmax = xtest_fvec.size() - 1;

  	// Testing
  	if(tmax >= tmin) // if there are any testing samples at all
  	{ 
    	svm_ale_mult_2_test.test(tmin, tmax, xtest, ytest, dims, emb_mat, emb_dim, att_mat_test,"test:  ");
  	}
	
	delete[] emb_mat;
  	return 0;
}


