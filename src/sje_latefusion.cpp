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
const char *embmatfile1 = 0;
const char *embmatfile2 = 0;
const char *meanfile = 0;
const char *variancefile = 0;
const char *maxfile = 0;

bool normalize = true;
bool isval = false;
double lambda = 1e-63;
double eta = 1e-5;

int epochs = 5;
int maxtrain = -1;
int nclass = 0;

double alpha = 0.0;

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
  double test(double* score_mat1, double* score_mat2, const yvec_t &yp, int nsamples, int nclass, double alpha);

  void getScore(const int imin, const int imax, double *x,  const int dims, double *emb_mat1, double *emb_mat2, const int emb_dim1, const int emb_dim2, const double *att_mat1, const double *att_mat2,double* score_mat1, double* score_mat2);

	double* getXproject(double* emb_mat, double *x, int dim, int emb_dim);
private:
  int nclass;
  int dim1;
	int dim2;
  double  lambda;
  double  eta,eta0;
  int t;
};

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

/// Testing
void SvmSgdSJE::getScore(const int imin, const int imax, double *xp, const int dims,  
		double *emb_mat1, double *emb_mat2, const int emb_dim1, const int emb_dim2, const double *att_mat1, const double *att_mat2, double *score_mat1, double *score_mat2)
{
	cout  << " Testing Multi-class for [" << imin << ", " << imax << "]."  << endl;

 	assert(imin <= imax);
 	int nsamples = imax-imin+1;

	for(int i = 0;i < nclass*nsamples; i++){
		score_mat1[i] = 0.0;
		score_mat2[i] = 0.0;
	}
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
				score_mat1[nsamples*j + i] += xproj1[iy] * att_mat1[emb_dim1*j + iy];
			for(int iy = 0; iy < emb_dim2; iy++)
				score_mat2[nsamples*j + i] += xproj2[iy] * att_mat2[emb_dim2*j + iy];
		}
 	}

	delete xproj1;
	xproj1 = NULL;
	delete xproj2; 
	xproj2 = NULL;
}
/// Testing
double SvmSgdSJE::test(double *score_mat1, double *score_mat2, const yvec_t &yp,  
													int nsamples, int nclass, double alpha)
{

	double beta = 1 - alpha;
  int* conf_mat = new int[nclass*nclass];
	for(int i = 0; i < nclass*nclass; i++)
		conf_mat[i] = 0;

  double nerr = 0;
  for(int i = 0; i < nsamples; i++)
  {
		int true_class = int(yp.at(i)-1);

		double max_score = -1e10f;
		int predicted_class = -1;

		for(int c = 0; c < nclass; c++)
		{
			if(score_mat1 == NULL || score_mat2 == NULL)
				cout << "Error" << endl;
			double score =  alpha*score_mat1[nsamples*c+i] + beta*score_mat2[nsamples*c+i]; 
			if ( score > max_score )
			{
				predicted_class = c;
				max_score = score;
			}
		}
		if(predicted_class == -1)
			cout << "Warning: max_score is below -1e10" << endl;

		conf_mat[nclass*true_class+predicted_class]++;
	
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

  	cout << "alpha=" << alpha << ", beta=" << beta << ", Per " << "class accuracy = " << setprecision(4) << 100 * double(sum_diag_conf / nclass) << "%." << endl;

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
	  	else if (embmatfile1 == 0)
        embmatfile1 = arg;
	  	else if (embmatfile2 == 0)
        embmatfile2 = arg;
	  	else if (meanfile == 0)
        meanfile = arg;
	  	else if (variancefile == 0)
        variancefile = arg;
	  	else if (maxfile == 0)
        maxfile = arg;
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
	  			else if (opt == "alpha" && i+1<argc)
     			{
              alpha = atof(argv[++i]);
              assert(alpha>=0);
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
double* xtest;
yvec_t ytest;

int emb_dim1;		//dim of output embedding
int emb_dim2;
int cls_dim_train;
int cls_dim_test;
int nsamples;

double* emb_mat1;
double* emb_mat2;
double* emb_tensor;


double frand(double fMin, double fMax)
{
    double f = (double)rand() / (double)RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main(int argc, const char **argv)
{
	parse(argc, argv);
	config(argv[0]);
 	
	srand(time(NULL)); 
	xvec_t xtest_fvec;
 // Load test images and their labels 
  if(testfile)
  	loadmult_datafile(testfile, xtest_fvec, ytest, dims);

	xtest = xvec_to_double(xtest_fvec.size(), dims, xtest_fvec);

	double *mean;
	double *variance;
	double max;
	mean = load_mean(meanfile, dims);
	variance = load_variance(variancefile, dims);
	max = load_max(maxfile);
	cout << "max=" << endl; 	
	normalization(xtest, dims, xtest_fvec.size(), mean, variance, false);	
	normalization2(xtest, dims, xtest_fvec.size(), max, false);	

  double *att1_mat_test; 	
  double *att2_mat_test; 	
  // load the attribute vectors for test as FVector type 
  if(att1_file_test)
  	att1_mat_test = load_attribute_matrix(att1_file_test, emb_dim1, cls_dim_test);
  if(att2_file_test)
  	att2_mat_test = load_attribute_matrix(att2_file_test, emb_dim2, cls_dim_test);
     	
  SvmSgdSJE svm_ale_mult_2_test(cls_dim_test, emb_dim1, emb_dim2, lambda, eta); 
	int tmin = 0, tmax = xtest_fvec.size() - 1;

	emb_mat1 = load_embedding_matrix(embmatfile1, emb_dim1, dims);
	emb_mat2 = load_embedding_matrix(embmatfile2, emb_dim2, dims);

	int nsamples = xtest_fvec.size();
	double *score_mat1 = new double[nsamples*cls_dim_test];
	double *score_mat2 = new double[nsamples*cls_dim_test];
	svm_ale_mult_2_test.getScore(tmin, tmax, xtest, dims, emb_mat1, emb_mat2, emb_dim1, emb_dim2, att1_mat_test, att2_mat_test, score_mat1, score_mat2);

	if(isval){
		double cur_accuracy = 0.0;
		double best_accuracy = 0.0;
		double best_alpha = 0.0;
		cout << "Start to cross validation on alpha..." << endl;
		for(int i = 0; i < 30; i++){
			alpha = frand(0.0, 1.0);
			cout << "alpha=" << alpha << endl;
			cur_accuracy = svm_ale_mult_2_test.test(score_mat1, score_mat2, ytest, xtest_fvec.size(), cls_dim_test, alpha);
			if(cur_accuracy > best_accuracy){
				best_accuracy = cur_accuracy;
				best_alpha = alpha;
			}
		}
		cout << "Per test class accuracy=" << best_accuracy << endl;
		cout << best_accuracy << " " << best_alpha << endl;
	}
	else{
		svm_ale_mult_2_test.test(score_mat1, score_mat2, ytest, xtest_fvec.size(), cls_dim_test, alpha);
	}
  return 0;
}

