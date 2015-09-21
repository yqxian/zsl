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


#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cmath>

#include "gzstream.h"
#include "assert.h"
#include "data_mult.h"

using namespace std;

double* xvec_to_double(int nsample, int ndim, xvec_t xp)
{
	double *x_double = new double[ndim*nsample];
    for(int i = 0; i < nsample; i++){
    	SVector x = xp.at(i);
        const SVector::Pair *p = x;
        if(p){
        	for(int j = 0; j < ndim; j++){
            	if(j == p->i ){
                	x_double[ndim*i + j] = (double)p->v;
                    p++;
                 }
                 else
                 	x_double[ndim*i + j] = 0.0;
           }
       }
	}
    return x_double;
}

static void
loadmult_datafile_sub(istream &f, bool binary, const char *fname, 
                  xvec_t &xp, yvec_t &yp, int &maxdim, int maxrows)
{
  cout << "# Reading file " << fname << endl;
  if (! f.good())
    assertfail("Cannot open " << fname);

  int pcount = 0;
  while (f.good() && maxrows--)
    {
	double y;
      	SVector x;
	y = (f.get());
        x.load(f);
      
      if (f.good())
        {          
          xp.push_back(x);
          yp.push_back(y);
          pcount += 1;
          if (x.size() > maxdim)
            maxdim = x.size();
        }
    }
  cout << "# Read " << pcount << " examples." << endl;
  

}


void
loadmult_datafile(const char *fname, 
              xvec_t &xp, yvec_t &yp, int &maxdim, int maxrows)
{
  bool binary = true;
  string filename = fname;
  igzstream f;
  f.open(fname);
  return loadmult_datafile_sub(f, binary, fname, xp, yp, 
                               maxdim,  maxrows);
}


void
load_classes(const char *fname, int &nclass)
{ 
  string filename = fname;
  ifstream f;
  SVector x;
  f.open(fname);
  x.load(f);
  nclass = x.size();  
}




double **array2D(int width, int height) { /* w lines, h columns */
  double **mat = NULL;
  if (height != 0 && width != 0) {
    mat = (double **) malloc(height * sizeof(double *));
    double  *tmp = (double *)  calloc(height * width, sizeof(double));
  
    for (int i = 0; i < height; i++) {
      mat[i] = tmp;
      tmp += width;
    }
  }
  return mat;
}
double*
load_attribute_matrix(const char *fname, int &emb_dim, int &cls_dim)
{
    FILE *f = fopen(fname, "rb");
  	cout << "# Reading file " << fname << endl;
    fread(&emb_dim, sizeof(int), 1, f);
	fread(&cls_dim, sizeof(int), 1, f);
    double *att_mat = new double[emb_dim*cls_dim];
    fread(att_mat, sizeof(double), emb_dim * cls_dim, f);
    fclose(f);

    return att_mat;
}

/*FVector**
load_attribute_matrix(const char *fname, int &emb_dim, int &cls_dim)
{
  FILE *f = fopen(fname, "rb");
  fread(&emb_dim, sizeof(int), 1, f);
  fread(&cls_dim, sizeof(int), 1, f);
  double *att_mat = new double[emb_dim*cls_dim];
  fread(att_mat, sizeof(double), emb_dim * cls_dim, f);   
  fclose(f);

  FVector** att_mat_fvec = new  FVector*[cls_dim];
  int i,j;
  for(i=0;i<cls_dim;i++)
  {
      att_mat_fvec[i] = new FVector[1];
      for(j=0;j<emb_dim;j++)
      {
        att_mat_fvec[i]->set(j,att_mat[i*emb_dim+j]);
        //cout << j << ":" << att_mat_fvec[i]->get(j) << " "<< endl;
      }
  }

  delete[] att_mat;
  return(att_mat_fvec);
}
*/
double rand_gen(){
	double a =0.0, b=0.0;
	while (a == 0.0){
		a = double(double(rand())/double(RAND_MAX));
	}
	while (b == 0.0){
		b = double(double(rand())/double(RAND_MAX));
	}
	return (sqrt(-2.0*log(a)) * cos(2*M_PI*b));
}


void
save_embedding_matrix(const char *fname, int &emb_dim, int &dims, FVector** emb_mat)
{
  FILE *f = fopen(fname, "wb");
  fwrite(&emb_dim, sizeof(int), 1, f);
  fwrite(&dims, sizeof(int), 1, f);
  double *emb= new double[emb_dim*dims];
  for (int e=0; e<emb_dim; e++)
  {
	for (int i=0; i<dims; i++)
	{
		emb[dims*e+i] = emb_mat[e]->get(i);		
	}
  }
  fwrite(emb, sizeof(double), emb_dim*dims, f); 
  fclose(f);
}

/*void
save_embedding_tensor(const char *fname, int &emb_dim1, int &emb_dim2, int &dims, FVector ***emb_tensor_fvec)
{
  FILE *f = fopen(fname, "wb");
  fwrite(&emb_dim1, sizeof(int), 1, f);
  fwrite(&emb_dim2, sizeof(int), 1, f);
  fwrite(&dims, sizeof(int), 1, f);
  double *emb= new double[emb_dim2*dims];
  for (int e1 = 0; e1 < emb_dim1; e1++)
  {
	for(int e2 = 0; e2 < emb_dim2; e2++){
		for (int i=0; i<dims; i++)
		{
			emb[dims*e2 + i] = emb_tensor_fvec[e1][e2]->get(i);		
		}
	}
	//since tensor is so large to be in the memory, per write: emb_dim2*dims doubles 
	fwrite(emb, sizeof(double), emb_dim2*dims, f); 
  }
  fclose(f);
}*/

void
save_emb_tensor(const char *fname, int &emb_dim1, int &emb_dim2, int &dims, double *emb_tensor)
{
	FILE *f = fopen(fname, "wb");
  	cout << "# Writing embedding tensor to file " << fname << endl;
  	fwrite(&emb_dim1, sizeof(int), 1, f);
  	fwrite(&emb_dim2, sizeof(int), 1, f);
  	fwrite(&dims, sizeof(int), 1, f);
  	double *emb= new double[emb_dim2*dims];
	int saver = dims*emb_dim2;
  	for (int e1 = 0; e1 < emb_dim1; e1++)
  	{
		for(int e2 = 0; e2 < emb_dim2; e2++){
			for (int i=0; i<dims; i++)
			{
				emb[dims*e2 + i] = emb_tensor[saver*e1 + dims*e2 + i];		
			}
		}
		//since tensor is so large to be in the memory, per write: emb_dim2*dims doubles 
		fwrite(emb, sizeof(double), saver, f); 
  	}
	delete[] emb;
  	fclose(f);
}
/*
FVector**
load_emb_matrix(const char *fname, int &emb_dim,  int &dims)
{
  FILE *f = fopen(fname, "rb");
  fread(&emb_dim, sizeof(int), 1, f);
  fread(&dims, sizeof(int), 1, f);
  double *emb_mat = new double[emb_dim*dims];
  fread(emb_mat, sizeof(double), emb_dim * dims, f);   
  fclose(f);

  FVector** emb_mat_fvec = new  FVector*[emb_dim];
  int e,i;
  for(e=0;e<emb_dim;e++)
  {
      emb_mat_fvec[e] = new FVector(dims);
      for(i=0;i<dims;i++)
      {
        emb_mat_fvec[e]->set(i,emb_mat[dims*e+i]);
      }
  }

  delete[] emb_mat;
  return(emb_mat_fvec);
}
*/
/*void
load_emb_tensor(const char *fname, int &emb_dim1, int &emb_dim2, int &dims, FVector ***emb_tensor_fvec)
{
  FILE *f = fopen(fname, "rb");
  fread(&emb_dim1, sizeof(int), 1, f);
  fread(&emb_dim2, sizeof(int), 1, f);
  fread(&dims, sizeof(int), 1, f);
  double *emb_mat = new double[emb_dim2*dims];
  
  for (int e1 = 0; e1 < emb_dim1; e1++)
  {
	fread(emb_mat, sizeof(double), emb_dim2 * dims, f);   	
	for(int e2 = 0; e2 < emb_dim2; e2++){
		for (int i=0; i<dims; i++)
		{
			emb_tensor_fvec[e1][e2]->set(i, emb_mat[dims*e2+i]);		
		}
	}
  }
  
  fclose(f);

  delete[] emb_mat;
}*/
void
load_emb_tensor(const char *fname, int &emb_dim1, int &emb_dim2, int &dims, double *emb_tensor)
{
  	cout << "# Reading embedding tensor to file " << fname << endl;
	FILE *f = fopen(fname, "rb");
  	fread(&emb_dim1, sizeof(int), 1, f);
  	fread(&emb_dim2, sizeof(int), 1, f);
  	fread(&dims, sizeof(int), 1, f);
  	double *emb_mat = new double[emb_dim2*dims];
	int saver = emb_dim2*dims;  
  	for (int e1 = 0; e1 < emb_dim1; e1++)
  	{
		fread(emb_mat, sizeof(double), saver, f);   	
		for(int e2 = 0; e2 < emb_dim2; e2++){
			for (int i=0; i<dims; i++)
			{
				emb_tensor[dims*emb_dim2*e1 + dims*e2 + i] = emb_mat[dims*e2+i];		
			}
		}
  	}
  
  	fclose(f);

  	delete[] emb_mat;
}

double*
load_embedding_matrix(const char *fname, int &emb_dim, int &dims)
{
    FILE *f = fopen(fname, "rb");
    cout << "# Reading embedding matrix file " << fname << endl;
    fread(&emb_dim, sizeof(int), 1, f);
    fread(&dims, sizeof(int), 1, f);
    double *emb_mat = new double[emb_dim*dims];
    fread(emb_mat, sizeof(double), emb_dim * dims, f);
    fclose(f);

    return emb_mat;
}

double* load_variance(const char *fname, int dim){
		cout << "load variance..." << endl;	
		double *variance = new double[dim];
    FILE *f = fopen(fname, "rb");
    fread(variance, sizeof(double), dim, f);
    fclose(f);
		return variance;
}

double* load_mean(const char *fname, int dim){
		cout << "load mean..." << endl;	
		double *mean = new double[dim];
    FILE *f = fopen(fname, "rb");
    fread(mean, sizeof(double), dim, f);
    fclose(f);
		return mean;
}

double load_max(const char *fname){
	cout << "load max..." << endl;
  double max = -1;
  FILE *f = fopen(fname, "rb");
  fread(&max, sizeof(double), 1, f);
  fclose(f);
	return max;
}

void normalization(double *xp, int dim, int nsamples, double *mean, double *variance,bool istrain){
  if(istrain){
    for(int i = 0; i < dim; i++){
      double square_sum = 0.0;
      double sum = 0.0;
      for(int j = 0; j < nsamples; j++){
        sum += xp[j*dim+i];
      }
      mean[i] = (double)(sum/nsamples);
      for(int j = 0; j < nsamples; j++){
        xp[j*dim+i] -= mean[i];
        square_sum += xp[j*dim+i]*xp[j*dim+i];
      }
      square_sum = square_sum / nsamples;
      variance[i] = sqrt(square_sum);
      for(int j = 0; j < nsamples; j++){
        if(variance[i] != 0)
          xp[j*dim+i] = xp[j*dim+i]/variance[i];
      }
    }
  }
  else{
    for(int i = 0; i < dim; i++){
      for(int j = 0; j < nsamples; j++){
        xp[j*dim+i] -= mean[i];
        if(variance[i] != 0)
          xp[j*dim+i] = xp[j*dim+i]/variance[i];
      }
    }
  }
}

void normalization2(double *xp, int dim, int nsamples, double &max, bool istrain){
  if(istrain){
    max = -1;
    for(int i = 0; i < nsamples; i++)
      for(int j = 0; j < dim; j++)
        if(max < xp[i*dim + j])
          max = xp[i*dim + j];
    if(max != 0){
      for(int i = 0; i < nsamples; i++)
        for(int j = 0; j < dim; j++)
          xp[i*dim + j] /= max;
    }
  }
  else{
    if(max != 0){
      for(int i = 0; i < nsamples; i++)
        for(int j = 0; j < dim; j++)
          xp[i*dim + j] /= max;
    }
  }
}


