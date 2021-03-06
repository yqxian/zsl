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

#ifndef DATA_MULT_H
#define DATA_MULT_H

#include <vector>
#include "vectors.h"

typedef std::vector<SVector> xvec_t;
typedef std::vector<double>  yvec_t;

void loadmult_datafile(const char *filename, 
                   xvec_t &xp, yvec_t &yp, int &maxdim,
                   int maxrows = -1);

double* load_attribute_matrix(const char *fname, int &emb_dim, int &cls_dim);
double* load_embedding_matrix(const char *fname, int &emb_dim, int &dims);
void save_embedding_matrix(const char *fname, int &emb_dim, int &dims, double* emb_mat);

double **array2D(int width, int height);
double rand_gen();
double* xvec_to_double(int nsample, int ndim, xvec_t xp);
void normalization2(double *xp, int dim, int nsamples, double &max, bool istrain);
void normalization(double *xp, int dim, int nsamples, double *mean, double  *variance, bool istrain);
#endif
