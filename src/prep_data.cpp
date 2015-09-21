// -*- C++ -*-
// SVM with stochastic gradient (preprocessing)
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
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

#include "assert.h"
#include "vectors.h"
#include "gzstream.h"

using namespace std;

#define DATADIR "/BS/akata_projects/work/data/CUB/"
#define DATAFILE_TRAIN "train_comb_zsh"
#define DATAFILE_TEST "val_comb_zsh"


typedef vector<SVector> xvec_t;
typedef vector<double> yvec_t;

int load(string fname, xvec_t &xp, yvec_t &yp)
{
  cerr << "# Reading " << fname.c_str() << endl;
  ifstream f(fname.c_str());
  if (! f.good())
    assertfail("Cannot open file " << fname.c_str());
  int count = 0;
  while (f.good())
    {
      double y;
      SVector x;
      f >> y >> x;
      if (f.good())
        {
          xp.push_back(x);
          yp.push_back(y);
          count += 1;
        }
    }
  if (! f.eof())
    assertfail("Failed reading " << fname.c_str());
  cerr << "# Done reading " << count << " examples." << endl;
  return count;
}

void saveBinary(string fname, xvec_t &xp, yvec_t &yp, vector<int> &index, int imax)
{
  
  cerr << "# Writing " << fname.c_str() << endl;
  ogzstream f;
  f.open(fname.c_str());
  if (! f.good())
    assertfail("ERROR: cannot open " << fname << " for writing.");
  int count = 0; 

  for (int ii=0; ii<imax; ii++)
    {
      int i = index[ii];
      double y = yp[i];
      SVector x = xp[i];
      // This is where we keep the original label of the data and not 1,0.
      // casting of label to int is required to make same number of bytes being written in the binary file
      f.put(y);
      x.save(f);
      count += 1;
    }
  cerr << "\n# Wrote " << count << " examples." << endl;
}

void print_help()
{
	cout << "Usage : pre_fisher --data_dir <path to directory where txt file containing data is located > \n--train_file <prefix of training file without the extension> \n--test_file <prefix of test file without the extension>" << endl;
	exit(0);

}
void parseArg(int argc, const char** argv, string &Dir, string &fileName_train, string &fileName_test)
{
	argc--;
	argv++;

	while(argc)
	{
		argc--;
		if(!strcmp(*argv,"--data_dir"))
		{
			argc--;argv++;
			Dir.clear();
			Dir.append(*argv);
			argv++;
		}
		else if(!strcmp(*argv,"--train_file"))
		{
			argc--;argv++;
			fileName_train.clear();
			fileName_train.append(*argv);
			argv++;
		}
		else if(!strcmp(*argv,"--test_file"))
		{
			argc--;argv++;
			fileName_test.clear();
			fileName_test.append(*argv);
			argv++;
		}
		else if(!strcmp(*argv,"--help") || !strcmp(*argv,"--h") || !strcmp(*argv,"-h"))
		{
			argc--; argv++;
			print_help();
			argv++;
		}
		else
		{
			cout << "Un-recognized option "<< *argv << endl;
			print_help();
		}
	}
	
	
	if(!fileName_train.size())
	{
		// default filename defined at top of the file
		fileName_train = string(DATAFILE_TRAIN);
	}
	if(!fileName_test.size())
	{
		// default filename defined at top of the file
		fileName_test = string(DATAFILE_TEST);
	}
  	if(!Dir.size())
	{
		// default directory 
		Dir = string(DATADIR);
	}
}
int main(int argc, const char** argv)
{
  // load data [structure to hold the data]
  vector<SVector> xp_train;
  vector<double> yp_train;
  vector<SVector> xp_test;
  vector<double> yp_test;

  // string argument to hold the data file name
  string dataFile_train;
  string dataFile_test;
  string dataDir;

  
  // parsing command line arguments

  parseArg( argc, argv, dataDir, dataFile_train, dataFile_test);
  int count_train = load(dataDir + dataFile_train +  string(".txt"), xp_train, yp_train);
  int count_test = load(dataDir + dataFile_test +  string(".txt"), xp_test, yp_test);

  //////////////////////////////////////////////////////
  //			TRAIN FILE                    //
  //////////////////////////////////////////////////////

  // compute random shuffle for train file
  cerr << "# Shuffling" << endl;
  vector<int> index_train(count_train);

  if( !count_train )
	cout << "No samples read from the file " << dataFile_train.c_str() << " please check the file is in the svmlight format " << endl;

  for (int i=0; i<count_train; i++) index_train[i] = i;

  // random shuffling of indices
  random_shuffle(index_train.begin(), index_train.end());
  random_shuffle(index_train.begin(), index_train.end());
  
  // training file
  saveBinary(dataDir + dataFile_train + string(".train.bin.gz"), xp_train, yp_train, index_train, count_train);  

  //////////////////////////////////////////////////////
  //			TEST FILE                     //
  //////////////////////////////////////////////////////

  // for test file

  cerr << "# Shuffling" << endl;
  vector<int> index_test(count_test);

  if( !count_test )
	cout << "No samples read from the file " << dataFile_test.c_str() << " please check the file is in the svmlight format " << endl;

  for (int i=0; i<count_test; i++) index_test[i] = i;
  
  // training file
  saveBinary(dataDir + dataFile_test + string(".test.bin.gz"), xp_test, yp_test, index_test, count_test);  

}
