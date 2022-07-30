/*
 * MyGa.h
 *
 *  Created on: Jan 12, 2012
 *      Author: steffen
 */

#ifndef MYGA_H_
#define MYGA_H_

#include <ga/GARealGenome.h>
#include <vector>
#include <map>
#include "random.h"
#include "HelperFunction.h"

using namespace std;

class MyGa
{
public:
  MyGa(const char* logFileName);
  virtual ~MyGa();
  void select(bool recordAllInd);
  void writePopulation(bool parents);
  void writeStats(bool parents);
  void selectRW(bool recordAllInd);

  vector<GARealGenome> genomes;
  int popSize;
  double mutRate;
  int generation;
  int noOfGenerations;
  bool done;
  int tSize;
  bool normalizeMutRate;
  int fixPopulation;
  int loadGeneration;
private:


  ofstream logFile;
  ofstream logFileParents;
  char _filePrefix[100];

};

#endif /* MYGA_H_ */
