/*
 * SimpleLog.h
 *
 *  Created on: Jan 24, 2012
 *      Author: steffen
 */
#ifndef SIMPLELOG_H_
#define SIMPLELOG_H_

#include<vector>
#include <map>
#include"HelperFunction.h"
using namespace std;

class SimpleLog {
public:
	SimpleLog();
	virtual ~SimpleLog();
  vector<double> values;
  bool tmpBool;
  double information;
  double tmp;
  double tmp2;
  map<double, int> probs;

};

#endif /* SIMPLELOG_H_ */
