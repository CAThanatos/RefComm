/*
 * SimpleLog.cpp
 *
 *  Created on: Jan 24, 2012
 *      Author: steffen
 */

#include "SimpleLog.h"

SimpleLog::SimpleLog() {
	values.resize(HelperFunction::Instance()->getNoOfTargets(),0.0);

  //map<double, int> probs;
  //tmpBool=true;
  //information=0.0;
  //tmp=0.0;
  //tmp2=0.0;
}

SimpleLog::~SimpleLog() {
	values.clear();
}

