/*
 * PopulationRecoder.h
 *
 *  Created on: Nov 18, 2011
 *      Author: steffen
 */

#ifndef POPULATIONRECODER_H_
#define POPULATIONRECODER_H_

#include "HelperFunction.h"
#include "SimpleLog.h"

#include <iostream>
#include <stdio.h>
#include <fstream>

class PopulationRecoder {
public:
	PopulationRecoder(bool recordAllInfor);
	virtual ~PopulationRecoder();
	void clearRecords();
	void setIndInfo(double all, double allSender, vector<SimpleLog*> &allLogs);
	void setIndInfo2(double sigDiv, double velDiv, double commTime);
	void setIndInfoGenes(bool sigGene, bool velGene);
	void setOnset(vector<int>& onsets);
	// void setIndInfoOnsetLength(vector<int> onsets, vector<int> lengths);
	void setTargetsFound(vector<int>& targetsFound);
	void setTargetsDiscriminated(vector<int>& targetsDiscriminated);
	void writeData(int gen);
	void writeData2(int gen);

	vector< vector<double> > channels;
	vector<bool> signalGene;
	vector<bool> speedGene;
	vector< vector<int> > vecOnsets;
	vector< vector<int> > vecLengths;
	vector< vector<int> > vecTargetsFound;
	vector< vector<int> > vecTargetsDiscriminated;
	bool recordInfo;

private:
	HelperFunction *_help;
	ofstream logFile;
	ofstream _diversityLogFile;
	vector<vector<double> > _diversityData;
	ofstream _corrOnsetLengthLogFile;
	ofstream _targetsFoundLogFile;
	ofstream _targetsDiscriminatedLogFile;
};

#endif /* POPULATIONRECODER_H_ */
