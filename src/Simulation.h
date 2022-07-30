/*
 * Simulation.h
 *
 *  Created on: Oct 31, 2011
 *      Author: steffen
 */

#ifndef SIMULATION_H_
#define SIMULATION_H_
#include <vector>
#include <map>
#include <stdio.h>
#include "Receiver.h"
#include "Sender.h"
#include "HelperFunction.h"
#include "PopulationRecoder.h"
#include "SimpleLog.h"



//18 for ten, 8 for five, 4 for three



class Simulation {
public:
	Simulation(const Setting& root);
	virtual ~Simulation();
	double evaluate(vector<float> &genes, int generation, PopulationRecoder *popRecorder, vector<double> &retVals);
	bool activateReceiver;
private:
	void updateLoggingInfoEndTrial(int target);
	void updateLoggingInfoStep(int step);
	void initLoggingTrial();
	void finalizeLogging(PopulationRecoder *& popRecorder, const float & retFitness);
	double calcDistance(vector<vector<double> > &data);

  SimpleLog *_aSignalAmplitude;
  SimpleLog *_aSignalAmplitudeSender;

	SimpleLog *_aSignalDuration;
  SimpleLog *_aSignalDurationSender;

  SimpleLog *_timeInEnvSender;
  SimpleLog *_returnTimeSender;
  SimpleLog *_velSender;
  SimpleLog *_velSenderInE;
  SimpleLog *_velSenderInComm;

  vector<SimpleLog*> _allLogs;
  vector<SimpleLog*> _allLogs2total;
  vector<SimpleLog*> _allLogs2totalSender;

	Receiver *_receiver;
	Sender *_sender;
	//double _halfDistanceBetweenTargets;
	vector<double> _targetPositions;
	double _startingPositions[5];
	double _MAX_INFO;
	double _totalInformation;
	double _totalInformationSender;
	HelperFunction* _help;
};

#endif /* SIMULATION_H_ */
