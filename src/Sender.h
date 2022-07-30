/*
 * AgentController.h
 *
 *  Created on: Oct 28, 2011
 *      Author: steffen
 */

#ifndef SENDER_H_
#define SENDER_H_

#include "Agent.h"



class Sender : public Agent {
public:
	Sender(const Setting &settings);
	virtual ~Sender();
	void update(const double targetPos, const int timeStep, bool receiverInCommArea, const int targetID);
	void initTrial(double startingPos, double targetPos);
	void setBehavioralGenes(vector<float> &genes, int start);
	void modularity();

	double signal;
	double targetSensor;
	bool hasSeenTarget;
	bool constSignal;
	bool constSpeed;
  int placeSwitch;

private:
	void updateTargetSensor();
	bool noSignal;
	bool useBehavioralGenes;
	bool noMovement;
	HelperFunction* _help;
	double _signalThreshold;
};

#endif /* SENDER_H_ */
