/*
 * Receiver.h
 *
 *  Created on: Oct 28, 2011
 *      Author: steffen
 */

#ifndef RECEIVER_H_
#define RECEIVER_H_

#include "Agent.h"
#include "Sender.h"


class Receiver : public Agent{
public:
	Receiver(const Setting &settings);
	virtual ~Receiver();
	void initTrial(double startingPos, double targetPos);
	void update(const double targetPos, Sender *sender, const int timeStep, const int targetID);
	double receivedSignal;
	void modularity();



private:
	void updateSignalSensor(Sender *sender);

	HelperFunction* _help;
	bool noMovement;
};

#endif /* RECEIVER_H_ */
