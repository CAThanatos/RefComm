/*
 * Agent.h
 *
 *  Created on: Nov 2, 2011
 *      Author: steffen
 */

#ifndef AGENT_H_
#define AGENT_H_

#include "CTRNN.h"
#include <iostream>
#include <fstream>
#include <libconfig.h++>
#include <vector>
#include <math.h>
#include "HelperFunction.h"
using namespace libconfig;
using namespace std;


#define MIN_COMM_AREA (-PI/4.0)
#define MAX_COMM_AREA (PI/4.0)
//corresponds to the length of one target if there are 10
#define SPEED_SCALE (PI/9)
#define CONST_SPEED (PI/18)
//#define SPEED_SCALE (PI/18)
//#define CONST_SPEED (PI/36)

//18 for ten, 8 for five, 4 for three


class Agent {
public:
	static void mapPos2Input(double &pos, double & cosPosInput, double & sinPosInput)
	{
		// XXX (GI) replaced sincos with __sincos
		// XXX (AB) y tho
		sincos(pos,&cosPosInput, &sinPosInput);
		cosPosInput += 1.0;
		cosPosInput /= 2.0;
		sinPosInput += 1.0;
		sinPosInput /= 2.0;
	}

	static bool inCommArea(const double &position)
	{
		if(position < MIN_COMM_AREA || position > MAX_COMM_AREA )
			return false;
		else
			return true;
	}

	void updatePosition(const double targetPos);


	Agent();
	virtual ~Agent();

	void setRNNvalues(vector<float> &genes, int start);


	double position;
	bool isInCommArea;
	bool isOnTarget;
	bool hasLeftCommArea;
	double dist2Target;
	double currentSpeed;
	CTRNN *rnn;
	int noOfInputs, noOfOutputs, noOfHidden, noOfGenes, noOfRnnParams;
	double velNoise;
	double motionStepNoise;
	double signalNoise;
	HelperFunction* _help;
	bool modular;



	void buildCTRNN(const Setting &settings);



private:

};

#endif /* AGENT_H_ */
