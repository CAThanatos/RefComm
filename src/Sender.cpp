/*
 * Sender.cpp
 *
 *  Created on: Oct 28, 2011
 *      Author: steffen
 */

#include "Sender.h"

Sender::Sender(const Setting &settings) {
	buildCTRNN(settings);
	signal = 0.0;
	targetSensor = 0.0;
	//for the two behavioral genes + RNN params, to be called after buildCTRNN()!
	noOfGenes=2+noOfRnnParams;

	constSpeed=false;
	constSignal=false;
	noSignal=false;
	hasSeenTarget = false;
	useBehavioralGenes = false;
	noMovement=false;
	_signalThreshold=0.1;
	placeSwitch = 0;
	settings.lookupValue("constSpeed", constSpeed);
	settings.lookupValue("constSignal", constSignal);
	settings.lookupValue("noSignal", noSignal);
	settings.lookupValue("useBehavioralGenes", useBehavioralGenes);
	settings.lookupValue("signalThreshold", _signalThreshold);
	settings.lookupValue("noMovement", noMovement);
	_help =HelperFunction::Instance();
	modular=false;
	settings.lookupValue("modular", modular);
    //cout << "ConstSpeed: " << constSpeed << " ConstSignal: " << constSignal << " NoSignal: " << noSignal << " Behavioral Genes: "<< useBehavioralGenes << endl;
}

Sender::~Sender() {

}

void Sender::initTrial(double startingPos, double targetPos)
{
	position = startingPos;
	currentSpeed = 0.0;
	signal = 0.0;
	targetSensor = 0.0;
	hasSeenTarget=false;
	hasLeftCommArea=false;
	updatePosition(targetPos);
	rnn->RandomizeCircuitState(0, 0);
	placeSwitch=0;
}

void Sender::modularity()
{
	if(!modular)
		return;

	for (int h=7; h<=8; h++)
	{
		//set synapsis to vel outputs to zero
		rnn->SetConnectionWeight(h, 9, 0.0);
		rnn->SetConnectionWeight(h, 10, 0.0);
		for (int h1=4;h1<=6;h1++)
		{
			//set all synapsis between hidden modules to zero
			rnn->SetConnectionWeight(h, h1, 0.0);
			rnn->SetConnectionWeight(h1, h, 0.0);
			//set synapsis to signal output to zero
			rnn->SetConnectionWeight(h1, 11, 0.0);
		}
	}
}

void Sender::update(const double targetPos, const int timeStep, bool receiverInCommArea, const int targetID)
{
  bool oldIsInCommArea = isInCommArea;
  //map position inputs
	double cosPosInput, sinPosInput;
	Agent::mapPos2Input(position, cosPosInput, sinPosInput);

	//feed inputs into neural network
	rnn->SetNeuronExternalInput(1, targetSensor);
	rnn->SetNeuronExternalInput(2, cosPosInput);
	rnn->SetNeuronExternalInput(3, sinPosInput);
	if(noOfInputs > 3)
	{
		rnn->SetNeuronExternalInput(4, isInCommArea);
	}

	//update neural network, stepSize should be around a factor of 10 smaller than the smallest time constant in the net
	rnn->EulerStep(0.1);

    //read output of neural network
	if(constSpeed)
		currentSpeed=CONST_SPEED;
	else
	{
		currentSpeed = (rnn->NeuronOutput(noOfInputs+noOfHidden+1)-rnn->NeuronOutput(noOfInputs+noOfHidden+2))*SPEED_SCALE;
	}
	if(velNoise>0)
		currentSpeed=GaussianRandom(currentSpeed, velNoise);
	if(motionStepNoise>0.0 && UniformRandom(0.0,1.0) <=motionStepNoise)
		currentSpeed = 0.0;


	if(_help->isUseAverage() == AVERAGE_SPEED)
	{
		currentSpeed=_help->getAvgSpeed(timeStep);
	}

	if(_help->isUseAverage() == AVERAGE_SIGNALONSET && placeSwitch>=2 && receiverInCommArea)
	{
		currentSpeed=0.0;
	}



	if(noMovement)
	{

		if(timeStep<10)
		{
			position=0.0;
			currentSpeed=0.0;
		}
		else if (timeStep < 20)
		{
			position=targetPos;
			currentSpeed=0.0;
		}
		else if (timeStep == 20)
		{
			position=0.0;
			currentSpeed=0.0;
		}
		//else start moving again
	}

	//update position + distance to target
	updatePosition(targetPos);
	updateTargetSensor();

	if(!noSignal)
	{
		double tmpOut = rnn->NeuronOutput(noOfInputs+noOfHidden+3);
		if(constSignal)
		{
			signal=1.0;
		}
		else
		{
			if(tmpOut > _signalThreshold)
				signal =  (tmpOut-_signalThreshold)/(1.0-_signalThreshold);
			if(signalNoise > 0.0)
				signal =GaussianRandom(signal, signalNoise);
		}

	}
	else
	{
		signal = 0.0;
	}

	if(_help->isUseAverage() == AVERAGE_SIGNAL || _help->isUseAverage() == AVERAGE_SIGNALONSET)
	{
		signal=_help->getAvgSig(timeStep);
	}



	if(oldIsInCommArea != isInCommArea)
	{
		++placeSwitch;
	}
}

void Sender::updateTargetSensor()
{
	//XXX this info might be updated only if the sender is within a range
	//from the target (which depends on the number of targets, see Simulation)
	if(dist2Target < HelperFunction::Instance()->getFitTargetRange())
	{
		//targetSensor = 1-dist2Target/(TARGENT_SENSOR_RANGE);
		targetSensor = 1.0;
		hasSeenTarget=true;
	}
	else
	{
		targetSensor = 0.0;
	}

}

void Sender::setBehavioralGenes(vector<float> &genes, int start)
{
	if(useBehavioralGenes)
	{
		if(genes[start] <1.0)
			constSpeed=false;
		else
			constSpeed=true;

		if(genes[start+1] <1.0)
			constSignal=false;
		else
			constSignal=true;
	}
//	cout << "G1: " << genes[start] << " cV: " << constSpeed
//	     << " G2: " << genes[start+1] << " cS: " << constSignal << endl;
}














