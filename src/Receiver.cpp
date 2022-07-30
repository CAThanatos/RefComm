/*
 * AgentController.cpp
 *
 *  Created on: Oct 28, 2011
 *      Author: steffen
 */

#include "Receiver.h"

Receiver::Receiver(const Setting &settings) {
	buildCTRNN(settings);
	receivedSignal=0.0;
	//to be called after buildCTRNN()
	noOfGenes=noOfRnnParams;
	modular=false;
	noMovement=false;
	settings.lookupValue("modular", modular);
	settings.lookupValue("noMovement", noMovement);
	_help =HelperFunction::Instance();
}

Receiver::~Receiver() {
}

void Receiver::initTrial(double startingPos, double targetPos)
{
	position = startingPos;
	hasLeftCommArea=false;
	currentSpeed = 0.0;
    receivedSignal = 0.0;
	updatePosition(targetPos);
	rnn->RandomizeCircuitState(0, 0);

}

void Receiver::modularity()
{
	if(!modular)
		return;

	//go from the 4th to the 5th hidden neuron and disable synapsis to first and second output (movement neurons)
	for (int h=7; h<=8; ++h)
	{
		//set synapsis from vel input to hidden 7-8 to zero
		rnn->SetConnectionWeight(2, h, 0.0);
		rnn->SetConnectionWeight(3, h, 0.0);
		for (int h1=4;h1<=6;++h1)
		{
			//set all synapsis between hidden modules to zero
			rnn->SetConnectionWeight(h, h1, 0.0);
			rnn->SetConnectionWeight(h1, h, 0.0);
			//set synapsis from signal input to hidden 4-6 to zero
			rnn->SetConnectionWeight(1, h1, 0.0);
		}
	}
}


void Receiver::update(const double targetPos, Sender *sender, const int timeStep, const int targetID)
{
	//map position inputs
	double cosPosInput, sinPosInput;
	Agent::mapPos2Input(position, cosPosInput, sinPosInput);

//  if(_help->isUseAverage() == AVERAGE_SIGNALONSET && sender->placeSwitch >= 2)
//	{
//  	double prevSignal=_help->getSignalVector(targetID)->at(timeStep-3);
//  	if (prevSignal>0.0)
//  	{
//  		receivedSignal =0.0;
//  	}
//	}

	//feed inputs into neural network
	rnn->SetNeuronExternalInput(1, receivedSignal);
	rnn->SetNeuronExternalInput(2, cosPosInput);
	rnn->SetNeuronExternalInput(3, sinPosInput);
	if(noOfInputs > 3)
	{
		if(sender->isInCommArea && isInCommArea)
			rnn->SetNeuronExternalInput(4, 1.0);
		else
			rnn->SetNeuronExternalInput(4, 0.0);
	}

	//mute signal if t is below the average signal onset
//	if(_help->isUseAverage() == AVERAGE_SIGNALONSET && sender->placeSwitch >= 2 && previousSignal > 0.0 )

	//update neural network, stepSize should be around a factor of 10 smaller than the smallest time constant in the net
    rnn->EulerStep(0.1);

    //read output of neural network
    currentSpeed = rnn->NeuronOutput(noOfInputs+noOfHidden+1)-rnn->NeuronOutput(noOfInputs+noOfHidden+2);
    currentSpeed *= SPEED_SCALE;
		if(velNoise>0)
			currentSpeed=GaussianRandom(currentSpeed, velNoise);
    if(motionStepNoise>0.0 && UniformRandom(0.0,1.0) <=motionStepNoise)
			currentSpeed = 0.0;

    if(noMovement)
    {
    	currentSpeed=0.0;
    	position=rnn->NeuronOutput(noOfInputs+noOfHidden+1)-rnn->NeuronOutput(noOfInputs+noOfHidden+2);
    	position*=PI;
    }

	//update position + distance to target
    updatePosition(targetPos);
    updateSignalSensor(sender);
  	if(_help->isUseAverage()==AVERAGE_SIGNALLENGTH2 || _help->isUseAverage()==AVERAGE_SIGNALLENGTH4)
  	{
//  		if(isInCommArea)
  			receivedSignal=_help->getSignalVector(targetID,timeStep);
//  		else
//  			receivedSignal=0.0;
  	}

}

void Receiver::updateSignalSensor(Sender *sender)
{
	if(isInCommArea && sender->isInCommArea)
	{
		receivedSignal = sender->signal;
	}
	else
	{
		receivedSignal = 0.0;
	}
	if(sender->constSignal && noOfInputs > 4)
	{
		receivedSignal = 0.0;
	}
}











