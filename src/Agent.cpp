/*
 * Agent.cpp
 *
 *  Created on: Nov 2, 2011
 *      Author: steffen
 */

#include "Agent.h"


Agent::Agent() {
	position=0.0;
	isInCommArea=true;
	velNoise=0.0;
	motionStepNoise=0.0;
	signalNoise=0.0;
	isOnTarget = false;
	hasLeftCommArea = false;
	modular = false;
	_help =HelperFunction::Instance();

	/*
	dist2Target=0.0;
	currentSpeed=0.0;
	noOfInputs=0;
	noOfOutputs=0;
	noOfHidden=0;
	noOfGenes=0;
	noOfRnnParams=0;
	*/
	//CTRNN *rnn;
}

Agent::~Agent() {
	delete rnn;
}

void Agent::setRNNvalues(vector<float> &genes, int start)
{
  hasLeftCommArea = false;
	delete rnn;
	rnn = new CTRNN(noOfInputs+noOfOutputs+noOfHidden);
	int counter=start;
	float tmp = 0.0;

//	cout << "start: " << counter << endl;
	int noOfNeurons = noOfInputs+noOfHidden+noOfOutputs;
	//bias values
	for(int i=1; i<=noOfNeurons;++i)
	{
		tmp=genes[counter];
		if(_help->isGaussianChange()) tmp*=2.0/5.0;
		rnn->SetNeuronBias(i, tmp);
//		cout << "B:" << rnn->NeuronBias(i)  << endl;
		++counter;
	}
//	cout << "after bias: " << counter << endl;
	/**
	 * this is for noHiddenLayerNetworks**/
	if(noOfHidden==0)
	{
		for(int from=1; from<=noOfNeurons; ++from)
		{
			//go through all hidden
			for (int to=1; to<=noOfNeurons; ++to)
			{
				tmp=genes[counter];
				if(_help->isGaussianChange()) tmp*=4.0/5.0;
				rnn->SetConnectionWeight(from, to, tmp);
				//    		cout << "S:" << rnn->ConnectionWeight(from, to) << endl;
				++counter;
			}
		}
	}
	else
	{
		//synapsis: input->hidden
		//go through all inputs
		for(int from=1; from<=noOfInputs; ++from)
		{
			//go through all hidden
			for (int to=noOfInputs+1; to<=noOfInputs+noOfHidden; ++to)
			{
				tmp=genes[counter];
				if(_help->isGaussianChange()) tmp*=4.0/5.0;
				rnn->SetConnectionWeight(from, to, tmp);
//				cout << "S:" << rnn->ConnectionWeight(from, to) << endl;
				++counter;
			}
		}

		if(!HelperFunction::Instance()->isFfn())
		{
			//synapsis: hidden->hidden
			//go through all hidden neurons
			for(int from=noOfInputs+1; from<=noOfInputs+noOfHidden; ++from)
			{
				//go through all hidden neurons
				for (int to=noOfInputs+1; to<=noOfInputs+noOfHidden; ++to)
				{
					tmp=genes[counter];
					if(_help->isGaussianChange()) tmp*=4.0/5.0;
					rnn->SetConnectionWeight(from, to, tmp);
					//				cout << "S:" << rnn->ConnectionWeight(from, to) << endl;
					++counter;
				}
			}
		}

		//synapsis: hidden->output
		//go through all hidden neurons
		for(int from=noOfInputs+1; from<=noOfInputs+noOfHidden; ++from)
		{
			//go through all output neurons
			for (int to=noOfInputs+noOfHidden+1; to<=noOfNeurons; ++to)
			{
				tmp=genes[counter];
				if(_help->isGaussianChange()) tmp*=4.0/5.0;
				rnn->SetConnectionWeight(from, to, tmp);
//				cout << "S:" << rnn->ConnectionWeight(from, to) << endl;
				++counter;
			}
		}
	}

//	cout << "after synapsis: " << counter << endl;
	//time constants
	for(int i=1; i<=noOfNeurons;++i)
	{
		tmp=genes[counter];
		if(_help->isGaussianChange())
		{
			tmp+=5.0;
			tmp/=10.0;
			tmp*=0.9;
			tmp+=0.1;
		}
		rnn->SetNeuronTimeConstant(i, tmp);
//		cout << "T:" << rnn->NeuronTimeConstant(i) << endl;
		++counter;
	}
//	cout << "end: " << counter << endl;
}

void Agent::buildCTRNN(const Setting & settings)
{
	settings.lookupValue("velNoise", velNoise);
	if(velNoise > 0)
	{
		//make it standard deviation
		velNoise =((velNoise*velNoise)*SPEED_SCALE)/2.0;
	}
	settings.lookupValue("motionStepNoise", motionStepNoise);

	settings.lookupValue("signalNoise", signalNoise);
	if(signalNoise > 0)
	{
		//make it standard deviation
		signalNoise *=signalNoise;
	}

	settings.lookupValue("inputNo", noOfInputs);
	settings.lookupValue("outputNo", noOfOutputs);
	settings.lookupValue("hiddenNo", noOfHidden);
	//

	rnn = new CTRNN(noOfInputs+noOfOutputs+noOfHidden);
	//for bias values + time constants
	noOfRnnParams=2*(noOfInputs+noOfHidden+noOfOutputs);
	//synaptic strenghts
	if(noOfHidden==0)
	{
		noOfRnnParams+=noOfInputs*noOfInputs+2*noOfInputs*noOfOutputs+noOfOutputs*noOfOutputs;
	}
	else
	{
		if(HelperFunction::Instance()->isFfn())
		{
			noOfRnnParams+=noOfInputs*noOfHidden+noOfHidden*noOfOutputs;
		}
		else
		{
			noOfRnnParams+=noOfInputs*noOfHidden+noOfHidden*noOfHidden+noOfHidden*noOfOutputs;
		}
	}
}

void Agent::updatePosition(const double targetPos)
{
  position+=currentSpeed;
  position=HelperFunction::normalizeRad(position);
  dist2Target=fabs(HelperFunction::getAngleDiff(position, targetPos));
  isInCommArea=Agent::inCommArea(position);
  if(dist2Target<=HelperFunction::Instance()->getFitTargetRange() )
    isOnTarget=true;
  else
    isOnTarget = false;

  if(!hasLeftCommArea && !isInCommArea)
    hasLeftCommArea=true;
}
