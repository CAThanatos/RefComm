/*
 * Simulation.cpp
 *
 *  Created on: Oct 31, 2011
 *      Author: steffen
 */

#include "Simulation.h"

Simulation::Simulation(const Setting& root) {
	_receiver =new Receiver(root["receiver"]);
	_sender = new Sender(root["sender"]);
	_help =HelperFunction::Instance();
	//cout << "GenesRec: " << _receiver->noOfGenes << " GenesSend: " << _sender->noOfGenes << endl;
	for (int i=0; i < _help->getNoOfTargets();++i){
		_targetPositions.push_back(PI/2.0+PI/(_help->getNoOfTargets()-1)*i);
		_targetPositions[i]=HelperFunction::normalizeRad(_targetPositions[i]);
	}

	_startingPositions[0]=PI/8.0;
	_startingPositions[1]=0.0;
	_startingPositions[2]=-_startingPositions[0];
	_MAX_INFO=(1.0/_help->getNoOfTargets())*log(1.0/_help->getNoOfTargets());
	_MAX_INFO*=_help->getNoOfTargets();
	_MAX_INFO*=-1;

	_totalInformation=0.0;
	_totalInformationSender=0.0;

	_aSignalAmplitude=new SimpleLog();
  _aSignalAmplitudeSender = new SimpleLog();

  _aSignalDuration=new SimpleLog();
  _aSignalDurationSender = new SimpleLog();

  _timeInEnvSender=new SimpleLog();
  _returnTimeSender=new SimpleLog();
  _velSender = new SimpleLog();
  _velSenderInE = new SimpleLog();
  _velSenderInComm = new SimpleLog();

  _allLogs.push_back(_aSignalAmplitude);
  _allLogs.push_back(_aSignalAmplitudeSender);
  _allLogs.push_back(_aSignalDuration);
  _allLogs.push_back(_aSignalDurationSender);
  _allLogs.push_back(_timeInEnvSender);
  _allLogs.push_back(_returnTimeSender);
  _allLogs.push_back(_velSender);
  _allLogs.push_back(_velSenderInE);
  _allLogs.push_back(_velSenderInComm);

  //this vector contains all the info channels that are considered in the total information calculation
  _allLogs2total.push_back(_aSignalDuration);
  _allLogs2total.push_back(_aSignalAmplitude);

  _allLogs2totalSender.push_back(_aSignalDurationSender);
  _allLogs2totalSender.push_back(_aSignalAmplitudeSender);
  _allLogs2totalSender.push_back(_velSenderInE);
  _allLogs2totalSender.push_back(_velSenderInComm);

  activateReceiver = false;
  root["simulation"].lookupValue("activateReceiver", activateReceiver);
}

Simulation::~Simulation() {

	delete _receiver;
	delete _sender;
	for(vector<SimpleLog*>::iterator it=_allLogs.begin();it<_allLogs.end();it++)
	{
	  delete *it;
	}
}



void Simulation::updateLoggingInfoEndTrial(int target)
{
  if(_help->isLogInfo())
  {
    if(_aSignalDuration->tmp > 0)
    {
      _aSignalAmplitude->values[target] = HelperFunction::round2(_aSignalAmplitude->tmp/_aSignalDuration->tmp);
    }

    if(_aSignalDurationSender->tmp > 0)
    {
      _aSignalAmplitudeSender->values[target] = HelperFunction::round2(_aSignalAmplitudeSender->tmp/_aSignalDurationSender->tmp);
    }

    _aSignalDuration->values[target] = _aSignalDuration->tmp;
    _aSignalDurationSender->values[target] = _aSignalDurationSender->tmp;

    //calculate the relative duration of the signal with respect to the time when communication would have been possible
//    _aSignalRelDuration->values[target] = _aSignalRelDuration->tmp - _aSignalDuration->tmp;
//    _aSignalRelDurationSender->values[target] = _aSignalRelDurationSender->tmp - _aSignalDurationSender->tmp;

    _timeInEnvSender->values[target] = _timeInEnvSender->tmp;
    _returnTimeSender->values[target] = _returnTimeSender->tmp;

    _velSender->values[target] = HelperFunction::round2(_velSender->tmp/_help->getSimTime());
    _velSenderInE->values[target] = HelperFunction::round2(_velSenderInE->tmp/_timeInEnvSender->values[target]);
    _velSenderInComm->values[target] = HelperFunction::round2(_velSenderInComm->tmp/(_help->getSimTime()-_timeInEnvSender->values[target]));
  }

}

void Simulation::updateLoggingInfoStep(int step)
{
	if(_help->isLogInfo())
	{
		if(_sender->signal>0)
		{
			_aSignalDurationSender->tmp++;
			_aSignalAmplitudeSender->tmp+=_sender->signal;
		}

		if(_receiver->receivedSignal>0)
		{
			_aSignalDuration->tmp++;
			_aSignalAmplitude->tmp+=_receiver->receivedSignal;
		}

		if(!_sender->isInCommArea)
		{
			_timeInEnvSender->tmp++;
			_velSenderInE->tmp+=_sender->currentSpeed/(PI/9);
		}
		else
		{
			_velSenderInComm->tmp+=_sender->currentSpeed/(PI/9);
		}

		if(_returnTimeSender->tmp == 0 && _sender->placeSwitch == 2)
		{
			_returnTimeSender->tmp = step;
		}
		_velSender->tmp+=_sender->currentSpeed/(PI/9);

	}
}

void Simulation::initLoggingTrial()
{
  if(_help->isLogInfo())
  {
    for(unsigned int i=0;i<_allLogs.size();++i)
    {
      _allLogs[i]->tmp=0;
      _allLogs[i]->probs.clear();
    }
  }
}

void Simulation::finalizeLogging(PopulationRecoder *& popRecorder, const float & retFitness)
{

  if(_help->isLogInfo())
	{
//    for(unsigned int t=0;t<NO_OF_TARGETS;++t)
//    {
//      for(unsigned int l=0;l<_allLogs.size();++l)
//        cout << _allLogs[l]->values[t] << " ";
//      cout << endl;
//    }
//    cout << endl;


    for(unsigned int l=0;l<_allLogs.size();++l)
    {
      SimpleLog *tmpLog =_allLogs[l];

      //calculate probabilities of single channels;
      for(int i=0;i<_help->getNoOfTargets();++i)
      {
        if(tmpLog->probs.find(tmpLog->values[i]) == tmpLog->probs.end())
          tmpLog->probs[tmpLog->values[i]] = 1;
        else
          tmpLog->probs[tmpLog->values[i]] += 1;
      }
      //calculate inidividual channels' information
      tmpLog->information=0;
      for(map<double, int>::iterator it=tmpLog->probs.begin();it!=tmpLog->probs.end();it++)
      {
        double prob=(double)(it->second)/_help->getNoOfTargets();
        tmpLog->information+=prob*log(prob);
      }
      tmpLog->information*=-1;
      tmpLog->information/=_MAX_INFO;
//      cout << tmpLog->information << " ";

    }
//    cout << endl;

    //calculate toalInformation Receiver
    map<int, int> probs;
    for(int t=0;t<_help->getNoOfTargets();++t)
    {
      probs[t]=1;
    }

    for(int currTar=0;currTar<_help->getNoOfTargets();currTar++)
    {
      //only if it hasn't been removed already
      if(!(probs.find(currTar) == probs.end()))
      {
        //checks against all the remaining
        for(int compTar=currTar+1; compTar < _help->getNoOfTargets(); ++compTar)
        {
          //only if it hasn't been removed already
          if(!(probs.find(compTar) == probs.end()))
          {
            bool isIdentical=true;
            for(unsigned int c=0;c<_allLogs2total.size();++c)
            {
              if(_allLogs2total[c]->values[currTar] != _allLogs2total[c]->values[compTar])
                isIdentical=false;
            }
            //if the values in all channels for two targets are identical, remove one and increase the prob counter
            if(isIdentical)
            {
              probs[currTar]+=1;
              probs.erase(compTar);
            }
          }
        }
      }
    }

    _totalInformation = 0;
    for(map<int, int>::iterator it=probs.begin();it!=probs.end();it++)
    {
      double prob=(double)(it->second)/_help->getNoOfTargets();
      _totalInformation+=prob*log(prob);
    }
    _totalInformation*=-1;
    _totalInformation/=_MAX_INFO;
//    cout << endl << "total: " << _totalInformation << endl;


    //calculate toalInformation Sender
    probs.clear();
    for(int t=0;t<_help->getNoOfTargets();++t)
    {
      probs[t]=1;
    }

    for(int currTar=0;currTar<_help->getNoOfTargets();currTar++)
    {
      //only if it hasn't been removed already
      if(!(probs.find(currTar) == probs.end()))
      {
        //checks against all the remaining
        for(int compTar=currTar+1; compTar < _help->getNoOfTargets(); ++compTar)
        {
          //only if it hasn't been removed already
          if(!(probs.find(compTar) == probs.end()))
          {
            bool isIdentical=true;
            for(unsigned int c=0;c<_allLogs2totalSender.size();++c)
            {
              if(_allLogs2totalSender[c]->values[currTar] != _allLogs2totalSender[c]->values[compTar])
                isIdentical=false;
            }
            //if the values in all channels for two targets are identical, remove one and increase the prob counter
            if(isIdentical)
            {
              probs[currTar]+=1;
              probs.erase(compTar);
            }
          }
        }
      }
    }

    _totalInformationSender = 0;
    for(map<int, int>::iterator it=probs.begin();it!=probs.end();it++)
    {
      double prob=(double)(it->second)/_help->getNoOfTargets();
      _totalInformationSender+=prob*log(prob);
    }
    _totalInformationSender*=-1;
    _totalInformationSender/=_MAX_INFO;
//    cout << endl << "totalSender: " << _totalInformationSender << endl;

		if(_help->isLogInfoFullGeneration())
		{
			_help->getDetailedInfoLogFile() << retFitness << " " << _totalInformation
					<< " " << _totalInformationSender;
			for(vector<SimpleLog*>::iterator it=_allLogs.begin();it<_allLogs.end();it++)
			{
				_help->getDetailedInfoLogFile() << " " << (*it)->information;
			}
			_help->getDetailedInfoLogFile()	<< endl;

			if(_help->isTesting())
			{
				for(int i =0;i<_help->getNoOfTargets();++i)
				{
					for(vector<SimpleLog*>::iterator it=_allLogs.begin();it<_allLogs.end();it++)
					{
						_help->getDetailedInfoLogFile() << (*it)->values[i] << " ";
					}
					_help->getDetailedInfoLogFile()	<< endl;
				}

			}
		}

		if(popRecorder != NULL)
		{
		  popRecorder->setIndInfo(_totalInformation, _totalInformationSender, _allLogs);
		}
	}
}

double Simulation::evaluate(vector<float> &genes, int generation, PopulationRecoder *popRecorder, vector<double> &retVals)
{
  int startFitnessCounting=_help->getSimTime()-_help->getFitEvalPeriod();

  int fitness =0 ;
  double retFitness = 0.0;
  double commTime=0.0;
  int signalOnset=0;

  double averageOnsetDiff = 0.0;

  vector<int> testTargets(_help->getNoOfTargets());
  vector<vector <double> > senderSigs(_help->getNoOfTargets(), vector<double>(_help->getSimTime(),0.0));
  vector<vector <double> > senderVels(_help->getNoOfTargets(), vector<double>(_help->getSimTime(),0.0));
  for(int i=0; i < _help->getNoOfTargets(); ++i)
  {
    testTargets[i] = i;
  }

  vector<int> vecOnsets(testTargets.size(), -1);
  vector<int> vecTargetsFound;
  vector<int> vecTargetsDiscriminated;

  vector<int>::iterator it;
  if(_help->isUseAverage()==AVERAGE_SIGNALLENGTH2)
  {
  	_help->rescaleReceivedSignalVectors();
  }
  if(_help->isUseAverage()==AVERAGE_SIGNALLENGTH3)
  {
    _help->computeAverageOnsetDiff();
    averageOnsetDiff = _help->getAverageOnsetDiff();
  }
  if(_help->isUseAverage()==AVERAGE_SIGNALLENGTH4)
  {
    _help->computeAverageOnsetDiff();
    averageOnsetDiff = _help->getAverageOnsetDiff();
    _help->rescaleReceivedSignalVectors();
  }
  for(it=testTargets.begin();it<testTargets.end();it++)
  {
  	fitness=0;
  	signalOnset=0;
 		//initLoggingTrial();

    //set new ctrnn
    _sender->setBehavioralGenes(genes, 0);
    _sender->modularity();
    _sender->setRNNvalues(genes, 2);
    //    _sender->modularity();
    _receiver->setRNNvalues(genes, _sender->noOfGenes);
    _receiver->modularity();

//    _receiver->modularity();
    bool startReceiver=false;

    bool targetFound = false;


//    _sender->initTrial(0.0, _targetPositions[*it]);
//    _receiver->initTrial(0.0, _targetPositions[*it]);
    double senderInitPos=UniformRandom(-(_help->getInitialPosNoise()), _help->getInitialPosNoise());
    _sender->initTrial(senderInitPos, _targetPositions[*it]);
    double receiverInitPos=UniformRandom(-(_help->getInitialPosNoise()), _help->getInitialPosNoise());
    _receiver->initTrial(receiverInitPos, _targetPositions[*it]);


    for (int t = 0; t<_help->getSimTime(); ++t)
    {
    	if(_help->isUseAverage()==AVERAGE_SIGNALLENGTH && _help->getOnset(*it) > 0
    			&& t >= _help->getOnset(*it) && t<_help->getAvgSigOnset())
    	{
//    		cout << "orig onset: " << _help->getOnset(*it) << " max: " << _help->getAvgSigOnset()<< endl;
//    		cout << "wait, ta: " << *it << " time: " << t <<endl;;
    	}
      else if(_help->isUseAverage()==AVERAGE_SIGNALLENGTH3 && _help->getOnset(*it) > 0
          && t >= _help->getOnset(*it) && t < (_help->getOnset(*it) + averageOnsetDiff))
      {
         // cout << "orig onset: " << _help->getOnset(*it) << " avg: " << averageOnsetDiff << endl;
         // cout << "wait, ta: " << *it << " time: " << t <<endl;
      }
    	else
    	{
    		_sender->update(_targetPositions[*it], t, _receiver->isInCommArea, *it);
    	}
    		senderSigs[*it][t]=_sender->signal;
    		//first normalize between -1 and 1 and then normalize between 0 and 1 (so it is in the same range as the signal)
    		senderVels[*it][t]=(1.0+(_sender->currentSpeed/SPEED_SCALE))/2.0;

    		if(activateReceiver)
    		{
    			if(!startReceiver && _sender->placeSwitch == 2)
    				startReceiver=true;
    			if(startReceiver)
    				_receiver->update(_targetPositions[*it], _sender, t, *it);
    		}
    		else
    		{
    			_receiver->update(_targetPositions[*it], _sender, t, *it);
    		}


    		if(_help->isAveraging())
    		{
    			_help->recordSpeedSignal(_sender->currentSpeed, _sender->signal, t);
    			//save received signal
    			_help->recordSignal(t, *it, _receiver->receivedSignal);
    			//save potentially available signal
//    			if(_sender->isInCommArea)
//    			{
//    				_help->recordSignal(t, *it, _sender->signal);
//    			}
//    			else
//    			{
//    				_help->recordSignal(t, *it, 0.0);
//    			}

    			//      	if(signalOnset==0 && _sender->placeSwitch >= 2 && _receiver->receivedSignal > 0.0)
      			// if(signalOnset==0 && _sender->placeSwitch >= 2 && _receiver->receivedSignal > 0.0)
            if(signalOnset==0 && _sender->placeSwitch >= 2) // AB: another way to test onset (so that onset is taken into account)
                                                            // even when sender enters in the CA before the receiver
    			{
//    				cout << "target: " << *it << " onset: " << t << endl;
    				signalOnset=t;
    				_help->recordSignalOnset(t, *it);
    			}
    		}

        if(!_help->isTesting())
        {
          if(popRecorder->recordInfo)
          {
            if(signalOnset == 0 && _sender->placeSwitch >= 2)
            {
              signalOnset = t;
              vecOnsets[*it] = t;
            }
          }
        }

      if(t>=startFitnessCounting && _receiver->dist2Target <= _help->getFitTargetRange())
      {
        fitness++;
        if(!targetFound)
        {
          targetFound = true;
          vecTargetsFound.push_back(*it);
        }
        //retFitness+=1.0-_receiver->dist2Target/FIT_TARGET_RANGE;
      }

      if(_sender->isInCommArea && _receiver->isInCommArea )
      	commTime++;

      //updateLoggingInfoStep(t);

    	if(_help->isTesting())
      {
        _help->getBehaviorLogFile()
        		<< t << " " //1
        		<< _targetPositions[*it] << " "//2
            << 0 << " " //3
            << 0 << " "//4
            << (double)fitness/(_help->getFitEvalPeriod()*_help->getNoOfTargets()) << " "//5
            << _sender->position << " " //6
            << _receiver->position << " "//7
            << _sender->targetSensor << " " //8
            << _sender->signal << " "//9
            << _receiver->receivedSignal << " "//10
            << _sender->currentSpeed << " "//11
            << _receiver->currentSpeed << " "//12
            << _receiver->isOnTarget << " "//13
            << _receiver->dist2Target << " "//14
            << endl;
      }
    }

    retFitness += fitness;

    if(fitness >= 15)
      vecTargetsDiscriminated.push_back(*it);
//    int signalLength = 0;
//    for(int tmp=0;tmp<100;tmp++)
//    	if(_help->getOnset(*it) > 0 && tmp>= _help->getOnset(*it) && _help->getSignalVector(*it, tmp) > 0)
//    		signalLength++;
//    cout << *it << " " << _help->getOnset(*it) << " " << double(fitness)/20.0 << " " << signalLength << endl;
		//updateLoggingInfoEndTrial(*it);
  }

  if(_help->isUseAverage() == AVERAGE_SIGNALLENGTH2 || _help->isUseAverage() == AVERAGE_SIGNALLENGTH4)
    _help->resetSignalVectors();

  retFitness = (double)retFitness/(_help->getFitEvalPeriod()*_help->getNoOfTargets());

  double d1=calcDistance(senderSigs);
  double d2=calcDistance(senderVels);
  retVals[0]=retFitness;
  retVals[1]=d1;
  retVals[2]=d2;
  retVals[3]=commTime/(_help->getSimTime()*_help->getNoOfTargets());

  //finalizeLogging(popRecorder, retFitness);
  if(!_help->isTesting())
  {
  	//popRecorder->setIndInfoGenes(_sender->constSignal, _sender->constSpeed);
  	popRecorder->setIndInfo2(d1, d2, retVals[3]);
    popRecorder->setOnset(vecOnsets);
    popRecorder->setTargetsFound(vecTargetsFound);
    popRecorder->setTargetsDiscriminated(vecTargetsDiscriminated);
  }

	return retFitness;
}

double Simulation::calcDistance(vector<vector<double> > &data)
{
	int noOfTests=data.size();
	int noOfTimeSteps=data[0].size();
	double sumDistance=0.0;
	for(int d1=0;d1<noOfTests;++d1)
	{
		for(int d2=d1+1;d2<noOfTests;++d2)
		{
			double tmp=0;
			for(int t=0;t<noOfTimeSteps;++t)
			{
				tmp+=pow( (data[d1][t]-data[d2][t]),2);
			}
			sumDistance+=(sqrt(tmp))/noOfTimeSteps;
		}
	}
	return sumDistance;
}



