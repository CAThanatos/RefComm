/*
 * HelperFunction.cpp
 *
 *  Created on: Nov 4, 2011
 *      Author: steffen
 */

#include "HelperFunction.h"

HelperFunction* HelperFunction::_instance = NULL;

HelperFunction* HelperFunction::Instance(){
  if(_instance == NULL){
    _instance = new HelperFunction();
  }
  return _instance;
}

void HelperFunction::readSettings()
{
  Config cfg;
  try
  {
    cfg.readFile("config1.cfg");
  }
  catch(const FileIOException &fioex)
  {
    std::cerr << "I/O error while reading file." << std::endl;
  }
  catch(const ParseException &pex)
  {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                          << " - " << pex.getError() << std::endl;
  }

  const Setting& root = cfg.getRoot();
  root["simulation"].lookupValue("noOfTest", _noOfTests);
  root["simulation"].lookupValue("simTime", _simTime);
  root["simulation"].lookupValue("noOfTargets", _noOfTargets);
  root["simulation"].lookupValue("popSize", _popSize);
  root["recording"].lookupValue("logAllInfo", _logInfo);
  root["rnn"].lookupValue("isFNN", _ffn);
  root["simulation"].lookupValue("tournamentSelection", _tournamentSelection);
  root["simulation"].lookupValue("fitEvalPeriod", _fitEvalPeriod);
  root["simulation"].lookupValue("cutOneTargetFitness", _cutOneTargetFitness);
  root["rnn"].lookupValue("biasLimit", _biasRange);
  root["rnn"].lookupValue("weightLimit", _weightRange);
  root["rnn"].lookupValue("tauLoLimit", _tauLow);
  root["rnn"].lookupValue("tauHiLimit", _tauHi);
  root["simulation"].lookupValue("gaussianChange", _gaussianChange);
  root["simulation"].lookupValue("initialPosNoise", _initialPosNoise);
  if(_initialPosNoise > 0.0)
  {
  	_initialPosNoise*=PI/4.0;
  }

  velStepNoiseEnv=0.0;


  if(_noOfTargets==5)_fitTargetRange=PI/8.0;
  else if(_noOfTargets==10)	_fitTargetRange=PI/18.0;

}

bool HelperFunction::isAveraging() const
{
    return _averaging;
}

void HelperFunction::initAveraging()
{
	_avgSigs.clear();
	_avgSpeed.clear();

	_avgSigs.resize(_simTime, 0);
	_avgSpeed.resize(_simTime, 0);
	_signals.resize(_noOfTargets);
	_onsets.resize(_noOfTargets, 0);
	for(int i=0;i<_noOfTargets;++i)
	{
		_signals[i].resize(_simTime, 0);
	}
	_averaging=true;
	_avgSigOnset=0.0;
	_useAverage=0;
	_minSignalOnset=0;
}

void HelperFunction::openBehavioralDataFile(const char *fileName)
{
	_behaviorLogFile.open(fileName,ios::trunc);
}

void HelperFunction::closeBehavioralDataFile()
{
	_behaviorLogFile.close();
//	for(int i=0;i<_simTime;i++)
//	  cout << i << " " << _avgSigs[i] << " " << _avgSpeed[i] << endl;
}

void HelperFunction::recordSpeedSignal(const double speed, const double signal, const int timeStep)
{
	_avgSigs[timeStep]+=signal/_noOfTargets;
	_avgSpeed[timeStep]+=speed/_noOfTargets;
}

int HelperFunction::getOnset(const int targetID)
{
	return _onsets[targetID];
}
void HelperFunction::recordSignalOnset(const int timeStep, const int targetID)
{

	_onsets[targetID]=timeStep;
	//_avgSigOnset+=(double)timeStep/_noOfTargets;
	_avgSigOnset=max(_avgSigOnset, (double)timeStep);
	if(timeStep>0)
	{
		if(_minSignalOnset==0)
			_minSignalOnset=timeStep;
		else
			_minSignalOnset=min(_minSignalOnset, timeStep);
	}


}

void HelperFunction::recordSignal(const int timeStep, const int targetID, const double signal)
{
	_signals[targetID][timeStep]=signal;
}

void HelperFunction::rescaleReceivedSignalVectors()
{
	_signalsOld = _signals;

	if(_useAverage == AVERAGE_SIGNALLENGTH2)
	{
		for(int t=0;t<5;++t)
		{
			int diff = _onsets[t]-_minSignalOnset;
			if(diff>0)
			{
	//			cout << "erase tar:" << t << " steps: " << diff <<endl;
				_signals[t].erase(_signals[t].begin(), _signals[t].begin()+diff);
				for(int z=0;z<diff;++z)
				{
					_signals[t].push_back(0.0);
				}
			}
		}
	}
	else if(_useAverage == AVERAGE_SIGNALLENGTH4)
	{
		for(int t=0;t<5;++t)
		{
			if(_onsets[t] > 0.0 && _averageOnsetDiff > 0.0)
			{
				// if(t == 0)
				// {
				// 	for(int i = 46; i < 57; ++i)
				// 	{
				// 		_signals[t][i] = 0.0;
				// 	}
				// 	// for(int i = 61; i < 77; ++i)
				// 	// {
				// 	// 	_signals[t][i] = 0.99;
				// 	// }
				// 	// _signals[t][49] = 0.0;
				// 	// _signals[t][50] = 0.0;
				// 	// _signals[t][51] = 0.0;
				// 	// _signals[t][52] = 0.0;
				// 	// _signals[t][53] = 0.0;
				// 	// _signals[t][54] = 0.99;
				// }
				// _signals[t].erase(_signals[t].end() - ceil(_averageOnsetDiff), _signals[t].end());
				// for(int z=0;z<ceil(_averageOnsetDiff);++z)
				// {
				// 	_signals[t].insert(_signals[t].begin(), 0.0);
				// }
				_signals[t].erase(_signals[t].begin(), _signals[t].begin()+ceil(_averageOnsetDiff));
				for(int z=0;z<ceil(_averageOnsetDiff);++z)
				{
					_signals[t].push_back(0.0);
				}
			}
		}
	}

//	for(int t=0;t<5;++t)
//	{
//		int diff = _avgSigOnset-_onsets[t];
//		if(_onsets[t] > 0 && diff>0)
//		{
////			cout << "insert tar:" << t << " steps: " << diff <<endl;
//			for(int tmp=0;tmp<diff;++tmp)
//			{
//				_signals[t].insert(_signals[t].begin(), 0.0);
//			}
//		}
//	}

//	for(int t=0;t<5;++t)
//	{
//		int diff = _onsets[t]-_minSignalOnset;
//		if(diff>0)
//		{
////			cout << "erase tar:" << t << " steps: " << diff <<endl;
//			_signals[t].erase(_signals[t].begin()+(_minSignalOnset-1), _signals[t].begin()+(_onsets[t]-1));
//			for(int z=0;z<diff;++z)
//			{
//				_signals[t].push_back(0.0);
//			}
//		}
//	}


}
	
void HelperFunction::resetSignalVectors()
{
	_signals = _signalsOld;
}

void HelperFunction::computeAverageOnsetDiff()
{
	double averageOnsetDiff = 0.0;
	int nbOnsetDiffs = 0;
	int nbOnsets = 0;
	vector<int> tmpOnsets = _onsets;
	std::sort(tmpOnsets.begin(), tmpOnsets.end());
	int lastOnset = -1;

	for(int t = 0; t < 5; ++t)
		if(tmpOnsets[t] > 0.0)
		{
			if(lastOnset > 0)
			{
				averageOnsetDiff += tmpOnsets[t] - lastOnset;
				nbOnsetDiffs += 1;
			}

			lastOnset = tmpOnsets[t];
			nbOnsets += 1;
		}


	if(nbOnsetDiffs > 0)
	{
		_averageOnsetDiff = averageOnsetDiff/(double)nbOnsetDiffs;
	}
	// If there is only one onset, then we need _averageOnsetDiff to have a fixed value
	// TODO: would be better to get this value from the experiments
	else if(nbOnsets > 0)
	{
		_averageOnsetDiff = 10;
	}
	// _averageOnsetDiff = 10;
}

double HelperFunction::getSignalVector(const int targetID, const int timeStep)
{
	return _signals[targetID][timeStep];
}

HelperFunction::HelperFunction()
{
	_testing=false;
	_loading=false;
	_noOfTests=1;
	_simTime=120;
	_noOfTargets=5;
	_fitTargetRange=PI/8.0;
	_logInfo=false;
	_logInfoFullGeneration=false;
	_ffn=false;
	_averaging=false;
	_useAverage=0;
	_tournamentSelection=false;
	_fitEvalPeriod=10;
	_cutOneTargetFitness=false;
	_testParents = false;
	_weightRange=0;
	_biasRange=0;
	_tauLow=0.1;
	_tauHi=1.0;
	_gaussianChange=false;
	_initialPosNoise=0.0;

	//_avgSigOnset=0.0;
	//_minSignalOnset=0;
	//velStepNoiseEnv=0.0;
	//_popSize=0;
}

bool HelperFunction::isTesting() const
{
    return _testing;
}

void HelperFunction::setTesting(bool testing)
{
    _testing = testing;
}

bool HelperFunction::isLoading() const
{
    return _loading;
}

void HelperFunction::setLoading(bool loading)
{
    _loading = loading;
}

HelperFunction::~HelperFunction() {
}

int HelperFunction::getNoOfTests() const
{
    return _noOfTests;
}

int HelperFunction::getSimTime() const
{
    return _simTime;
}

int HelperFunction::getNoOfTargets() const
{
    return _noOfTargets;
}

double HelperFunction::getFitTargetRange() const
{
    return _fitTargetRange;
}


bool HelperFunction::isLogInfo() const
{
    return _logInfo;
}

bool HelperFunction::isLogInfoFullGeneration() const
{
    return _logInfoFullGeneration;
}

void HelperFunction::setLogInfoFullGeneration(bool logInfoFullGeneration)
{
    _logInfoFullGeneration = logInfoFullGeneration;
}

ofstream & HelperFunction::getDetailedInfoLogFile()
{
    return _detailedInfoLogFile;
}

void HelperFunction::openDetailedInfoLogFile(const char *fileName)
{
	_detailedInfoLogFile.open(fileName,ios::trunc);
}

void HelperFunction::closeDetailedInfoLogFile()
{
	_detailedInfoLogFile.close();
}

int HelperFunction::getPopSize() const
{
    return _popSize;
}

bool HelperFunction::isFfn() const
{
    return _ffn;
}

void HelperFunction::setAveraging(bool averaging)
{
    _averaging = averaging;
}

ofstream & HelperFunction::getBehaviorLogFile()
{
    return _behaviorLogFile;
}

int HelperFunction::isUseAverage() const
{
    return _useAverage;
}

void HelperFunction::setUseAverage(int useAverage)
{
    _useAverage = useAverage;
}

bool HelperFunction::isTournamentSelection() const
{
    return _tournamentSelection;
}

int HelperFunction::getFitEvalPeriod() const
{
    return _fitEvalPeriod;
}

bool HelperFunction::isCutOneTargetFitness() const
{
    return _cutOneTargetFitness;
}

bool HelperFunction::isTestParents() const
{
    return _testParents;
}

void HelperFunction::setTestParents(bool testParents)
{
    _testParents = testParents;
}

bool HelperFunction::isGaussianChange() const
{
    return _gaussianChange;
}

double HelperFunction::getAvgSigOnset() const
{
    return _avgSigOnset;
}

double HelperFunction::getInitialPosNoise() const
{
    return _initialPosNoise;
}

void HelperFunction::setInitialPosNoise(double initialPosNoise)
{
    _initialPosNoise = initialPosNoise;
}
