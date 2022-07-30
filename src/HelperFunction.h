/*
 * HelperFunction.h
 *
 *  Created on: Nov 4, 2011
 *      Author: steffen
 */

#ifndef HELPERFUNCTION_H_
#define HELPERFUNCTION_H_

#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <libconfig.h++>
using namespace std;
using namespace libconfig;

static const double PI=3.14159265358979323846;
static const double TWOPI = 2.0*PI;

#define AVERAGE_SPEED 1
#define AVERAGE_SIGNAL 2
#define AVERAGE_SIGNALONSET 3
#define AVERAGE_SIGNALLENGTH 4
#define AVERAGE_SIGNALLENGTH2 5
#define AVERAGE_SIGNALLENGTH3 6
#define AVERAGE_SIGNALLENGTH4 7

class HelperFunction {

public:
	static HelperFunction* Instance();
	virtual ~HelperFunction();
  bool isTesting() const;
  void setTesting(bool testing);
  bool isLoading() const;
  void setLoading(bool loading);
  void initAveraging();
  void recordSpeedSignal(const double speed, const double signal, const int timeStep);
  void recordSignalOnset(const int timeStep, const int targetID);
  int getOnset(const int targetID);
  void rescaleReceivedSignalVectors();
  void resetSignalVectors();
  void computeAverageOnsetDiff();


  void readSettings();
  int getNoOfTests() const;
  int getSimTime() const;
  int getNoOfTargets() const;
  double getFitTargetRange() const;
  bool isLogInfo() const;
  bool isLogInfoFullGeneration() const;
  void setLogInfoFullGeneration(bool logInfoFullGeneration);
  ofstream & getDetailedInfoLogFile();
  void openDetailedInfoLogFile(const char *fileName);
  void closeDetailedInfoLogFile();
  void openBehavioralDataFile(const char *fileName);
  void closeBehavioralDataFile();
  int getPopSize() const;
  bool isFfn() const;
  bool isAveraging() const;
  void setAveraging(bool averaging);
  ofstream & getBehaviorLogFile();
  int isUseAverage() const;
  void setUseAverage(int useAverage);
    bool isTournamentSelection() const;
    int getFitEvalPeriod() const;
    bool isCutOneTargetFitness() const;
    bool isTestParents() const;
    void setTestParents(bool testParents);
    bool isGaussianChange() const;
    double getAvgSigOnset() const;
    double getInitialPosNoise() const;
    void setInitialPosNoise(double initialPosNoise);
  const double getAvgSig(int timeStep) { return _avgSigs[timeStep];}
  const double getAvgSpeed(int timeStep) { return _avgSpeed[timeStep];}
  void recordSignal(const int timeStep, const int targetID, const double signal);
  double getSignalVector(const int targetID, const int timeStep);
  double getAverageOnsetDiff() { return _averageOnsetDiff; }

  double velStepNoiseEnv;

	static double normalizeRad(double x)
	{
		if (x > PI)
			return(fmod(x+PI, TWOPI) - PI);
		else if (x < -PI)
			return(fmod(x-PI, -TWOPI) + PI);
		else
			return x;
	}

	static double getAngleDiff(double a, double b)
	{
		return(HelperFunction::normalizeRad(a-b));
	}

	static double getMean(vector<double> &vals)
	{
		double s=0;
		vector<double>::iterator it;
		for(it=vals.begin();it<vals.end();it++)
		{
			s+=(*it);
		}
		return s/vals.size();
	}

	static double getMean(vector<bool> &vals)
	{
		double s=0;
		vector<bool>::iterator it;
		for(it=vals.begin();it<vals.end();it++)
		{
			s+=(*it);
		}
		return s/vals.size();
	}

	static double getSD(vector<double> &vals, const double &mean)
	{
		double to = 0;

		for(vector<double>::iterator it = vals.begin();it<vals.end();it++)
			to += pow(mean - *it,2);

		to/=vals.size()-1;
		return sqrt(to);
	}

	static double getSD(vector<bool> &vals, const double &mean)
	{
		double to = 0;

		for(vector<bool>::iterator it = vals.begin();it<vals.end();it++)
			to += pow(mean - *it,2);

		to/=vals.size()-1;
		return sqrt(to);
	}

	static double getMin(vector<double> &vals)
	{
	  return *min_element(vals.begin(),vals.end());
	}

  static double getMax(vector<double> &vals)
  {
    return *max_element(vals.begin(),vals.end());
  }

	static double round2( double x )
	{
	  const double sd = 1000; //for accuracy to 3 decimal places
	  //const double sd = 10; //for accuracy to 2 decimal places
	  return int(x*sd + (x<0? -0.5 : 0.5))/sd;
	}

  static void readGenesFromFile(char *str, int testIndividual, vector<float> &genes)
  {
  	ifstream ifs(str);
    string line;
    int curInd = 0;
    do
    {
      getline( ifs, line );
      if(curInd == testIndividual)
      {
        istringstream ss( line );
        while(ss)
        {
          float f;
          ss >> f;
          genes.push_back(f);
//          cout << f << " ";
        }
      }
      curInd++;
    }while(curInd <= testIndividual);
    genes.erase(genes.end()-1);
//    cout << endl;
  }

  static void readGenesFromFile(bool isSender, int testGeneration, int testIndividual, vector<float> &genes)
  {
    char strOut[100];
    if(!HelperFunction::Instance()->isTestParents())
    {
    	if(isSender)
    		sprintf( strOut, "%d.sender.pop", testGeneration );
    	else
    		sprintf( strOut, "%d.receiver.pop", testGeneration );
    }
    else
    {

    	if(isSender)
    		sprintf( strOut, "%d.sender.parents.pop", testGeneration );
    	else
    		sprintf( strOut, "%d.receiver.parents.pop", testGeneration );
    }

    ifstream ifs(strOut);
    string line;
    int curInd = 0;
    do
    {
      getline( ifs, line );
      if(curInd == testIndividual)
      {
        istringstream ss( line );
        while(ss)
        {
          float f;
          ss >> f;
          genes.push_back(f);
//          cout << f << " ";
        }
      }
      curInd++;
    }while(curInd <= testIndividual);
    ifs.close();
    genes.erase(genes.end()-1);
//    cout << endl;
  }

protected:
	HelperFunction();

private:
	static HelperFunction* _instance;
	bool _testing;
  bool _loading;
	int _noOfTests;
	int _simTime;
	int _noOfTargets;
	double _fitTargetRange;
	bool _logInfo;
	bool _logInfoFullGeneration;
	bool _ffn;
	bool _tournamentSelection;
	bool _cutOneTargetFitness;
	int _fitEvalPeriod;
	ofstream _detailedInfoLogFile;
	ofstream _behaviorLogFile;
	int _popSize;
	int _averaging;
	int _useAverage;
  vector<double> _avgSigs;
  vector<int> _onsets;
  vector<vector <double> > _signals;
  vector<vector <double> > _signalsOld;
  vector<double> _avgSpeed;
  double _avgSigOnset;
  bool _testParents;
  double _weightRange;
  double _biasRange;
  double _tauLow;
  double _tauHi;
  bool _gaussianChange;
  double _initialPosNoise;
  int _minSignalOnset;
  double _averageOnsetDiff;


};

#endif /* HELPERFUNCTION_H_ */
