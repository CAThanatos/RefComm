/*
 * PopulationRecoder.cpp
 *
 *  Created on: Nov 18, 2011
 *      Author: steffen
 */

#include "PopulationRecoder.h"

PopulationRecoder::PopulationRecoder(bool recordAllInfo) {
	_help = HelperFunction::Instance();
	recordInfo = recordAllInfo;
	logFile.open ("populationRecord.txt",ios::trunc);
	if(recordInfo)
	{
		logFile << "#1:gen 2+3:sigGene 4+5:velGene 6+7:totalInfo 8+9: totalInfoS";
		logFile << "10+11:sigAmp 12+13:sigAmpS 14+15:sigDur 16+17:sigDurS";
		logFile << "18+19:tInE 20+21:retT 22+23:velS 24+25:velSinE 26+27:velSinComm" << endl;
	}
	else
	{
		logFile << "#1:gen 2:sigGene 3:sd 4:velGene 5:sd" << endl;
	}
	int noOfChannels=2+9;
	channels.resize(noOfChannels, vector<double>());
	_diversityData.resize(3, vector<double>());
	_diversityLogFile.open("diversity.txt");
	_corrOnsetLengthLogFile.open("corrOnsetLength.txt");
	_targetsFoundLogFile.open("targetsFound.txt");
	_targetsDiscriminatedLogFile.open("targetsDiscriminated.txt");
}

PopulationRecoder::~PopulationRecoder() {
	logFile.close();
	_diversityLogFile.close();
	_corrOnsetLengthLogFile.close();
	_targetsFoundLogFile.close();
	_targetsDiscriminatedLogFile.close();
}

void PopulationRecoder::clearRecords()
{
	for(unsigned int i=0;i<channels.size();++i)
	{
	  channels[i].clear();
	}
	signalGene.clear();
	speedGene.clear();
	for(unsigned int i=0;i<_diversityData.size();++i)
	{
	  _diversityData[i].clear();
	}
	vecOnsets.clear();
	vecTargetsFound.clear();
	vecTargetsDiscriminated.clear();
}

void PopulationRecoder::setIndInfo(double all, double allSender, vector<SimpleLog*> &allLogs)
{
   if(recordInfo)
   {
  	 channels[0].push_back(all);
  	 channels[1].push_back(allSender);
  		for(unsigned int i=0;i<allLogs.size();++i)
  		{
  		  channels[i+2].push_back(allLogs[i]->information);
  		}
   }
}

void PopulationRecoder::setIndInfo2(double sigDiv, double velDiv, double commTime)
{
  if(recordInfo)
  {
  	_diversityData[0].push_back(sigDiv);
  	_diversityData[1].push_back(velDiv);
  	_diversityData[2].push_back(commTime);
  }
}

void PopulationRecoder::setIndInfoGenes(bool sigGene, bool velGene)
{
	   signalGene.push_back(sigGene);
	   speedGene.push_back(velGene);
}

void PopulationRecoder::setOnset(vector<int>& onsets)
{
	vecOnsets.push_back(onsets);
}

void PopulationRecoder::setTargetsFound(vector<int>& targetsFound)
{
	vecTargetsFound.push_back(targetsFound);
}

void PopulationRecoder::setTargetsDiscriminated(vector<int>& targetsDiscriminated)
{
	vecTargetsDiscriminated.push_back(targetsDiscriminated);
}




void PopulationRecoder::writeData(int gen)
{
	double mean, sd;
	logFile << gen;
	mean = HelperFunction::getMean(signalGene);
	sd = HelperFunction::getSD(signalGene, mean);
	logFile << " " << mean << " " << sd;

	mean = HelperFunction::getMean(speedGene);
	sd = HelperFunction::getSD(speedGene, mean);
	logFile << " " << mean << " " << sd;

	if(recordInfo)
	{
		for(unsigned int i=0;i<channels.size();++i)
		{
			mean = HelperFunction::getMean(channels[i]);
			sd = HelperFunction::getSD(channels[i], mean);
			logFile << " " << mean << " " << sd;
		}
	}
	logFile << endl;
	logFile.flush();
}

void PopulationRecoder::writeData2(int gen)
{
	if(recordInfo)
	{
		double mean, sd;
		_diversityLogFile << gen;

		for(unsigned int i=0;i<_diversityData.size();++i)
		{
			mean = HelperFunction::getMean(_diversityData[i]);
			sd = HelperFunction::getSD(_diversityData[i], mean);
			_diversityLogFile << " " << mean << " " << sd;
		}
		_diversityLogFile << endl;
		_diversityLogFile.flush();

		_corrOnsetLengthLogFile << gen;
		for(unsigned int i = 0; i < vecOnsets.size(); ++i)
		{
			_corrOnsetLengthLogFile << " [";
			for(unsigned int j = 0; j < vecOnsets[i].size(); ++j)
			{
				_corrOnsetLengthLogFile << vecOnsets[i][j];

				if(j < (vecOnsets[i].size() - 1))
					_corrOnsetLengthLogFile << ",";
			}
			_corrOnsetLengthLogFile << "]";
		}
		_corrOnsetLengthLogFile << endl;
		_corrOnsetLengthLogFile.flush();

		_targetsFoundLogFile << gen;
		for(unsigned int i = 0; i < vecTargetsFound.size(); ++i)
		{
			_targetsFoundLogFile << " [";
			for(unsigned int j = 0; j < vecTargetsFound[i].size(); ++j)
			{
				_targetsFoundLogFile << vecTargetsFound[i][j];

				if(j < (vecTargetsFound[i].size() - 1))
					_targetsFoundLogFile << ",";
			}
			_targetsFoundLogFile << "]";
		}
		_targetsFoundLogFile << endl;
		_targetsFoundLogFile.flush();

		_targetsDiscriminatedLogFile << gen;
		for(unsigned int i = 0; i < vecTargetsDiscriminated.size(); ++i)
		{
			_targetsDiscriminatedLogFile << " [";
			for(unsigned int j = 0; j < vecTargetsDiscriminated[i].size(); ++j)
			{
				_targetsDiscriminatedLogFile << vecTargetsDiscriminated[i][j];

				if(j < (vecTargetsDiscriminated[i].size() - 1))
					_targetsDiscriminatedLogFile << ",";
			}
			_targetsDiscriminatedLogFile << "]";
		}
		_targetsDiscriminatedLogFile << endl;
		_targetsDiscriminatedLogFile.flush();
	}

}


