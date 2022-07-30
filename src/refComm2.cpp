//============================================================================
// Name        : refComm2.cpp
// Author      : Steffen Wischmann
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <string.h>
#include <libconfig.h++>
#include <vector>
#include <iterator>
#include <algorithm>

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "Simulation.h"
#include "random.h"
#include "PopulationRecoder.h"
#include "HelperFunction.h"


//#include <ga/GASimpleGA.h>
//#include <ga/GABin2DecGenome.h>
//#include <ga/GA1DArrayGenome.h>
//#include <ga/std_stream.h>
#include <ga/ga.h>
#define INSTANTIATE_REAL_GENOME
#include <ga/GARealGenome.h>
#include "MyGa.h"
using namespace std;
using namespace libconfig;

float Objective(GAGenome &);
unsigned int randomSeed();
MyGa* fillRealGenome(bool isSender, int loadGeneration);
void setNoHiddenNetwork(GARealAlleleSetArray &array, const char* agentName, const Setting& root);
void setHiddenLayerNetwork(GARealAlleleSetArray &array, const char* agentName, const Setting& root);
void readSettings();
Simulation *sim;
PopulationRecoder *popRecorder;

vector<int> logGenerations;


int main(int argc, char **argv) {
  int testIndividual = 0;
  int testGeneration = 0;
  int loadGeneration = -1;
  HelperFunction *help= HelperFunction::Instance();

  help->setTesting(false);
  help->setLoading(false);
  if(argc > 2)
  {
    if(0 == strcmp(argv[1], "-l"))
    {
      help->setLoading(true);
      loadGeneration = atoi(argv[2]);
    }
    else if(0 == strcmp(argv[1], "-t"))
    {
    	help->setTesting(true);
      testGeneration = atoi(argv[2]);
      testIndividual = atoi(argv[3]);
    }
    else
    {
      cerr << "Error: argument " << argv[1] << " not recognized." << endl;
      return EXIT_FAILURE;
    }
  }

  //important! always use this function first and use the returned seed for all random number generators.
  // this guarantees repeatability
  unsigned int seed=randomSeed();
  //initialize random number generator for the ga
  GARandomSeed(seed);
  //initialize random number generator for the ctrrn
  SetRandomSeed(seed);
  srand (seed);

  readSettings();
  vector<double> retVals(4);

  if(help->isTesting())
  {
    vector<int> senderIDs, receiverIDs;
  	help->setTestParents(false);
  
  	//comment in for behavioral analysis of all individuals
  	// senderIDs.push_back(testIndividual);
  	// receiverIDs.push_back(testIndividual);

  	if (testIndividual < 0)
    {
    	testIndividual=help->getPopSize();
    	help->setTestParents(false);
    }
    else
    {
    	help->setTestParents(true);
    }

    for(int j=0; j <testIndividual; ++j)
    {
    	senderIDs.push_back(j);
    	receiverIDs.push_back(j);
    }

    random_shuffle(senderIDs.begin(), senderIDs.end());
    random_shuffle(receiverIDs.begin(), receiverIDs.end());
    for(testIndividual = 0; testIndividual<senderIDs.size(); ++testIndividual)
    {
    	help->setLogInfoFullGeneration(true);
    	help->openDetailedInfoLogFile("tmpInfo.log.txt");
    	help->openBehavioralDataFile("tmp.txt");
      
    	//commment in for behavioral analysis of all individuals
    	char strOut[100];
    	// sprintf(strOut, "behavior.%i.txt", senderIDs[testIndividual]);
    	// help->openBehavioralDataFile(strOut);

    	vector<double> fit;
    	vector<float> genes;
    	double sigDiv, velDiv;
    	HelperFunction::readGenesFromFile(true, testGeneration, senderIDs[testIndividual], genes);
    	HelperFunction::readGenesFromFile(false, testGeneration, receiverIDs[testIndividual], genes);

    	help->initAveraging();
    	sim->evaluate(genes, testGeneration, popRecorder, retVals);
    	fit.push_back(retVals[0]);
    	sigDiv=retVals[1];
    	velDiv=retVals[2];

    	help->closeDetailedInfoLogFile();
    	help->closeBehavioralDataFile();
//    	for(int i=0;i<5;++i)
//    		cout << i << " " << help->getOnset(i) << " "<< fit[0] << endl;

    	help->setAveraging(false);
    	help->setUseAverage(AVERAGE_SPEED);
    	help->openDetailedInfoLogFile("tmpInfo.log.cV.txt");
    	help->openBehavioralDataFile("tmp.cV.txt");
    	sim->evaluate(genes, testGeneration, popRecorder, retVals);
    	fit.push_back(retVals[0]);
    	help->closeDetailedInfoLogFile();
    	help->closeBehavioralDataFile();

    	help->setUseAverage(AVERAGE_SIGNAL);
    	help->openDetailedInfoLogFile("tmpInfo.log.cS.txt");
    	help->openBehavioralDataFile("tmp.cS.txt");
    	sim->evaluate(genes, testGeneration, popRecorder, retVals);
    	fit.push_back(retVals[0]);
    	help->closeDetailedInfoLogFile();
    	help->closeBehavioralDataFile();

    	help->setUseAverage(AVERAGE_SIGNALONSET);
    	help->openDetailedInfoLogFile("tmpInfo.log.onset.txt");
    	// help->openBehavioralDataFile("tmp.onset.txt");
      sprintf(strOut, "behaviorOnset.%i.txt", senderIDs[testIndividual]);
      help->openBehavioralDataFile(strOut);
    	//help->velStepNoiseEnv=0.2;
    	sim->evaluate(genes, testGeneration, popRecorder, retVals);
    	fit.push_back(retVals[0]);
    	help->closeDetailedInfoLogFile();
    	help->closeBehavioralDataFile();

    	help->setUseAverage(AVERAGE_SIGNALLENGTH);
    	help->openDetailedInfoLogFile("tmpInfo.log.length.txt");
    	help->openBehavioralDataFile("tmp.length.txt");
      // sprintf(strOut, "behaviorLength.%i.txt", senderIDs[testIndividual]);
      // help->openBehavioralDataFile(strOut);
    	//help->velStepNoiseEnv=0.2;
    	sim->evaluate(genes, testGeneration, popRecorder, retVals);
    	fit.push_back(retVals[0]);
    	help->closeDetailedInfoLogFile();
    	help->closeBehavioralDataFile();

    	help->setUseAverage(AVERAGE_SIGNALLENGTH2);
    	help->openDetailedInfoLogFile("tmpInfo.log.length2.txt");
    	help->openBehavioralDataFile("tmp.length2.txt");
      // sprintf(strOut, "behaviorLength2.%i.txt", senderIDs[testIndividual]);
      // help->openBehavioralDataFile(strOut);
    	//help->velStepNoiseEnv=0.2;
    	sim->evaluate(genes, testGeneration, popRecorder, retVals);
    	fit.push_back(retVals[0]);
    	help->closeDetailedInfoLogFile();
    	help->closeBehavioralDataFile();

      // std::cout << "length3" << std::endl;
      help->setUseAverage(AVERAGE_SIGNALLENGTH3);
      help->openDetailedInfoLogFile("tmpInfo.log.length3.txt");
      help->openBehavioralDataFile("tmp.length3.txt");
      // sprintf(strOut, "behaviorLength3.%i.txt", senderIDs[testIndividual]);
      // help->openBehavioralDataFile(strOut);
      //help->velStepNoiseEnv=0.2;
      sim->evaluate(genes, testGeneration, popRecorder, retVals);
      fit.push_back(retVals[0]);
      help->closeDetailedInfoLogFile();
      help->closeBehavioralDataFile();

      // std::cout << "length4" << std::endl;
      help->setUseAverage(AVERAGE_SIGNALLENGTH4);
      help->openDetailedInfoLogFile("tmpInfo.log.length4.txt");
      help->openBehavioralDataFile("tmp.length4.txt");
      // sprintf(strOut, "behaviorLength4.%i.txt", senderIDs[testIndividual]);
      // help->openBehavioralDataFile(strOut);
      //help->velStepNoiseEnv=0.2;
      sim->evaluate(genes, testGeneration, popRecorder, retVals);
      fit.push_back(retVals[0]);
      help->closeDetailedInfoLogFile();
      help->closeBehavioralDataFile();

    	cout << testIndividual;
    	for (int i=0;i<fit.size();i++)
    	{
    		cout << " " << fit[i];
    	}
    	cout << " " << sigDiv << " " << velDiv;
    	cout <<endl;
    }
  }
  else
  {
  	MyGa* sGa=fillRealGenome(true, loadGeneration);
  	MyGa* rGa=fillRealGenome(false, loadGeneration);
  	//cout << "Sender genome size: " << sGa->genomes[0].size() <<endl;
  	//cout << "Receiver genome size: " << rGa->genomes[0].size() <<endl;

    vector<int> senderIDs, receiverIDs;

    for(int i = 0; i<help->getNoOfTests();++i)
    {
      for(int j=0; j < rGa->popSize; ++j)
      {
        senderIDs.push_back(j);
        receiverIDs.push_back(j);
      }
    }

    //first generation
    while(!rGa->done)
    {
    	help->setLogInfoFullGeneration(false);
      //simTime=UniformRandomInteger(simTimeLimits[0],simTimeLimits[1]);
      //cout << "simtime : " << simTime << endl;

      //evaluate
      random_shuffle(senderIDs.begin(), senderIDs.end());
      random_shuffle(receiverIDs.begin(), receiverIDs.end());
      bool recordAllInd=false;

      if(!logGenerations.empty() && (logGenerations[0] == rGa->generation || logGenerations[0] < 0))  
      {
        help->setLogInfoFullGeneration(true);
        char strOut[100];
        sprintf( strOut, "%d.log.txt", rGa->generation );
        help->openDetailedInfoLogFile(strOut);
        recordAllInd = true;

        if (logGenerations[0] >= 0)
          logGenerations.erase(logGenerations.begin());
      }

      for(unsigned int i = 0; i<senderIDs.size(); ++i)
      {
        GARealGenome & tmpSend=sGa->genomes[senderIDs[i]];
        GARealGenome & tmpRec=rGa->genomes[receiverIDs[i]];

        vector<float> genes(tmpSend.size()+tmpRec.size());

        //fill with sender genes
        for(int g=0;g<tmpSend.size();++g)
          genes[g] = tmpSend.gene(g);

        //fill with receiver genes
        for(int g=0;g<tmpRec.size();++g)
          genes[tmpSend.size()+g] = tmpRec.gene(g);

        sim->evaluate(genes, rGa->generation, popRecorder, retVals);
        if(help->isCutOneTargetFitness())
        {
        	if (retVals[0]<=1.0/help->getNoOfTargets()) retVals[0]=0;
        }

        for(unsigned int r=0;r<retVals.size();++r)
        {
        	retVals[r]/=help->getNoOfTests();
        }
        sGa->genomes[senderIDs[i]].fitness(sGa->genomes[senderIDs[i]].fitness()+retVals[0]);
        sGa->genomes[senderIDs[i]].sigDiv+=retVals[1];
        sGa->genomes[senderIDs[i]].velDiv+=retVals[2];
        sGa->genomes[senderIDs[i]].commTime+=retVals[3];

        rGa->genomes[receiverIDs[i]].fitness(rGa->genomes[receiverIDs[i]].fitness()+retVals[0]);
        rGa->genomes[receiverIDs[i]].sigDiv+=retVals[1];
        rGa->genomes[receiverIDs[i]].velDiv+=retVals[2];
        rGa->genomes[receiverIDs[i]].commTime+=retVals[3];
      }

      if(help->isLogInfoFullGeneration())
      {
        //sGa->writePopulation("sender");
        //rGa->writePopulation("receiver");
        help->closeDetailedInfoLogFile();
      }

      popRecorder->writeData2(rGa->generation);
      popRecorder->clearRecords();

      //cout << "Gen: " << sGa->generation << endl;
      sGa->writeStats(false);

      if(help->isTournamentSelection())
      {
      	rGa->select(recordAllInd);
      	sGa->select(recordAllInd);
      }
      else
      {
        rGa->selectRW(recordAllInd);
        sGa->selectRW(recordAllInd);
      }
    }
    delete rGa;
    delete sGa;
  }

  delete sim;
  delete popRecorder;
  delete help;
  return(EXIT_SUCCESS);
}


/**
 * this is the main function for fitness evaluation
 * this function is passed to the ga
 *
 * @param g comes fromt the ga
 * @return fitness value
 */
float Objective(GAGenome& g)
{
  return 0;
}

/**
 * generates and stores a random seed in "seed.txt" that is used throughout an experiment
 * this can be used to repeat exactly the same experiment.
 *
 * @return random seed
 */
unsigned int randomSeed(){
  struct timeval startTime;
  gettimeofday(&startTime, NULL);
  //calc microseconds
  int a = startTime.tv_sec*1000000 + (startTime.tv_usec);
  a = abs(a);
  if(!HelperFunction::Instance()->isTesting())
  {
    ofstream myfile;
    myfile.open ("seed.txt",ios::trunc);
    myfile << a << endl;
    myfile.close();
  }
  return a;
}

void readSettings()
{
	HelperFunction::Instance()->readSettings();
  // Read the config file. If there is an error, report it and exit.
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

  const Setting& tmpSet = root["recording"]["logGenerations"];
  //cout << " log gens: ";
    for(int i=0;i<tmpSet.getLength(); ++i)
    {
    //  cout << (int)tmpSet[i] << ", ";
      logGenerations.push_back((int)tmpSet[i]);
    }
    //cout << endl;


    const Setting& tmpSet2 = root["recording"];
    bool tmpBool = false;
    tmpSet2.lookupValue("logAllInfo", tmpBool);

    if(!HelperFunction::Instance()->isTesting())
    	popRecorder = new PopulationRecoder(tmpBool);

    //initSimulation
    sim = new Simulation(root);
}

MyGa* fillRealGenome(bool isSender, int loadGeneration = -1){
	string agentName;
	GARealAlleleSetArray array;

	// Read the config file. If there is an error, report it and exit.
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

  if(isSender)
  {
    //2 genes for movement signal control
    array.add(0, 2);
    array.add(0, 2);
    agentName="sender";
  }
  else
  {
    agentName="receiver";
  }

  int hidden;
  const Setting& agentSetting = root[agentName.c_str()];
  agentSetting.lookupValue("hiddenNo", hidden);

  if(hidden==0)
    setNoHiddenNetwork(array, agentName.c_str(), root);
  else
    setHiddenLayerNetwork(array, agentName.c_str(), root);

  GARealGenome genome(array, Objective);
  genome.crossover(GARealOnePointCrossover);
  //if a gene is mutated, it flips to a random number within the bounds of the gene
  if(!HelperFunction::Instance()->isGaussianChange())
    genome.mutator(GARealGenome::FlipMutator);

  const Setting& simSettings = root["simulation"];
  MyGa* ga=new MyGa(agentName.c_str());
  simSettings.lookupValue("popSize", ga->popSize);
  simSettings.lookupValue("mutationRate", ga->mutRate);
  simSettings.lookupValue("normalizeMutation", ga->normalizeMutRate);
  simSettings.lookupValue("noOfGenerations", ga->noOfGenerations);
  simSettings.lookupValue("tSize", ga->tSize);

  for(int i = 0; i<ga->popSize;++i)
    ga->genomes.push_back(GARealGenome(genome));

  for(int i = 0; i<ga->popSize;++i)
  {
  	//ga->genomes[i].initializer(genome.initializer());	// XXX (GI) this initialization might be needed
  	ga->genomes[i].initialize();
  	ga->genomes[i].isParent=true;
  	ga->genomes[i].sigDiv=0.0;
  	ga->genomes[i].velDiv=0.0;
  	ga->genomes[i].commTime=0.0;
  }

  agentSetting.lookupValue("fixPopulation", ga->fixPopulation);
  if(ga->fixPopulation>=0)
  {
  	cout << agentName.c_str() << " fixed." << endl;
    for(int i = 0; i<ga->popSize;++i)
    {
    	vector<float> genes;
    	HelperFunction::readGenesFromFile(isSender, ga->fixPopulation, i, genes);
    	for(int g=0; g<ga->genomes[i].size(); ++g)
    	{
    		ga->genomes[i].gene(g, genes[g]);
    	}
    }
  }
  if(loadGeneration > 0)
  {
    ga->loadGeneration = loadGeneration;
    ga->generation = loadGeneration;

    cout << agentName.c_str() << " loaded." << endl;
    for(int i = 0; i<ga->popSize;++i)
    {
      vector<float> genes;
      HelperFunction::readGenesFromFile(isSender, loadGeneration, i, genes);
      for(int g=0; g<ga->genomes[i].size(); ++g)
      {
        ga->genomes[i].gene(g, genes[g]);
      }
    }
  }
  char strOut[100];
  sprintf( strOut, "%s.start.pop", agentName.c_str() );
  ifstream ifile(strOut);
  if (ifile)
  {
  	cout << strOut << " exists" << endl;
    for(int i = 0; i<ga->popSize;++i)
    {
//    	cout << i << " exists" << endl;
    	vector<float> genes;
    	HelperFunction::readGenesFromFile(strOut, i, genes);
    	for(int g=0; g<ga->genomes[i].size(); ++g)
    	{
    		ga->genomes[i].gene(g, genes[g]);
    	}
    }
  }

	return ga;
}

void setNoHiddenNetwork(GARealAlleleSetArray &array, const char* agentName, const Setting& root)
{
	float biasLimit, weightLimit, tauLoLimit, tauHiLimit;
	int inputs,outputs,hidden;
	//cout << agentName << endl;
	const Setting& rnnSetting = root["rnn"];
	rnnSetting.lookupValue("biasLimit", biasLimit);
	rnnSetting.lookupValue("weightLimit", weightLimit);
	rnnSetting.lookupValue("tauLoLimit", tauLoLimit);
	rnnSetting.lookupValue("tauHiLimit", tauHiLimit);
    //cout << " Limits: bias=" << biasLimit << "; weight=" << weightLimit << "; tauLo=" << tauLoLimit << "; tauHi=" << tauHiLimit << endl;

	const Setting& setting = root[agentName];
	setting.lookupValue("inputNo", inputs);
	setting.lookupValue("outputNo", outputs);
	setting.lookupValue("hiddenNo", hidden);
    //cout << " inputs: " << inputs << "; hidden: " << hidden << "; outputs: " << outputs << endl;

	//bias
	for(int i=0; i<inputs+outputs; i++)
		array.add(-biasLimit, biasLimit);

	//synapses between input->hidden + hidden-> output, self-connections of hidden, hidden->hidden
	for(int i=0; i<inputs*inputs+2*inputs*outputs+outputs*outputs; i++)
		array.add(-weightLimit, weightLimit);

	//time constants
  	for(int i=0; i<inputs+outputs; i++)
		array.add(tauLoLimit, tauHiLimit);
}

void setHiddenLayerNetwork(GARealAlleleSetArray &array, const char* agentName, const Setting& root)
{
	float biasLimit, weightLimit, tauLoLimit, tauHiLimit;
	int inputs,outputs,hidden;
	//cout << agentName << endl;
	const Setting& rnnSetting = root["rnn"];
	rnnSetting.lookupValue("biasLimit", biasLimit);
	rnnSetting.lookupValue("weightLimit", weightLimit);
	rnnSetting.lookupValue("tauLoLimit", tauLoLimit);
	rnnSetting.lookupValue("tauHiLimit", tauHiLimit);
  bool fixTau=false;
  bool fixBias=false;
  bool fixRange=false;
  if(strcmp(agentName, "sender")==0)
  {
  	root["sender"].lookupValue("fixTau", fixTau);
  	root["sender"].lookupValue("fixBias", fixBias);
  	root["sender"].lookupValue("fixRange", fixRange);
  }
//	cout << " Limits: bias=" << biasLimit << "; weight=" << weightLimit << "; tauLo=" << tauLoLimit << "; tauHi=" << tauHiLimit << endl;

	const Setting& agentSetting = root[agentName];
	agentSetting.lookupValue("inputNo", inputs);
	agentSetting.lookupValue("outputNo", outputs);
	agentSetting.lookupValue("hiddenNo", hidden);
//	cout << " inputs: " << inputs << "; hidden: " << hidden << "; outputs: " << outputs << endl;
	//bias
	for(int i=0; i<inputs+hidden+outputs; i++)
	{
		if(fixBias && i==0)
		{
			if(fixRange)
				array.add(-1.6681, 0.4132);
			else
				array.add(-0.5319, -0.5319);
		}
		else
			array.add(-biasLimit, biasLimit);
	}

	if(HelperFunction::Instance()->isFfn())
	{
		//synapses between input->hidden + hidden-> output
		for(int i=0; i<inputs*hidden+hidden*outputs; i++)
			array.add(-weightLimit, weightLimit);
	}
	else
	{
		//synapses between input->hidden + hidden-> output, self-connections of hidden, hidden->hidden
		for(int i=0; i<inputs*hidden+hidden*hidden+hidden*outputs; i++)
		{
			array.add(-weightLimit, weightLimit);
		}
	}

	//time constants
	for(int i=0; i<inputs+hidden+outputs; i++)
	{
		if(fixTau && i == 0)
		{
			if(fixRange)
				array.add(0.1003, 0.4390);
			else
				array.add(0.2154, 0.2154);
		}
		else if (fixTau && i==inputs+hidden+outputs-1)
		{
			if(fixRange)
				array.add(0.1018, 0.3758);
			else
				array.add(0.1628, 0.1628);
		}
		else
		{
			array.add(tauLoLimit, tauHiLimit);
		}
	}
}

