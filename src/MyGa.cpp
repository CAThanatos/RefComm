/*
 * MyGa.cpp
 *
 *  Created on: Jan 12, 2012
 *      Author: steffen
 */

#include "MyGa.h"


MyGa::MyGa(const char* logFileNamePrefix)
{
  genomes.clear();
  mutRate=0.005;
  generation = 0;
  popSize=200;
  noOfGenerations = 2000;
  done = false;
  tSize=2;
  normalizeMutRate=false;
  fixPopulation=-1;
  char strOut[100];
  sprintf( strOut, "generations.%s.dat", logFileNamePrefix );
  logFile.open(strOut,ios::trunc);

  sprintf( strOut, "generations.%s.parents.dat", logFileNamePrefix );
  logFileParents.open(strOut,ios::trunc);
  strcpy(_filePrefix,logFileNamePrefix);
}

MyGa::~MyGa()
{
  logFile.close();
  logFileParents.close();
}

void MyGa::writePopulation(bool parents)
{
	if(fixPopulation<0)
	{
  ofstream popFile;
  char strOut[100];
  if(parents)
  	sprintf( strOut, "%d.%s.parents.pop", generation, _filePrefix );
  else
  	sprintf( strOut, "%d.%s.pop", generation, _filePrefix );
  //overwrite existing file
  popFile.open(strOut,ios::trunc);

  //sort according to fitness (ascending)
  multimap <float, GARealGenome&> sorted;
  for(int i=0;i<popSize;++i)
  {
  	if(parents)
  	{
  		if(genomes[i].isParent)
  		{
        sorted.insert(pair<float, GARealGenome&>(genomes[i].fitness(), genomes[i]));
  		}
  	}
  	else
  	{
      sorted.insert(pair<float, GARealGenome&>(genomes[i].fitness(), genomes[i]));
  	}
  }

  //reverse order, so it is descending
  multimap<float, GARealGenome&>::iterator it=sorted.end();
  do
  {
    --it;
    popFile << (*it).second << endl;
//    cout << counter << ":  " << (*it).first << " [" << (*it).second << "]" << endl;
  }while(it!=sorted.begin());

  popFile.close();
	}
}

void MyGa::writeStats(bool parents)
{
	if(parents)
	{
		vector<double> fit,sigDiv,velDiv,commTime,noOfMutations;

	  for(int i=0;i<popSize;++i)
	  {
	  	if(genomes[i].isParent)
	  	{
	  		fit.push_back(genomes[i].fitness());
	  		sigDiv.push_back(genomes[i].sigDiv);
	  		velDiv.push_back(genomes[i].velDiv);
	  		commTime.push_back(genomes[i].commTime);
	  		noOfMutations.push_back(genomes[i].noOfMutations);
	  	}
	  }
	  double mean, sd;
	  mean=HelperFunction::getMean(fit);
	  sd=HelperFunction::getSD(fit, mean);
	  //1-3
    logFileParents << generation << " " << mean << " " << sd;

    //4-5
    mean=HelperFunction::getMean(sigDiv);
	  sd=HelperFunction::getSD(sigDiv, mean);
	  logFileParents << " " << mean << " " << sd;
	  //6-7
	  mean=HelperFunction::getMean(velDiv);
	  sd=HelperFunction::getSD(velDiv, mean);
	  logFileParents << " " << mean << " " << sd;
	  //8
    logFileParents <<  " " << fit.size();

    //9-10
    mean=HelperFunction::getMean(commTime);
	  sd=HelperFunction::getSD(commTime, mean);
    logFileParents <<  " " << mean << " " << sd;

    //11-12
    mean=HelperFunction::getMean(noOfMutations);
	  sd=HelperFunction::getSD(noOfMutations, mean);
    logFileParents <<  " " << mean << " " << sd;

    logFileParents << endl;
    logFileParents.flush();
	}
	else
	{
		vector<double> fit(popSize);
		for(int i=0;i<popSize;++i)
		{
			fit[i]=genomes[i].fitness();
		}
	  double meanFit=HelperFunction::getMean(fit);
	  double maxFit=HelperFunction::getMax(fit);
	  double minFit=HelperFunction::getMin(fit);
	  double sd=HelperFunction::getSD(fit, meanFit);

	  cout << generation
	  		 << " " << meanFit << " " << maxFit
	          << " " << minFit
	          << " " << sd <<  endl;
    logFile << generation << " " << meanFit << " " << maxFit
        << " " << minFit
        << " " << sd <<  endl;

    logFile.flush();
	}
}

void MyGa::select(bool recordAllInd)
{
  if(generation==noOfGenerations)
  {
    done=true;
    return;
  }

  if(fixPopulation<0)
  {
  	vector<GARealGenome> newGenomes;

  	int keep,tmp;
  	for(int i=0;i<popSize;++i)
  	{
  		vector<int> competitors;
  		while(competitors.size()<tSize)
  		{
  			tmp=UniformRandomInteger(0,popSize-1);
  			// XXX (GI) added pointers here
  			if( find(competitors.begin(), competitors.end(), tmp) == competitors.end() )
  			{
  				competitors.push_back(tmp);
  			}
  		}

  		keep=competitors[0];
  		for(int j=1;j<tSize;++j)
  		{
  			if(genomes[competitors[j]].fitness()>genomes[keep].fitness()) keep=competitors[j];
  		}

  		//  	for(int j=1;j<tSize;++j)
  		//  	{
  		//  		tmp=UniformRandomInteger(0,popSize-1);
  		//  		if(genomes[tmp].fitness()>genomes[keep].fitness()) keep=tmp;
  		//  	}

  		newGenomes.push_back(GARealGenome(genomes[keep]));
  		genomes[keep].isParent=true;
  	}
  	writeStats(true);
  	if(recordAllInd)
  	{
  		writePopulation(false);
  		writePopulation(true);
  	}
  	genomes.clear();
  	genomes = newGenomes;

  	float normMutRate=mutRate/genomes[0].size();

  	for(int i=0;i<popSize;++i)
  	{
  		genomes[i].isParent=false;
  		if(normalizeMutRate)    genomes[i].noOfMutations = genomes[i].mutate(normMutRate);
  		else genomes[i].noOfMutations = genomes[i].mutate(mutRate);
  	}
  }
  else
  {
  	writeStats(true);
  }


	for(int i=0;i<popSize;++i)
	{
		genomes[i].fitness(0.0);
		genomes[i].sigDiv=0.0;
		genomes[i].velDiv=0.0;
		genomes[i].commTime=0.0;
	}

  generation++;


}

void MyGa::selectRW(bool recordAllInd)
{
  if(generation==noOfGenerations)
  {
    done=true;
    return;
  }
  vector<GARealGenome> newGenomes;

  double totalFitness=0.0;
  for(int i=0;i<popSize;++i)
  {
  	totalFitness+=genomes[i].fitness();
  }

  if(totalFitness<=0)
  {
    for(int i=0;i<popSize;++i)
    {
    	genomes[i].fitness(1.0/popSize);
    }
  }
  else
  {
    for(int i=0;i<popSize;++i)
    {
    	genomes[i].fitness(genomes[i].fitness()/totalFitness);
    }
  }


  double randD, sum;
  int parent;
  for(int i=0;i<popSize;i++)
  {
  	randD=UniformRandom(0.0, 1.0);
  	sum=0;
  	for(parent=0;parent<popSize;parent++)
  	{
  		sum+=genomes[parent].fitness();
  		if(sum>=randD) break;
  	}
    newGenomes.push_back(genomes[parent]);
    genomes[parent].isParent=true;
  }


//	//sort according to fitness (ascending)
//  multimap <float, GARealGenome&> sorted;
//  for(int i=0;i<popSize;++i)
//    sorted.insert(pair<float, GARealGenome&>(genomes[i].fitness(), genomes[i]));
//
//  double randD, sum;
//  multimap<float, GARealGenome&>::iterator it;
//  for(int i=0;i<popSize;++i)
//  {
//  	randD=UniformRandom(0.0, 1.0);
//  	sum=0;
//  	for(it=sorted.begin();it!=sorted.end();it++)
//  	{
//  		sum+=(*it).first;
//  		if(sum>=randD) break;
//  	}
//    newGenomes.push_back(GARealGenome((*it).second));
//  }

  for(int i=0;i<popSize;++i)
  {
  	genomes[i].fitness(genomes[i].fitness()*totalFitness);
  }
  writeStats(true);
  if(recordAllInd)
  {
  	writePopulation(false);
    writePopulation(true);
  }

  genomes.clear();
  genomes = newGenomes;
  double normMutRate=mutRate/genomes[0].size();

  for(int i=0;i<popSize;++i)
  {
    genomes[i].fitness(0.0);
    // if(normalizeMutRate)  genomes[i].mutate(normMutRate);
    // else genomes[i].mutate(mutRate);
    if(normalizeMutRate)  genomes[i].noOfMutations = genomes[i].mutate(normMutRate);
    else genomes[i].noOfMutations = genomes[i].mutate(mutRate);
    genomes[i].isParent=false;
    genomes[i].sigDiv=0.0;
    genomes[i].velDiv=0.0;
    genomes[i].commTime=0.0;
  }
  generation++;

}
