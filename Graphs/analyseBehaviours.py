#!/usr/bin/env python

import os
import sys
import re
import argparse
import math
import shutil
import pickle

# For ssh compatibility
# matplotlib.use('Agg')

from matplotlib import rcParams
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import seaborn as sns
from pylab import Polygon
from scipy import stats
import brewer2mpl
import numpy as np


# SEABORN
sns.set()
sns.set_style('white')
sns.set_context('paper')

# BREWER
# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
palette = bmap.mpl_colors
# palette = sns.color_palette("husl", 4)

# MATPLOTLIB PARAMS
rcParams['font.size'] = 24
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 24
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 24
rcParams['mathtext.default'] = 'regular'

DPI = 96
# FIGSIZE = (1024/DPI, 744/DPI)
FIGSIZE = (1280/DPI, 1024/DPI)

MARGIN_BARS_DIFF = 0.2
WIDTH_BARS_DIFF = (1.-2.*MARGIN_BARS_DIFF)/2.
MARGIN_BARS_RATIO = 0.3
WIDTH_BARS_RATIO = (1.-2.*MARGIN_BARS_RATIO)/2.

THRESHOLD_SUCCESS = 15
TARGETS = {
						1 : [[3.*math.pi/8., math.pi/2.], [math.pi/2., 5.*math.pi/8]],
						2 : [[5.*math.pi/8., 3.*math.pi/4.], [3.*math.pi/4., 7.*math.pi/8.]],
						3 : [[7.*math.pi/8., math.pi], [-math.pi, -7.*math.pi/8.]],
						4 : [[-7.*math.pi/8., -3.*math.pi/4.], [-3.*math.pi/4., -5.*math.pi/8]],
						5 : [[-5.*math.pi/8., -math.pi/2.], [-math.pi/2., -3.*math.pi/8]]
}

OUTPUT_DIR = "./OutputGraphs"

def main(args) :
	for d in args.directories :
		if os.path.isdir(d) :
			tuples = os.walk(d)
			listExpDirs = {}

			regExpRep = re.compile(r"^(\d+)$")
			for tuple in tuples :
				if len(tuple[1]) > 0 :
					print("->" + tuple[0])
					if regExpRep.search(tuple[1][0]) :
						hashExp = dict()

						replicates = [replicate for replicate in os.listdir(tuple[0]) if regExpRep.search(replicate)]

						if args.onset :
							regExpBehaviourFile = re.compile(r"^behaviorLength4\.(\d+)\.txt$")
						elif args.length :
							regExpBehaviourFile = re.compile(r"^behaviorOnset\.(\d+)\.txt$")
						else :
							regExpBehaviourFile = re.compile(r"^behavior\.(\d+)\.txt$")

						hashTargetsFound = dict()
						hashSigDiff = dict()
						nbIndStratOneSiteTot = 0
						nbIndStratAllSitesTot = 0
						nbIndStratTot = 0
						for replicate in replicates :
							if args.replicate == None or replicate in args.replicate :
								print("\tReplicate " + str(replicate))
								s = regExpRep.search(replicate)
								if s :
									numRep = s.group(1)
									replicateDir = os.path.join(tuple[0], replicate)
									listBehaviourFiles = [file for file in os.listdir(replicateDir) if regExpBehaviourFile.search(file)]

									hashReplicate = dict()

									if args.load != None :
										fileLoad = args.load + "Dir" + replicate + ".dat"
										if os.path.isfile(fileLoad) :
											print("\t\t-> Loading : " + (fileLoad))
											hashReplicate = pickle.load(open(fileLoad, 'rb'))
									else :
										print("\t\t-> Reading behavioural files...")
										for behaviourFile in listBehaviourFiles :
											# print("\t\tBehaviour file : " + behaviourFile)
											s = regExpBehaviourFile.search(behaviourFile)
											if s :
												numInd = s.group(1)

												with open(os.path.join(replicateDir, behaviourFile), "r") as fileRead :
													fileRead = fileRead.readlines()

													numTrial = 0
													dictTrials = {}
													dictTrial = list()
													for line in fileRead :
														line = line.rstrip('\n').split()
														time = int(line[0])

														if time == 0 :
															if numTrial > 0 :
																dictTrials[numTrial] = dictTrial
																dictTrial = list()

															numTrial += 1

														hashBehaviourStep = dict()
														hashBehaviourStep['targetPos'] = float(line[1])
														hashBehaviourStep['senderPos'] = float(line[5])
														hashBehaviourStep['receiverPos'] = float(line[6])

														hashBehaviourStep['fitness'] = float(line[4])

														hashBehaviourStep['senderSignal'] = float(line[8])
														hashBehaviourStep['receiverSignal'] = float(line[9])
														dictTrial.append(hashBehaviourStep)

													dictTrials[numTrial] = dictTrial
													hashReplicate[numInd] = dictTrials

										if args.serialize != None :
											fileSerialization = args.serialize + "Dir" + replicate + ".dat"
											print("\t\t-> Saving to : " + (fileSerialization))
											if os.path.isfile(fileSerialization) :
												os.remove(fileSerialization)

											pickle.dump(hashReplicate, open(fileSerialization, 'wb'))

									listTargetsPos = transform2Pi([hashReplicate['0'][trial][0]['targetPos'] for trial in hashReplicate['0'].keys()])
									indStats = dict()
									for ind in hashReplicate :
										listOnsets = list()
										listLengths = list()
										listTargetsFound = list()
										fitness = 0
										listComparisonSignal = list()
										for trial in hashReplicate[ind] :
											hashTrial = hashReplicate[ind][trial]

											exitedCom = False 
											onset = -1
											together = False
											length = -1
											nbStepsInTarget = 0
											signalTrial = list()
											for timeStep in range(0, len(hashTrial)) :
												if inCommArea(hashTrial[timeStep]['senderPos']) :
													if exitedCom :
														if not together and inCommArea(hashTrial[timeStep]['receiverPos']) :
															together = True

														if together :
															if onset == -1 :
																onset = timeStep
												else :
													if not exitedCom :
														exitedCom = True

												if not inCommArea(hashTrial[timeStep]['receiverPos']) :
													if together and onset > -1 and length == -1 :
														length = timeStep - onset

												if timeStep == len(hashTrial) - 1 :
													fitness += hashTrial[timeStep]['fitness'] 

												if timeStep >= 80 and inTargetArea(hashTrial[timeStep]['receiverPos'], trial) :
													nbStepsInTarget += 1

												signalTrial.append(hashTrial[timeStep]['senderSignal'])

											targetFound = False
											if nbStepsInTarget >= THRESHOLD_SUCCESS :
												targetFound = True

											listComparisonSignal.append(signalTrial)
											listOnsets.append(onset)
											listLengths.append(length)
											listTargetsFound.append(targetFound)
										indStats[ind] = dict()
										indStats[ind]['onsets'] = listOnsets
										indStats[ind]['onsetsCorrelation'] = stats.pearsonr(list(filter(lambda x : x > -1, listOnsets)), [listTargetsPos[cpt] for cpt in range(0, len(listOnsets)) if listOnsets[cpt] > -1])
										indStats[ind]['lengths'] = listLengths
										indStats[ind]['lenghtsCorrelation'] = stats.pearsonr(list(filter(lambda x : x > -1, listLengths)), [listTargetsPos[cpt] for cpt in range(0, len(listLengths)) if listLengths[cpt] > -1])
										indStats[ind]['targetFound'] = listTargetsFound
										indStats[ind]['fitness'] = fitness

										diffSignal = 0
										nbComp = 0
										for trial in range(0, len(listComparisonSignal)) :
											for trial2 in range(trial + 1, len(listComparisonSignal)) :
												diffSignal += differenceSignal(listComparisonSignal[trial], listComparisonSignal[trial2])
												nbComp += 1
										assert(nbComp == 10)
										indStats[ind]['signalComp'] = diffSignal/float(nbComp)

									if args.length :
										outputData = os.path.join(os.path.join(OUTPUT_DIR, "Length"), "Dir" + replicate)
									elif args.onset :
										outputData = os.path.join(os.path.join(OUTPUT_DIR, "Onset"), "Dir" + replicate)
									else :
										outputData = os.path.join(os.path.join(OUTPUT_DIR, "Unconstrained"), "Dir" + replicate)

									if os.path.isdir(outputData) :
										shutil.rmtree(outputData)

									os.makedirs(outputData)

									hashTargetsFound[replicate] = list()
									hashSigDiff[replicate] = list()

									# ----- Individual info on onset and length -----
									print('\t\t-> Printing individual information')

									listTrials = [3, 4, 0, 1, 2]
									stringOut = ''

									diffSignalTotal = 0

									listInd = sorted(map(lambda ind : int(ind), indStats.keys()))
									listInd2 = [ind for ind,_ in sorted(zip(indStats.keys(), [indStats[ind]['fitness'] for ind in indStats.keys()]), key = lambda x : x[1], reverse = True)]
									# print(str(zip(indStats.keys(), [indStats[ind]['fitness'] for ind in indStats.keys()])))
									# print(sorted(zip(indStats.keys(), [indStats[ind]['fitness'] for ind in indStats.keys()]), key = lambda x : x[1]))
									# print(str(listInd2))
									nbIndStratOneSite = 0
									nbIndStratAllSites = 0
									nbIndStrat = 0
									for ind in listInd2 :
										stringOut += "Individual " + str(ind) + " : \n"
										stringOut += "Targets found : " + str(sum([1 for found in indStats[str(ind)]['targetFound'] if found])) + " (fitness = " + str(indStats[str(ind)]['fitness']) + ")\n"
										hashTargetsFound[replicate].append(sum([1 for found in indStats[str(ind)]['targetFound'] if found]))
										# stringOut += "\t Fitness = " + str(indStats[str(ind)]['fitness'][0]) + "\n"

										stringOut += "\t Onsets : "
										stringOut += str(indStats[str(ind)]['onsets'])
										stringOut += "/ Correlation : " + str(indStats[str(ind)]['onsetsCorrelation'])
										stringOut += "\n"

										stringOut += "\t Lengths : "
										stringOut += str(indStats[str(ind)]['lengths'])
										stringOut += "/ Correlation : " + str(indStats[str(ind)]['lenghtsCorrelation'])
										stringOut += "\n"

										stringOut += "\t Diff signal : "
										stringOut += str(indStats[str(ind)]['signalComp'])
										stringOut += "\n"
										diffSignalTotal += indStats[str(ind)]['signalComp']
										hashSigDiff[replicate].append(indStats[str(ind)]['signalComp'])

										if indStats[str(ind)]['fitness'] >= 0.19:
											if sum([1 for found in indStats[str(ind)]['targetFound'] if found]) == 1:
												nbIndStratOneSite += 1
												nbIndStrat += 1
											elif sum([1 for found in indStats[str(ind)]['targetFound'] if found]) == 0 :
												nbIndStratAllSites += 1
												nbIndStrat += 1

									nbIndStratAllSitesTot += nbIndStratAllSites
									nbIndStratOneSiteTot += nbIndStratOneSite
									nbIndStratTot += nbIndStrat

									diffSignalTotal /= float(len(listInd2))
									print('\t\t-> Total signal difference : ' + str(diffSignalTotal))

									stringOut = "NbIndividualsStrat = " + str(nbIndStrat) + "/StratOneSite = " + str(nbIndStratOneSite) + "/StratAllSites = " + str(nbIndStratAllSites) + "\n" + stringOut

									with open(os.path.join(outputData, 'fileInfo.txt'), 'w') as fileWrite :
										fileWrite.write(stringOut)


									# ----- ONSET AND LENGTH BOXPLOTS -----
									print('\t\t-> Drawing onset and length boxplots...')

									dataBoxPlotOnset = []
									dataBoxPlotLength = []
									for trial in listTrials :
										listTrialOnsets = list(filter(lambda onset : onset > -1, [indStats[ind]['onsets'][trial] for ind in indStats.keys()]))
										listTrialLengths = list(filter(lambda length : length > -1, [indStats[ind]['lengths'][trial] for ind in indStats.keys()]))
										dataBoxPlotOnset.append(listTrialOnsets)
										dataBoxPlotLength.append(listTrialLengths)

									# if numRep == "2" :
									# 	print(dataBoxPlotOnset)

									fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
									plt.grid()

									# dataStats = {}
									# index = 0
									# for data in dataBoxPlotOnset :
									# 	# print(data)
									# 	curDataStats = list()
									# 	for dataCmp in dataBoxPlotOnset :
									# 		# print(dataCmp)
									# 		U, P = stats.mannwhitneyu(data, dataCmp)
									# 		stars = "-"
									# 		if P < 0.0001:
									# 		   stars = "****"
									# 		elif (P < 0.001):
									# 		   stars = "***"
									# 		elif (P < 0.01):
									# 		   stars = "**"
									# 		elif (P < 0.05):
									# 		   stars = "*"

									# 		curDataStats.append(stars)
									# 	dataStats[index] = curDataStats
									# 	index += 1

									# for i in range(0, len(dataStats)) :
									# 	print("Data " + str(i) + " : ")
									# 	for j in range(0, len(dataStats[i])) :
									# 		print("Data " + str(j) + " : P = " + str(dataStats[i][j]))

									bp = ax.boxplot(dataBoxPlotOnset)

									# ax.spines['top'].set_visible(False)
									# ax.spines['right'].set_visible(False)
									# ax.spines['left'].set_visible(False)
									# ax.get_xaxis().tick_bottom()
									# ax.get_yaxis().tick_left()
									# ax.tick_params(axis='x', direction='out')
									# ax.tick_params(axis='y', length=0)

									ax.set_xlabel("Trial")
									ax.set_ylabel("Onset")
									ax.set_ylim(0, 100)

									x_ticks = range(0, len(listTrials))
									x_ticksLabels = listTrials
									ax.set_xticklabels(x_ticksLabels)

									# ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
									# ax.set_axisbelow(True)

									for i in range(0, len(bp['boxes'])):
										bp['boxes'][i].set_color(palette[i])
										# we have two whiskers!
										bp['whiskers'][i*2].set_color(palette[i])
										bp['whiskers'][i*2 + 1].set_color(palette[i])
										bp['whiskers'][i*2].set_linewidth(2)
										bp['whiskers'][i*2 + 1].set_linewidth(2)
										# top and bottom fliers
										# (set allows us to set many parameters at once)
										# bp['fliers'][i].set(markerfacecolor=palette[1],
										#                 marker='o', alpha=0.75, markersize=6,
										#                 markeredgecolor='none')
										# bp['fliers'][1].set(markerfacecolor=palette[1],
										#                 marker='o', alpha=0.75, markersize=6,
										#                 markeredgecolor='none')
										if (i * 2) < len(bp['fliers']) :
										   bp['fliers'][i * 2].set(markerfacecolor='black',
										                   marker='o', alpha=0.75, markersize=6,
										                   markeredgecolor='none')
										if (i * 2 + 1) < len(bp['fliers']) :
										   bp['fliers'][i * 2 + 1].set(markerfacecolor='black',
										                   marker='o', alpha=0.75, markersize=6,
										                   markeredgecolor='none')
										bp['medians'][i].set_color('black')
										bp['medians'][i].set_linewidth(3)
										# and 4 caps to remove
										# for c in bp['caps']:
										#    c.set_linewidth(0)

									for i in range(len(bp['boxes'])):
									   box = bp['boxes'][i]
									   box.set_linewidth(0)
									   boxX = []
									   boxY = []
									   for j in range(5):
									       boxX.append(box.get_xdata()[j])
									       boxY.append(box.get_ydata()[j])
									       boxCoords = zip(boxX,boxY)
									       boxPolygon = Polygon(boxCoords, facecolor = palette[i], linewidth=0)
									       ax.add_patch(boxPolygon)

									# fig.subplots_adjust(left=0.2)

									# y_max = np.max(np.concatenate(np.concatenate(dataBoxPlot[0][:], dataBoxPlot[1][:]), dataBoxPlot[2][:]))
									# y_min = np.min(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
									# y_min = 0

									# print y_max

									# Print stats
									# cptData = 0
									# while cptData < len(dataStats) :
									# 	cptComp = cptData + 1
									# 	while cptComp < len(dataStats[cptData]) :
									# 		if dataStats[cptData][cptComp] != "-" :
									# 			if cptData == 0 and cptComp == 2 :
									# 				ax.annotate("", xy=(cptData + 1, 7300), xycoords='data',
									# 				           xytext=(cptComp + 1, 7300), textcoords='data',
									# 				           arrowprops=dict(arrowstyle="-", ec='#000000',
									# 				                           connectionstyle="bar,fraction=0.05"))
									# 				ax.text((cptComp - cptData)/2 + cptData + 1, 7725, dataStats[cptData][cptComp],
									# 				       horizontalalignment='center',
									# 				       verticalalignment='center')
									# 			else :
									# 				ax.annotate("", xy=(cptData + 1, 7000), xycoords='data',
									# 				           xytext=(cptComp + 1, 7000), textcoords='data',
									# 				           arrowprops=dict(arrowstyle="-", ec='#000000',
									# 				                           connectionstyle="bar,fraction=0.05"))
									# 				ax.text(float(cptComp - cptData)/2 + cptData + 1, 7250, dataStats[cptData][cptComp],
									# 				       horizontalalignment='center',
									# 				       verticalalignment='center')

									# 		cptComp += 1
									# 	cptData += 1


									plt.savefig(outputData + "/boxplotOnset.png", bbox_inches = 'tight', dpi = DPI)
									plt.savefig(outputData + "/boxplotOnset.svg", bbox_inches = 'tight', dpi = DPI)
									plt.close()


									fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
									plt.grid()

									# dataStats = {}
									# index = 0
									# for data in dataBoxPlotLength :
									# 	# print(data)
									# 	curDataStats = list()
									# 	for dataCmp in dataBoxPlotLength :
									# 		# print(dataCmp)
									# 		U, P = stats.mannwhitneyu(data, dataCmp)
									# 		stars = "-"
									# 		if P < 0.0001:
									# 		   stars = "****"
									# 		elif (P < 0.001):
									# 		   stars = "***"
									# 		elif (P < 0.01):
									# 		   stars = "**"
									# 		elif (P < 0.05):
									# 		   stars = "*"

									# 		curDataStats.append(stars)
									# 	dataStats[index] = curDataStats
									# 	index += 1

									# for i in range(0, len(dataStats)) :
									# 	print("Data " + str(i) + " : ")
									# 	for j in range(0, len(dataStats[i])) :
									# 		print("Data " + str(j) + " : P = " + str(dataStats[i][j]))

									bp = ax.boxplot(dataBoxPlotLength)

									# ax.spines['top'].set_visible(False)
									# ax.spines['right'].set_visible(False)
									# ax.spines['left'].set_visible(False)
									# ax.get_xaxis().tick_bottom()
									# ax.get_yaxis().tick_left()
									# ax.tick_params(axis='x', direction='out')
									# ax.tick_params(axis='y', length=0)

									ax.set_xlabel("Trial")
									ax.set_ylabel("Length")
									ax.set_ylim(0, 100)

									x_ticks = range(0, len(listTrials))
									x_ticksLabels = listTrials
									ax.set_xticklabels(x_ticksLabels)

									# ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
									# ax.set_axisbelow(True)

									for i in range(0, len(bp['boxes'])):
										bp['boxes'][i].set_color(palette[i])
										# we have two whiskers!
										bp['whiskers'][i*2].set_color(palette[i])
										bp['whiskers'][i*2 + 1].set_color(palette[i])
										bp['whiskers'][i*2].set_linewidth(2)
										bp['whiskers'][i*2 + 1].set_linewidth(2)
										# top and bottom fliers
										# (set allows us to set many parameters at once)
										# bp['fliers'][i].set(markerfacecolor=palette[1],
										#                 marker='o', alpha=0.75, markersize=6,
										#                 markeredgecolor='none')
										# bp['fliers'][1].set(markerfacecolor=palette[1],
										#                 marker='o', alpha=0.75, markersize=6,
										#                 markeredgecolor='none')
										if (i * 2) < len(bp['fliers']) :
										   bp['fliers'][i * 2].set(markerfacecolor='black',
										                   marker='o', alpha=0.75, markersize=6,
										                   markeredgecolor='none')
										if (i * 2 + 1) < len(bp['fliers']) :
										   bp['fliers'][i * 2 + 1].set(markerfacecolor='black',
										                   marker='o', alpha=0.75, markersize=6,
										                   markeredgecolor='none')
										bp['medians'][i].set_color('black')
										bp['medians'][i].set_linewidth(3)
										# and 4 caps to remove
										# for c in bp['caps']:
										#    c.set_linewidth(0)

									for i in range(len(bp['boxes'])):
									   box = bp['boxes'][i]
									   box.set_linewidth(0)
									   boxX = []
									   boxY = []
									   for j in range(5):
									       boxX.append(box.get_xdata()[j])
									       boxY.append(box.get_ydata()[j])
									       boxCoords = zip(boxX,boxY)
									       boxPolygon = Polygon(boxCoords, facecolor = palette[i], linewidth=0)
									       ax.add_patch(boxPolygon)

									# fig.subplots_adjust(left=0.2)

									# y_max = np.max(np.concatenate(np.concatenate(dataBoxPlot[0][:], dataBoxPlot[1][:]), dataBoxPlot[2][:]))
									# y_min = np.min(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
									# y_min = 0

									# print y_max

									# Print stats
									# cptData = 0
									# while cptData < len(dataStats) :
									# 	cptComp = cptData + 1
									# 	while cptComp < len(dataStats[cptData]) :
									# 		if dataStats[cptData][cptComp] != "-" :
									# 			if cptData == 0 and cptComp == 2 :
									# 				ax.annotate("", xy=(cptData + 1, 7300), xycoords='data',
									# 				           xytext=(cptComp + 1, 7300), textcoords='data',
									# 				           arrowprops=dict(arrowstyle="-", ec='#000000',
									# 				                           connectionstyle="bar,fraction=0.05"))
									# 				ax.text((cptComp - cptData)/2 + cptData + 1, 7725, dataStats[cptData][cptComp],
									# 				       horizontalalignment='center',
									# 				       verticalalignment='center')
									# 			else :
									# 				ax.annotate("", xy=(cptData + 1, 7000), xycoords='data',
									# 				           xytext=(cptComp + 1, 7000), textcoords='data',
									# 				           arrowprops=dict(arrowstyle="-", ec='#000000',
									# 				                           connectionstyle="bar,fraction=0.05"))
									# 				ax.text(float(cptComp - cptData)/2 + cptData + 1, 7250, dataStats[cptData][cptComp],
									# 				       horizontalalignment='center',
									# 				       verticalalignment='center')

									# 		cptComp += 1
									# 	cptData += 1


									plt.savefig(outputData + "/boxplotLength.png", bbox_inches = 'tight', dpi = DPI)
									plt.savefig(outputData + "/boxplotLength.svg", bbox_inches = 'tight', dpi = DPI)
									plt.close()

									# ----- FREQUENCY OF TARGETS FOUND -----
									print('\t\t-> Drawing frequency of targets found...')


									fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
									plt.grid()

									listTargetsInd = [0, 0, 0, 0, 0, 0]
									nbIndividuals = 0
									for ind in indStats.keys() :
										targetsFound = sum([1 for found in indStats[ind]['targetFound'] if found])
										listTargetsInd[targetsFound] += 1
										nbIndividuals += 1
		

									listTargetsInd = map(lambda x : x*100./float(nbIndividuals), listTargetsInd)

									ax.bar([x + MARGIN_BARS_DIFF for x in range(len(listTargetsInd))], listTargetsInd, width = WIDTH_BARS_DIFF, color = palette[1])

									xticks = [x + 0.5 for x in range(len(listTargetsInd))]
									xticksLabels = range(0, len(listTargetsInd))
									ax.set_xticks(xticks)
									ax.set_xticklabels(xticksLabels)
									ax.set_xlabel("Number of food sources found")

									ax.set_ylabel("Percentage of individuals (%)")
									ax.set_ylim(0, 100)

									plt.savefig(outputData + "/barTargetsFound.png", bbox_inches = 'tight', dpi = DPI)
									plt.savefig(outputData + "/barTargetsFound.svg", bbox_inches = 'tight', dpi = DPI)
									plt.close()


									# # ----- GLOBAL STATS ON REPLICATE -----

									# print("Printing summary info file...")
									# with open(outputData + "/fileInfo.txt", "w") as fileWrite :
									# 	fileWrite.write("Exp\n")
									# 	for replicate in hashSigDiff.keys() :
									# 		fileWrite.write(str(np.mean(hashSigDiff[replicate])) + "\n")
									# 			for ind in indStats.keys()


						# ----- FREQUENCY OF TARGETS FOUND -----
						print('\t\t-> Drawing frequency of targets found...')

						if args.length :
							outputData = os.path.join(OUTPUT_DIR, "Length")
						elif args.onset :
							outputData = os.path.join(OUTPUT_DIR, "Onset")
						else :
							outputData = os.path.join(OUTPUT_DIR, "Unconstrained")

						fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
						plt.grid()

						listTargetsInd = [0, 0, 0, 0, 0, 0]
						nbIndividuals = 0
						for replicate in hashTargetsFound.keys() :
							for cpt in range(0, len(hashTargetsFound[replicate])) :
								listTargetsInd[hashTargetsFound[replicate][cpt]] += 1
								nbIndividuals += 1

						listTargetsInd = map(lambda x : x*100./float(nbIndividuals), listTargetsInd)

						print("Proportion of foraging sites found :")
						for cptTargets in range(0, len(listTargetsInd)) :
							print("\t " + str(cptTargets) + " targets : " + str(listTargetsInd[cptTargets]))

						ax.bar([x + MARGIN_BARS_DIFF for x in range(len(listTargetsInd))], listTargetsInd, width = WIDTH_BARS_DIFF, color = palette[0])

						xticks = [x + MARGIN_BARS_DIFF for x in range(len(listTargetsInd))]
						xticksLabels = range(0, len(listTargetsInd))
						ax.set_xticks(xticks)
						ax.set_xticklabels(xticksLabels)
						ax.set_xlabel("Number of food sources found")

						ax.set_ylabel("Percentage of individuals (%)")
						ax.set_ylim(0, 100)

						plt.savefig(outputData + "/barTargetsFound.png", bbox_inches = 'tight', dpi = DPI)
						plt.savefig(outputData + "/barTargetsFound.svg", bbox_inches = 'tight', dpi = DPI)
						plt.close()

						fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
						plt.grid()

						# dataStats = {}
						# index = 0
						# for data in dataBoxPlotLength :
						# 	# print(data)
						# 	curDataStats = list()
						# 	for dataCmp in dataBoxPlotLength :
						# 		# print(dataCmp)
						# 		U, P = stats.mannwhitneyu(data, dataCmp)
						# 		stars = "-"
						# 		if P < 0.0001:
						# 		   stars = "****"
						# 		elif (P < 0.001):
						# 		   stars = "***"
						# 		elif (P < 0.01):
						# 		   stars = "**"
						# 		elif (P < 0.05):
						# 		   stars = "*"

						# 		curDataStats.append(stars)
						# 	dataStats[index] = curDataStats
						# 	index += 1

						# for i in range(0, len(dataStats)) :
						# 	print("Data " + str(i) + " : ")
						# 	for j in range(0, len(dataStats[i])) :
						# 		print("Data " + str(j) + " : P = " + str(dataStats[i][j]))

						dataBoxPlotSignal = [np.mean(hashSigDiff[replicate]) for replicate in hashSigDiff.keys()]

						bp = ax.boxplot(dataBoxPlotSignal)

						print("Mean signal variation : s = " + str(np.mean(dataBoxPlotSignal)) + ", var = " + str(np.var(dataBoxPlotSignal)))

						# print(np.mean(dataBoxPlotSignal))
						# res = stats.ttest_1samp(dataBoxPlotSignal, 0.0)
						# res = stats.wilcoxon(dataBoxPlotSignal)

						# ax.spines['top'].set_visible(False)
						# ax.spines['right'].set_visible(False)
						# ax.spines['left'].set_visible(False)
						# ax.get_xaxis().tick_bottom()
						# ax.get_yaxis().tick_left()
						# ax.tick_params(axis='x', direction='out')
						# ax.tick_params(axis='y', length=0)

						ax.set_xlabel("Setting")
						ax.set_ylabel("Signal variation")
						ax.set_ylim(0, 1.0)

						x_ticks = range(0, len(listTrials))
						x_ticksLabels = listTrials
						ax.set_xticklabels(x_ticksLabels)

						# ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
						# ax.set_axisbelow(True)

						for i in range(0, len(bp['boxes'])):
							bp['boxes'][i].set_color(palette[i])
							# we have two whiskers!
							bp['whiskers'][i*2].set_color(palette[i])
							bp['whiskers'][i*2 + 1].set_color(palette[i])
							bp['whiskers'][i*2].set_linewidth(2)
							bp['whiskers'][i*2 + 1].set_linewidth(2)
							# top and bottom fliers
							# (set allows us to set many parameters at once)
							# bp['fliers'][i].set(markerfacecolor=palette[1],
							#                 marker='o', alpha=0.75, markersize=6,
							#                 markeredgecolor='none')
							# bp['fliers'][1].set(markerfacecolor=palette[1],
							#                 marker='o', alpha=0.75, markersize=6,
							#                 markeredgecolor='none')
							if (i * 2) < len(bp['fliers']) :
							   bp['fliers'][i * 2].set(markerfacecolor='black',
							                   marker='o', alpha=0.75, markersize=6,
							                   markeredgecolor='none')
							if (i * 2 + 1) < len(bp['fliers']) :
							   bp['fliers'][i * 2 + 1].set(markerfacecolor='black',
							                   marker='o', alpha=0.75, markersize=6,
							                   markeredgecolor='none')
							bp['medians'][i].set_color('black')
							bp['medians'][i].set_linewidth(3)
							# and 4 caps to remove
							# for c in bp['caps']:
							#    c.set_linewidth(0)

						for i in range(len(bp['boxes'])):
						   box = bp['boxes'][i]
						   box.set_linewidth(0)
						   boxX = []
						   boxY = []
						   for j in range(5):
						       boxX.append(box.get_xdata()[j])
						       boxY.append(box.get_ydata()[j])
						       boxCoords = zip(boxX,boxY)
						       boxPolygon = Polygon(boxCoords, facecolor = palette[i], linewidth=0)
						       ax.add_patch(boxPolygon)

						# fig.subplots_adjust(left=0.2)

						# y_max = np.max(np.concatenate(np.concatenate(dataBoxPlot[0][:], dataBoxPlot[1][:]), dataBoxPlot[2][:]))
						# y_min = np.min(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
						# y_min = 0

						# print y_max

						# Print stats
						# cptData = 0
						# while cptData < len(dataStats) :
						# 	cptComp = cptData + 1
						# 	while cptComp < len(dataStats[cptData]) :
						# 		if dataStats[cptData][cptComp] != "-" :
						# 			if cptData == 0 and cptComp == 2 :
						# 				ax.annotate("", xy=(cptData + 1, 7300), xycoords='data',
						# 				           xytext=(cptComp + 1, 7300), textcoords='data',
						# 				           arrowprops=dict(arrowstyle="-", ec='#000000',
						# 				                           connectionstyle="bar,fraction=0.05"))
						# 				ax.text((cptComp - cptData)/2 + cptData + 1, 7725, dataStats[cptData][cptComp],
						# 				       horizontalalignment='center',
						# 				       verticalalignment='center')
						# 			else :
						# 				ax.annotate("", xy=(cptData + 1, 7000), xycoords='data',
						# 				           xytext=(cptComp + 1, 7000), textcoords='data',
						# 				           arrowprops=dict(arrowstyle="-", ec='#000000',
						# 				                           connectionstyle="bar,fraction=0.05"))
						# 				ax.text(float(cptComp - cptData)/2 + cptData + 1, 7250, dataStats[cptData][cptComp],
						# 				       horizontalalignment='center',
						# 				       verticalalignment='center')

						# 		cptComp += 1
						# 	cptData += 1


						plt.savefig(outputData + "/boxplotSignal.png", bbox_inches = 'tight', dpi = DPI)
						plt.savefig(outputData + "/boxplotSignal.svg", bbox_inches = 'tight', dpi = DPI)
						plt.close()

						print("Printing summary info file...")
						with open(outputData + "/fileInfo.txt", "w") as fileWrite :
							fileWrite.write("Exp\n")
							for replicate in hashSigDiff.keys() :
								fileWrite.write("NbIndividualsStrat = " + str(nbIndStratTot) + "/StratOneSite = " + str(nbIndStratOneSiteTot) + "/StratAllSites = " + str(nbIndStratAllSitesTot) + "\n")
								fileWrite.write(str(np.mean(hashSigDiff[replicate])) + "\n")


def inCommArea(pos) :
	if pos >= -math.pi/4.0 and pos <= math.pi/4.0 :
		return True
	else :
		return False


def transform2Pi(listTargets) :
	for cpt in range(0, len(listTargets)) :
		if listTargets[cpt] < 0.0 :
			listTargets[cpt] = listTargets[cpt] + 2.*math.pi

	return listTargets


def inTargetArea(pos, target) :
	if target in TARGETS.keys() :
		if (pos >= TARGETS[target][0][0] and pos <= TARGETS[target][0][1]) or (pos >= TARGETS[target][1][0] and pos <= TARGETS[target][1][1]) :
			return True
		else :
			return False
	else :
		print('Target ' + str(target) + ' unknown !')
		return False

def differenceSignal(signal1, signal2) :
	diff = 0
	assert(len(signal1) == len(signal2))

	for step in range(0, len(signal1)) :
		diff += math.fabs(signal1[step] - signal2[step])
	return diff/float(len(signal1))





if __name__ == "__main__" :
	parser = argparse.ArgumentParser()

	# action = parser.add_mutually_exclusive_group(required = True)
	parser.add_argument('-d', '--directories', help = "Directories to plot", type=str, nargs='+')

	parser.add_argument('-l', '--load', help = "Load serialized data rather than read behaviour files", type=str, default = None)
	parser.add_argument('-r', '--replicate', help = "Check for particular replicate", default = None, type = str, nargs='*')
	parser.add_argument('-e', '--length', help = "Length strategy", default = False, action = 'store_true')
	parser.add_argument('-o', '--onset', help = "Onset strategy", default = False, action = 'store_true')
	parser.add_argument('-s', '--serialize', help = "Serialize data", default = None)
	# parser.add_argument('-c', '--check', help = "Check everything and store it in file", default = False, action = 'store_true')
	
	args = parser.parse_args()

	main(args)
