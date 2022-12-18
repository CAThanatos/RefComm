#!/usr/bin/env python

import cv2
import matplotlib

# For ssh compatibility
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import lines
import seaborn as sns
import numpy as np

from pylab import Polygon
from scipy import stats

import os
import argparse
import re
import math
import shutil
import brewer2mpl
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# GLOBAL PARAMETERS
RECURSIVE = False
OUTPUT_DIR = "./OutputGraphs"
TRIALS = [1,2,3,4,5]
PRECISION = 100
MAX = -1
REPLICATES = None

BAR_TO_SHOW = ["onset", "length4"]
# BAR_TO_SHOW = ["onset"]
# BAR_TO_SHOW = ["length4"]


# BAR_TO_SHOW = ["length"]
# BAR_TO_SHOW = ["length2"]
# BAR_TO_SHOW = ["length3"]
# BAR_TO_SHOW = ["length4"]
# BAR_TO_SHOW = ["onset", "length3"]
# BAR_TO_SHOW = ["onset", "length", "length2", "length3"]
NUM_BAR = {
						"onset" : 0,
						"length" : 1,
						"length2" : 2,
						"length3" : 3,
						"length4" : 4
}
# MARGIN_BARS = 0.1
MARGIN_BARS_DIFF = 0.2
WIDTH_BARS_DIFF = (1.-2.*MARGIN_BARS_DIFF)/float(len(BAR_TO_SHOW) + 1)
MARGIN_BARS_RATIO = 0.3
WIDTH_BARS_RATIO = (1.-2.*MARGIN_BARS_RATIO)/float(len(BAR_TO_SHOW))
# WIDTH_BARS = (1.-2.*MARGIN_BARS)/4.
# WIDTH_BARS = (1.-2.*MARGIN_BARS)/3.
# WIDTH_BARS = (1.-2.*MARGIN_BARS)/2.
# WIDTH_BARS = 0.5
# NB_REPLICATE_PER_PLOT = 20
NB_REPLICATE_PER_PLOT = 40

RUNS_ONSET = [1, 2, 3, 4, 7, 9 ,10, 11, 13, 17, 19, 21, 26, 28, 30, 32, 38, 39]
RUNS_LENGTH = [12, 29, 31, 37]
RUNS_BOTH = [0,5, 8, 14, 15, 16, 18, 20,23,24, 25, 33, 34, 35, 36]

# FOLDER_INFO = "BehaviorsAnalyses"
FOLDER_INFO = "BehaviourAnalyses"

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
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelsize'] = 15
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['legend.fontsize'] = 15
matplotlib.rcParams['mathtext.default'] = 'regular'

# GRAPHS GLOBAL VARIABLES
TARGET_COLOR = '#B22400'
COMM_AREA_COLOR ='#006BB2'
linewidth = 2
# linestyles = ['-', '--', '-.']
linestyles = ['-', '-', '-', '-']
markers = [None, None, None, None] #['o', '+', '*']

VIDEO_SIZE = (750, 750)
FPS = 20

DPI = 96
# size = (1920/dpi, 1536/dpi)
FIGSIZE = (1280/DPI, 1024/DPI)
# FIGSIZE = (1024/DPI, 744/DPI)
# size = (880, 640)

def drawGenThreshold(directories, threshold) :
	hashData = {}
	regRep = re.compile(r"^(\d+)$")

	print('Reading data...')
	for d in directories :
		if os.path.isdir(d) :
			if RECURSIVE :
				tuplesWalk = os.walk(d)

				for t in tuplesWalk :
					listReplicates = [replicate for replicate in t[1] if regRep.match(replicate)]

					if len(listReplicates) > 0 :
						dictReplicates = {}
						tabGenerations = list()
						for replicate in listReplicates :
							m = regRep.match(replicate)

							replicateDir = os.path.join(t[0], replicate)
							numReplicate = int(m.group(1))

							dictGenerations = {}
							if os.path.isfile(os.path.join(replicateDir, 'generations.sender.dat')) :
								with open(os.path.join(replicateDir, 'generations.sender.dat'), 'r') as fileRead :
									fileRead = fileRead.readlines()

									for line in fileRead :
										dictCurGen = {}
										lineSplit = line.split(' ')

										if len(lineSplit) > 0 :
											gen = int(lineSplit[0])

											if gen%PRECISION == 0 and (MAX == -1 or gen <= MAX):
												dictCurGen['meanFit'] = float(lineSplit[1])
												dictCurGen['maxFit'] = float(lineSplit[2])
												dictCurGen['minFit'] = float(lineSplit[3])
												dictCurGen['sd'] = float(lineSplit[4])

												dictGenerations[gen] = dictCurGen

												if gen not in tabGenerations :
													tabGenerations.append(gen)


							dictReplicates[numReplicate] = dictGenerations

						hashData[os.path.basename(t[0])] = (dictReplicates, tabGenerations)
			else :
				listReplicates = [replicate for replicate in os.listdir(d) if regRep.match(replicate)]

				if len(listReplicates) > 0 :
					dictReplicates = {}
					tabGenerations = list()
					for replicate in listReplicates :
						m = regRep.match(replicate)

						replicateDir = os.path.join(d, replicate)
						numReplicate = int(m.group(1))

						dictGenerations = {}
						if os.path.isfile(os.path.join(replicateDir, 'generations.sender.dat')) :
							with open(os.path.join(replicateDir, 'generations.sender.dat'), 'r') as fileRead :
								fileRead = fileRead.readlines()

								for line in fileRead :
									dictCurGen = {}
									lineSplit = line.split(' ')

									if len(lineSplit) > 0 :
										gen = int(lineSplit[0])

										if gen%PRECISION == 0 and (MAX == -1 or gen <= MAX):
											dictCurGen['meanFit'] = float(lineSplit[1])
											dictCurGen['maxFit'] = float(lineSplit[2])
											dictCurGen['minFit'] = float(lineSplit[3])
											dictCurGen['sd'] = float(lineSplit[4])

											dictGenerations[gen] = dictCurGen

											if gen not in tabGenerations :
												tabGenerations.append(gen)


						dictReplicates[numReplicate] = dictGenerations

					hashData[os.path.basename(d)] = (dictReplicates, tabGenerations)



	# ----- PERFORMANCE THRESHOLD BOXPLOTS -----
	print('Drawing performance threshold boxplots...')

	outputData = OUTPUT_DIR

	if os.path.isdir(outputData) :
		shutil.rmtree(outputData)

	os.makedirs(outputData)

	dataBoxPlot = []
	x_ticksLabels = []
	for exp in sorted(hashData.keys()) :
		listGenThreshold = list()
		for replicate in hashData[exp][0].keys() :
			genThreshold = -1
			found = False
			for generation in sorted(hashData[exp][0][replicate].keys()) :
				if hashData[exp][0][replicate][generation]['meanFit'] > threshold :
					found = True
					break
				else :
					genThreshold = generation

			if genThreshold == -1 :
				print('wat')

			if not found :
				print('Threshold not found for ' + exp + ', replicate ' + str(replicate))

			listGenThreshold.append(genThreshold)

		print(np.mean(listGenThreshold))
		dataBoxPlot.append(listGenThreshold)
		x_ticksLabels.append(exp)

		print("Min generation : " + str(np.min(listGenThreshold)) + "\tMax generation : " + str(np.max(listGenThreshold)))
		print(exp + " -> Mean number of generations : g = " + str(np.mean(listGenThreshold)) + ", var = " + str(np.var(listGenThreshold)))

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
	plt.grid()
	# palette = sns.color_palette("husl", len(hashData.keys()))

	dataStats = {}
	index = 0
	for data in dataBoxPlot :
		# print(data)
		curDataStats = list()
		for dataCmp in dataBoxPlot :
			# print(dataCmp)
			U, P = stats.mannwhitneyu(data, dataCmp)
			stars = "n.s."
			if P < 0.0001:
			   stars = "****"
			elif (P < 0.001):
			   stars = "***"
			elif (P < 0.01):
			   stars = "**"
			elif (P < 0.05):
			   stars = "*"

			curDataStats.append(stars)
		dataStats[index] = curDataStats
		index += 1

	# for i in range(0, len(dataStats)) :
	# 	print("Data " + str(i) + " : ")
	# 	for j in range(0, len(dataStats[i])) :
	# 		print("Data " + str(j) + " : P = " + str(dataStats[i][j]))

	bp = ax.boxplot(dataBoxPlot)

	# ax.spines['top'].set_visible(False)
	# ax.spines['right'].set_visible(False)
	# ax.spines['left'].set_visible(False)
	# ax.get_xaxis().tick_bottom()
	# ax.get_yaxis().tick_left()
	# ax.tick_params(axis='x', direction='out')
	# ax.tick_params(axis='y', length=0)

	ax.set_xlabel("Setting")
	ax.set_ylabel("Generations")
	ax.set_ylim(0, 8000)

	# x_ticksLabels = ['Unconstrained', 'Constant signal', 'Constant velocity']			
	x_ticksLabels = ['Unconstrained', 'No onset/length variation']			
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
	cptData = 0
	while cptData < len(dataStats) :
		cptComp = cptData + 1
		while cptComp < len(dataStats[cptData]) :
			if dataStats[cptData][cptComp] != "-" :
				if cptData == 0 and cptComp == 2 :
					ax.annotate("", xy=(cptData + 1, 7300), xycoords='data',
					           xytext=(cptComp + 1, 7300), textcoords='data',
					           arrowprops=dict(arrowstyle="-", ec='#000000',
					                           connectionstyle="bar,fraction=0.05"))
					ax.text((cptComp - cptData)/2 + cptData + 1, 7725, dataStats[cptData][cptComp],
					       horizontalalignment='center',
					       verticalalignment='center')
				else :
					ax.annotate("", xy=(cptData + 1, 7000), xycoords='data',
					           xytext=(cptComp + 1, 7000), textcoords='data',
					           arrowprops=dict(arrowstyle="-", ec='#000000',
					                           connectionstyle="bar,fraction=0.05"))
					ax.text(float(cptComp - cptData)/2 + cptData + 1, 7250, dataStats[cptData][cptComp],
					       horizontalalignment='center',
					       verticalalignment='center')

			cptComp += 1
		cptData += 1


	plt.savefig(outputData + "/boxplotGenThreshold.png", bbox_inches = 'tight', dpi = DPI)
	plt.savefig(outputData + "/boxplotGenThreshold.svg", bbox_inches = 'tight', dpi = DPI)
	plt.close()


	# Fitness Boxplots Stags
	fig, axe1 = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
	# plt.axes(frameon=0)
	plt.grid()

	cptExp = 0
	listExp = sorted(hashData.keys())
	for cptExp in range(len(listExp)) :
		exp = listExp[cptExp]
		# if cptExp == 1 :
		if False:
			dataExp = hashData[exp][0]
			tabGen = sorted(hashData[exp][1])

			cptReplicate = 0
			for replicate in dataExp.keys() :
				dataPlot = [dataExp[replicate][generation]['meanFit'] for generation in tabGen if generation < 10001]
				axe1.plot(range(len(dataPlot)), dataPlot, color = palette[cptExp], linestyle = '-', linewidth = 4)
				cptReplicate += 1
		else :
			dataExp = hashData[exp][0]
			tabGen = sorted(hashData[exp][1])

			dataPlot = []
			# dataPerc25 = []
			# dataPerc75 = []
			for generation in tabGen :
				if generation < 10001 :
					fitnessGen = [dataExp[replicate][generation]['meanFit'] for replicate in dataExp.keys()]
					fitnessMed = np.median(fitnessGen)
					# fitnessMed = np.mean(fitnessGen)

					# perc25 = fitnessMed
					# perc75 = fitnessMed
					# if len(fitnessGen) > 1 :
					# 	perc25 = np.percentile(fitnessGen, 25)
					# 	perc75 = np.percentile(fitnessGen, 75)

					dataPlot.append(fitnessMed)
					# dataPerc25.append(perc25)
					# dataPerc75.append(perc75)

			# print(exp + " -> Mean fitness at generation " + str(tabGen[-1]) + " : m = " + str(np.mean([dataExp[replicate][tabGen[-1]]['meanFit'] for replicate in dataExp.keys()])) + ", var = " + str(np.var([dataExp[replicate][tabGen[-1]]['meanFit'] for replicate in dataExp.keys()])))
			# print(str([dataExp[replicate][tabGen[-1]]['meanFit'] for replicate in dataExp.keys()]))

				# cpt = 0
				# while cpt < len(dataPlot) :
				# 	if math.isnan(dataPlot[cpt]) :
				# 		if cpt > 0 and cpt < len(dataPlot) - 1 :
				# 			dataPlot[cpt] = (dataPlot[cpt + 1] + dataPlot[cpt - 1])/2
				# 			dataPerc25[cpt] = (dataPerc25[cpt + 1] + dataPerc25[cpt - 1])/2
				# 			dataPerc75[cpt] = (dataPerc75[cpt + 1] + dataPerc75[cpt - 1])/2
				# 		elif cpt > 0 :
				# 			dataPlot[cpt] = dataPlot[cpt - 1]
				# 			dataPerc25[cpt] = dataPerc25[cpt - 1]
				# 			dataPerc75[cpt] = dataPerc75[cpt - 1]
				# 		else :
				# 			dataPlot[cpt] = dataPlot[cpt + 1]
				# 			dataPerc25[cpt] = dataPerc25[cpt + 1]
				# 			dataPerc75[cpt] = dataPerc75[cpt + 1]

				# 	cpt += 1

			axe1.plot(range(len(dataPlot)), dataPlot, color = palette[cptExp], linestyle = '-', linewidth = 4)
			# plt.fill_between(range(len(dataPlot)), dataPerc25, dataPerc75, alpha=0.25, linewidth=0, color=palette[cptExp])

		# 	dataPlot = []
		# 	dataPerc25 = []
		# 	dataPerc75 = []
		# 	for generation in tabGen :
		# 		fitnessGen = [dataExp[replicate][generation]['maxFit'] for replicate in dataExp.keys()]
		# 		fitnessMed = np.median(fitnessGen)

		# 		perc25 = fitnessMed
		# 		perc75 = fitnessMed
		# 		if len(fitnessGen) > 1 :
		# 			perc25 = np.percentile(fitnessGen, 25)
		# 			perc75 = np.percentile(fitnessGen, 75)

		# 		dataPlot.append(fitnessMed)
		# 		dataPerc25.append(perc25)
		# 		dataPerc75.append(perc75)

		# 		# cpt = 0
		# 		# while cpt < len(dataPlot) :
		# 		# 	if math.isnan(dataPlot[cpt]) :
		# 		# 		if cpt > 0 and cpt < len(dataPlot) - 1 :
		# 		# 			dataPlot[cpt] = (dataPlot[cpt + 1] + dataPlot[cpt - 1])/2
		# 		# 			dataPerc25[cpt] = (dataPerc25[cpt + 1] + dataPerc25[cpt - 1])/2
		# 		# 			dataPerc75[cpt] = (dataPerc75[cpt + 1] + dataPerc75[cpt - 1])/2
		# 		# 		elif cpt > 0 :
		# 		# 			dataPlot[cpt] = dataPlot[cpt - 1]
		# 		# 			dataPerc25[cpt] = dataPerc25[cpt - 1]
		# 		# 			dataPerc75[cpt] = dataPerc75[cpt - 1]
		# 		# 		else :
		# 		# 			dataPlot[cpt] = dataPlot[cpt + 1]
		# 		# 			dataPerc25[cpt] = dataPerc25[cpt + 1]
		# 		# 			dataPerc75[cpt] = dataPerc75[cpt + 1]

		# 		# 	cpt += 1

		# axe1.plot(range(len(dataPlot)), dataPlot, color = palette[1], linestyle = '-', linewidth = 4)
		# plt.fill_between(range(len(dataPlot)), dataPerc25, dataPerc75, alpha=0.25, linewidth=0, color=palette[1])

		cptExp += 1

	xticks = range(0, len(dataPlot), len(dataPlot)/5)
	# xticks = range(0, len(dataPlot), len(dataPlot)/10)
	if len(dataPlot) - 1 not in xticks :
		xticks.append(len(dataPlot) - 1)
	xticksLabels = [tabGen[t] for t in xticks]
	
	axe1.set_xticks(xticks)
	axe1.set_xticklabels(xticksLabels)
	axe1.set_xlabel("Generations")

	axe1.set_ylabel("Performance")
	axe1.set_ylim(0, 1.0)

	# legend = plt.legend(['Unconstrained', 'Constant signal', 'Constant velocity'], loc = 4, frameon=True)
	legend = plt.legend(['Unconstrained', 'Constrained\nNo onset/length variation'], loc = 4, frameon=True)
	# legend = plt.legend(['Average', 'Best'], loc = 4, frameon=True)
	# legend = plt.legend(['Unconstrained', 'No signaling'], loc = 1, frameon=True)
	# frame = legend.get_frame()
	# frame.set_facecolor('0.9')
	# frame.set_edgecolor('0.9')

	plt.savefig(outputData + "/avgPerformance.png", bbox_inches = 'tight', dpi = DPI)
	plt.savefig(outputData + "/avgPerformance.svg", bbox_inches = 'tight', dpi = DPI)
	plt.close()



	listGenOvercome = list()
	for replicate in hashData[listExp[0]][0].keys() :
		for generation in sorted(hashData[listExp[0]][0][replicate].keys()) :
			meanCtrl = np.mean([hashData[listExp[0]][0][replicate2][generation]['meanFit'] for replicate2 in hashData[listExp[0]][0].keys()])

			if generation > 2000 :
				if hashData[listExp[1]][0][replicate][generation]['meanFit'] > meanCtrl :
					print(generation)
					listGenOvercome.append(generation)
					break

	print("Mean number of generations : g = " + str(np.mean(listGenOvercome)) + ", var = " + str(np.var(listGenOvercome)))


def drawAvgPerf(directories) :
	hashData = {}
	regRep = re.compile(r"^(\d+)$")

	print('Reading data...')
	for d in directories :
		if os.path.isdir(d) :
			if RECURSIVE :
				tuplesWalk = os.walk(d)

				for t in tuplesWalk :
					listReplicates = [replicate for replicate in t[1] if regRep.match(replicate)]

					if len(listReplicates) > 0 :
						dictReplicates = {}
						tabGenerations = list()
						for replicate in listReplicates :
							m = regRep.match(replicate)

							replicateDir = os.path.join(t[0], replicate)
							numReplicate = int(m.group(1))

							dictGenerations = {}
							if os.path.isfile(os.path.join(replicateDir, 'generations.sender.dat')) :
								with open(os.path.join(replicateDir, 'generations.sender.dat'), 'r') as fileRead :
									fileRead = fileRead.readlines()

									for line in fileRead :
										dictCurGen = {}
										lineSplit = line.split(' ')

										if len(lineSplit) > 0 :
											gen = int(lineSplit[0])

											if gen%PRECISION == 0 and (MAX == -1 or gen <= MAX):
												dictCurGen['meanFit'] = float(lineSplit[1])
												dictCurGen['maxFit'] = float(lineSplit[2])
												dictCurGen['minFit'] = float(lineSplit[3])
												dictCurGen['sd'] = float(lineSplit[4])

												dictGenerations[gen] = dictCurGen

												if gen not in tabGenerations :
													tabGenerations.append(gen)


							dictReplicates[numReplicate] = dictGenerations

						hashData[os.path.basename(t[0])] = (dictReplicates, tabGenerations)
			else :
				listReplicates = [replicate for replicate in os.listdir(d) if regRep.match(replicate)]

				if len(listReplicates) > 0 :
					dictReplicates = {}
					tabGenerations = list()
					for replicate in listReplicates :
						m = regRep.match(replicate)

						replicateDir = os.path.join(d, replicate)
						numReplicate = int(m.group(1))

						dictGenerations = {}
						if os.path.isfile(os.path.join(replicateDir, 'generations.sender.dat')) :
							with open(os.path.join(replicateDir, 'generations.sender.dat'), 'r') as fileRead :
								fileRead = fileRead.readlines()

								for line in fileRead :
									dictCurGen = {}
									lineSplit = line.split(' ')

									if len(lineSplit) > 0 :
										gen = int(lineSplit[0])

										if gen%PRECISION == 0 and (MAX == -1 or gen <= MAX):
											dictCurGen['meanFit'] = float(lineSplit[1])
											dictCurGen['maxFit'] = float(lineSplit[2])
											dictCurGen['minFit'] = float(lineSplit[3])
											dictCurGen['sd'] = float(lineSplit[4])

											dictGenerations[gen] = dictCurGen

											if gen not in tabGenerations :
												tabGenerations.append(gen)


						dictReplicates[numReplicate] = dictGenerations

					hashData[os.path.basename(d)] = (dictReplicates, tabGenerations)



	# ----- AVERAGE FITNESS PLOTS -----
	print('Drawing average performance plots...')

	outputData = OUTPUT_DIR

	if os.path.isdir(outputData) :
		shutil.rmtree(outputData)

	os.makedirs(outputData)

	# Fitness Boxplots Stags
	fig, axe1 = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
	# plt.axes(frameon=0)
	plt.grid()

	cptExp = 0
	for exp in sorted(hashData.keys()) :
		dataExp = hashData[exp][0]
		tabGen = sorted(hashData[exp][1])

		dataPlot = []
		dataPerc25 = []
		dataPerc75 = []
		for generation in tabGen :
			fitnessGen = [dataExp[replicate][generation]['meanFit'] for replicate in dataExp.keys()]
			fitnessMed = np.median(fitnessGen)

			perc25 = fitnessMed
			perc75 = fitnessMed
			if len(fitnessGen) > 1 :
				perc25 = np.percentile(fitnessGen, 25)
				perc75 = np.percentile(fitnessGen, 75)

			dataPlot.append(fitnessMed)
			dataPerc25.append(perc25)
			dataPerc75.append(perc75)

		print(exp + " -> Mean fitness at generation " + str(tabGen[-1]) + " : m = " + str(np.mean([dataExp[replicate][tabGen[-1]]['meanFit'] for replicate in dataExp.keys()])) + ", var = " + str(np.var([dataExp[replicate][tabGen[-1]]['meanFit'] for replicate in dataExp.keys()])))
		# print(str([dataExp[replicate][tabGen[-1]]['meanFit'] for replicate in dataExp.keys()]))

			# cpt = 0
			# while cpt < len(dataPlot) :
			# 	if math.isnan(dataPlot[cpt]) :
			# 		if cpt > 0 and cpt < len(dataPlot) - 1 :
			# 			dataPlot[cpt] = (dataPlot[cpt + 1] + dataPlot[cpt - 1])/2
			# 			dataPerc25[cpt] = (dataPerc25[cpt + 1] + dataPerc25[cpt - 1])/2
			# 			dataPerc75[cpt] = (dataPerc75[cpt + 1] + dataPerc75[cpt - 1])/2
			# 		elif cpt > 0 :
			# 			dataPlot[cpt] = dataPlot[cpt - 1]
			# 			dataPerc25[cpt] = dataPerc25[cpt - 1]
			# 			dataPerc75[cpt] = dataPerc75[cpt - 1]
			# 		else :
			# 			dataPlot[cpt] = dataPlot[cpt + 1]
			# 			dataPerc25[cpt] = dataPerc25[cpt + 1]
			# 			dataPerc75[cpt] = dataPerc75[cpt + 1]

			# 	cpt += 1

		axe1.plot(range(len(dataPlot)), dataPlot, color = palette[cptExp], linestyle = '-', linewidth = 4)
		plt.fill_between(range(len(dataPlot)), dataPerc25, dataPerc75, alpha=0.25, linewidth=0, color=palette[cptExp])

	# 	dataPlot = []
	# 	dataPerc25 = []
	# 	dataPerc75 = []
	# 	for generation in tabGen :
	# 		fitnessGen = [dataExp[replicate][generation]['maxFit'] for replicate in dataExp.keys()]
	# 		fitnessMed = np.median(fitnessGen)

	# 		perc25 = fitnessMed
	# 		perc75 = fitnessMed
	# 		if len(fitnessGen) > 1 :
	# 			perc25 = np.percentile(fitnessGen, 25)
	# 			perc75 = np.percentile(fitnessGen, 75)

	# 		dataPlot.append(fitnessMed)
	# 		dataPerc25.append(perc25)
	# 		dataPerc75.append(perc75)

	# 		# cpt = 0
	# 		# while cpt < len(dataPlot) :
	# 		# 	if math.isnan(dataPlot[cpt]) :
	# 		# 		if cpt > 0 and cpt < len(dataPlot) - 1 :
	# 		# 			dataPlot[cpt] = (dataPlot[cpt + 1] + dataPlot[cpt - 1])/2
	# 		# 			dataPerc25[cpt] = (dataPerc25[cpt + 1] + dataPerc25[cpt - 1])/2
	# 		# 			dataPerc75[cpt] = (dataPerc75[cpt + 1] + dataPerc75[cpt - 1])/2
	# 		# 		elif cpt > 0 :
	# 		# 			dataPlot[cpt] = dataPlot[cpt - 1]
	# 		# 			dataPerc25[cpt] = dataPerc25[cpt - 1]
	# 		# 			dataPerc75[cpt] = dataPerc75[cpt - 1]
	# 		# 		else :
	# 		# 			dataPlot[cpt] = dataPlot[cpt + 1]
	# 		# 			dataPerc25[cpt] = dataPerc25[cpt + 1]
	# 		# 			dataPerc75[cpt] = dataPerc75[cpt + 1]

	# 		# 	cpt += 1

	# axe1.plot(range(len(dataPlot)), dataPlot, color = palette[1], linestyle = '-', linewidth = 4)
	# plt.fill_between(range(len(dataPlot)), dataPerc25, dataPerc75, alpha=0.25, linewidth=0, color=palette[1])

		cptExp += 1

	xticks = range(0, len(dataPlot), len(dataPlot)/5)
	# xticks = range(0, len(dataPlot), len(dataPlot)/10)
	if len(dataPlot) - 1 not in xticks :
		xticks.append(len(dataPlot) - 1)
	xticksLabels = [tabGen[t] for t in xticks]
	
	axe1.set_xticks(xticks)
	axe1.set_xticklabels(xticksLabels)
	axe1.set_xlabel("Generations")

	axe1.set_ylabel("Performance")
	axe1.set_ylim(0, 1.0)

	# legend = plt.legend(['Unconstrained', 'Constant signal', 'Constant velocity'], loc = 4, frameon=True)
	# legend = plt.legend(['Unconstrained', 'No onset/length variation'], loc = 4, frameon=True)
	# legend = plt.legend(['Average', 'Best'], loc = 4, frameon=True)
	legend = plt.legend(['Communication', 'No communication'], loc = 1, frameon=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.9')
	frame.set_edgecolor('0.9')

	plt.savefig(outputData + "/avgPerformance.png", bbox_inches = 'tight', dpi = DPI)
	plt.savefig(outputData + "/avgPerformance.svg", bbox_inches = 'tight', dpi = DPI)
	plt.close()


	cptExp = 0
	for exp in sorted(hashData.keys()) :
		fig, axe1 = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
		# plt.axes(frameon=0)
		plt.grid()

		dataExp = hashData[exp][0]
		tabGen = sorted(hashData[exp][1])

		dataPlot = []

		for replicate in dataExp.keys() :
			dataPlot = [dataExp[replicate][generation]['meanFit'] for generation in tabGen]

			axe1.plot(range(len(dataPlot)), dataPlot, color = palette[cptExp], linestyle = '-', linewidth = 4)

		cptExp += 1

		xticks = range(0, len(dataPlot), len(dataPlot)/5)
		# xticks = range(0, len(dataPlot), len(dataPlot)/10)
		if len(dataPlot) - 1 not in xticks :
			xticks.append(len(dataPlot) - 1)
		xticksLabels = [tabGen[t] for t in xticks]
		
		axe1.set_xticks(xticks)
		axe1.set_xticklabels(xticksLabels)
		axe1.set_xlabel("Generations")

		axe1.set_ylabel("Performance")
		axe1.set_ylim(0, 1.0)

		# legend = plt.legend(['Unconstrained', 'Constant signal', 'Constant velocity'], loc = 4, frameon=True)
		# legend = plt.legend(['Unconstrained', 'No onset/length variation'], loc = 4, frameon=True)
		# legend = plt.legend(['Average', 'Best'], loc = 4, frameon=True)
		# legend = plt.legend(['Unconstrained', 'No signaling'], loc = 1, frameon=True)
		# frame = legend.get_frame()
		# frame.set_facecolor('0.9')
		# frame.set_edgecolor('0.9')

		plt.savefig(outputData + "/avgPerformanceAllReplicates" + exp + ".png", bbox_inches = 'tight', dpi = DPI)
		plt.savefig(outputData + "/avgPerformanceAllReplicates" + exp + ".svg", bbox_inches = 'tight', dpi = DPI)
		plt.close()


def drawAvgFit(directories) :
	hashData = {}

	print('Reading data...')
	for d in directories :
		if os.path.isdir(d) :
			if RECURSIVE :
				tuplesWalk = os.walk(d)

				for t in tuplesWalk :
					if 'avg.txt' in t[2] :
						print('\t->' + t[0])
						with open(os.path.join(t[0], 'avg.txt'), 'r') as fileRead :
							fileRead = fileRead.readlines()

							dictResults = {}

							for line in fileRead :
								dictTmp = {}
								lineSplit = line.split(' ')

								if len(lineSplit) > 0 :
									run = int(lineSplit[0])
									avgFit = float(lineSplit[1])
									bestFit = float(lineSplit[2])

									fitCompCtrl = float(lineSplit[9])
									fitCompSigFixed = float(lineSplit[10])
									fitCompVelFixed = float(lineSplit[11])

									dictTmp['avgFit'] = avgFit
									dictTmp['bestFit'] = bestFit

									dictTmp['fitCompCtrl'] = fitCompCtrl
									dictTmp['fitCompSigFixed'] = fitCompSigFixed
									dictTmp['fitCompVelFixed'] = fitCompVelFixed

									dictResults[run] = dictTmp

						hashData[os.path.basename(t[0])] = dictResults
			else :
				listFiles = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]

				if 'avg.txt' not in listFiles :
					print("Not avg.txt in " + d)
					continue
				else :
					print('\t->' + d)
					with open(os.path.join(d, 'avg.txt'), 'r') as fileRead :
						fileRead = fileRead.readlines()

						dictResults = {}

						for line in fileRead :
							dictTmp = {}
							lineSplit = line.split(' ')

							if len(lineSplit) > 0 :
								run = int(lineSplit[0])
								avgFit = float(lineSplit[1])
								bestFit = float(lineSplit[2])

								fitCompCtrl = float(lineSplit[9])
								fitCompSigFixed = float(lineSplit[10])
								fitCompVelFixed = float(lineSplit[11])

								dictTmp['avgFit'] = avgFit
								dictTmp['bestFit'] = bestFit

								dictTmp['fitCompCtrl'] = fitCompCtrl
								dictTmp['fitCompSigFixed'] = fitCompSigFixed
								dictTmp['fitCompVelFixed'] = fitCompVelFixed

								dictResults[run] = dictTmp

				hashData[os.path.basename(d)] = dictResults





	# ----- AVERAGE FITNESS BOXPLOTS -----
	print('Drawing average performance boxplots...')

	outputData = OUTPUT_DIR

	if os.path.isdir(outputData) :
		shutil.rmtree(outputData)

	os.makedirs(outputData)

	dataBoxPlot = []
	x_ticksLabels = []
	cpt = 0
	nbByFig = 3
	numFig = 1
	for exp in sorted(hashData.keys()) :
		listAvgFit = [hashData[exp][run]['avgFit'] for run in hashData[exp].keys()]
		dataBoxPlot.append(listAvgFit)
		x_ticksLabels.append(exp)

		cpt += 1
		if cpt%nbByFig == 0 or cpt == len(hashData.keys()) :
			fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
			plt.grid()
			# palette = sns.color_palette("husl", len(hashData))
			# palette = sns.color_palette("husl", nbByFig if cpt % nbByFig == 0 else cpt)

			dataStats = {}
			index = 0
			for data in dataBoxPlot :
				curDataStats = list()
				for dataCmp in dataBoxPlot :
					U, P = stats.mannwhitneyu(data, dataCmp)
					stars = "n.s."
					if P < 0.0001:
					   stars = "****"
					elif (P < 0.001):
					   stars = "***"
					elif (P < 0.01):
					   stars = "**"
					elif (P < 0.05):
					   stars = "*"

					curDataStats.append(stars)
				dataStats[index] = curDataStats
				index += 1

			# for i in range(0, len(dataStats)) :
			# 	print("Data " + str(i) + " : ")
			# 	for j in range(0, len(dataStats[i])) :
			# 		print("Data " + str(j) + " : P = " + str(dataStats[i][j]))

			bp = ax.boxplot(dataBoxPlot)

			# ax.spines['top'].set_visible(False)
			# ax.spines['right'].set_visible(False)
			# ax.spines['left'].set_visible(False)
			# ax.get_xaxis().tick_bottom()
			# ax.get_yaxis().tick_left()
			# ax.tick_params(axis='x', direction='out')
			# ax.tick_params(axis='y', length=0)

			ax.set_xlabel("Setting")
			ax.set_ylabel("Performance")
			ax.set_ylim(0, 1.0)

			x_ticksLabels = ['Unconstrained', 'Constrained\nNo onset/length variation']			
			# x_ticksLabels = ['Unconstrained', 'Constant signal', 'Constant velocity']			
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

			# y_max = np.max(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
			# y_min = np.min(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
			# y_min = 0

			# print y_max

			# Print stats
			cptData = 0
			while cptData < len(dataStats) :
				cptComp = cptData + 1
				while cptComp < len(dataStats[cptData]) :
					if dataStats[cptData][cptComp] != "-" :
						if cptData == 0 and cptComp == 2 :
							ax.annotate("", xy=(cptData + 1, 0.85), xycoords='data',
							           xytext=(cptComp + 1, 0.85), textcoords='data',
							           arrowprops=dict(arrowstyle="-", ec='#000000',
							                           connectionstyle="bar,fraction=0.05"))
							ax.text((cptComp - cptData)/2 + cptData + 1, 0.90, dataStats[cptData][cptComp],
							       horizontalalignment='center',
							       verticalalignment='center')
						else :
							ax.annotate("", xy=(cptData + 1, 0.75), xycoords='data',
							           xytext=(cptComp + 1, 0.75), textcoords='data',
							           arrowprops=dict(arrowstyle="-", ec='#000000',
							                           connectionstyle="bar,fraction=0.05"))
							ax.text(float(cptComp - cptData)/2 + cptData + 1, 0.80, dataStats[cptData][cptComp],
							       horizontalalignment='center',
							       verticalalignment='center')

					cptComp += 1
				cptData += 1


			plt.savefig(outputData + "/boxplotTest" + str(numFig) + ".png", bbox_inches = 'tight', dpi = DPI)
			plt.savefig(outputData + "/boxplotTest" + str(numFig) + ".svg", bbox_inches = 'tight', dpi = DPI)
			plt.close()

			numFig += 1
			dataBoxPlot = []
			x_ticksLabels = []



	# ----- AVERAGE FITNESS STRATEGY -----
	print('Drawing performance depending on strategy...')

	for exp in sorted(hashData.keys()) :
		dataBoxPlot = []
		x_ticksLabels = []
		outputData = os.path.join(OUTPUT_DIR, exp)

		if os.path.isdir(outputData) :
			shutil.rmtree(outputData)

		os.makedirs(outputData)

		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
		plt.grid()
		# palette = sns.color_palette("husl", len(hashData))
		# palette = sns.color_palette("husl", nbByFig if cpt % nbByFig == 0 else cpt)

		dataBoxPlot.append([hashData[exp][run]['avgFit'] for run in RUNS_ONSET])
		dataBoxPlot.append([hashData[exp][run]['avgFit'] for run in RUNS_LENGTH])
		dataBoxPlot.append([hashData[exp][run]['avgFit'] for run in RUNS_BOTH])

		dataStats = {}
		index = 0
		for data in dataBoxPlot :
			curDataStats = list()
			for dataCmp in dataBoxPlot :
				U, P = stats.mannwhitneyu(data, dataCmp)
				# print(str(U) + "/" + str(P))
				stars = "n.s."
				if P < 0.0001:
				   stars = "****"
				elif (P < 0.001):
				   stars = "***"
				elif (P < 0.01):
				   stars = "**"
				elif (P < 0.05):
				   stars = "*"

				curDataStats.append(stars)
			dataStats[index] = curDataStats
			index += 1

		print("Onset : mean fitness at last generation : m = " + str(np.mean([hashData[exp][run]['avgFit'] for run in RUNS_ONSET])) + ", var = " + str(np.var([hashData[exp][run]['avgFit'] for run in RUNS_ONSET])))
		print("Length : mean fitness at last generation : m = " + str(np.mean([hashData[exp][run]['avgFit'] for run in RUNS_LENGTH])) + ", var = " + str(np.var([hashData[exp][run]['avgFit'] for run in RUNS_LENGTH])))
		print("Both : mean fitness at last generation : m = " + str(np.mean([hashData[exp][run]['avgFit'] for run in RUNS_BOTH])) + ", var = " + str(np.var([hashData[exp][run]['avgFit'] for run in RUNS_BOTH])))

		bp = ax.boxplot(dataBoxPlot)

		# ax.spines['top'].set_visible(False)
		# ax.spines['right'].set_visible(False)
		# ax.spines['left'].set_visible(False)
		# ax.get_xaxis().tick_bottom()
		# ax.get_yaxis().tick_left()
		# ax.tick_params(axis='x', direction='out')
		# ax.tick_params(axis='y', length=0)

		ax.set_xlabel("Strategy")
		ax.set_ylabel("Performance")
		ax.set_ylim(0, 1.0)

		x_ticksLabels = ['Onset variation', 'Length variation', 'Both']			
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

		# y_max = np.max(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
		# y_min = np.min(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
		# y_min = 0

		# print y_max

		# Print stats
		cptData = 0
		while cptData < len(dataStats) :
			cptComp = cptData + 1
			while cptComp < len(dataStats[cptData]) :
				if dataStats[cptData][cptComp] != "-" :
					if cptData == 0 and cptComp == 2 :
						ax.annotate("", xy=(cptData + 1, 0.85), xycoords='data',
						           xytext=(cptComp + 1, 0.85), textcoords='data',
						           arrowprops=dict(arrowstyle="-", ec='#000000',
						                           connectionstyle="bar,fraction=0.05"))
						ax.text((cptComp - cptData)/2 + cptData + 1, 0.93, dataStats[cptData][cptComp],
						       horizontalalignment='center',
						       verticalalignment='center')
					else :
						ax.annotate("", xy=(cptData + 1, 0.75), xycoords='data',
						           xytext=(cptComp + 1, 0.75), textcoords='data',
						           arrowprops=dict(arrowstyle="-", ec='#000000',
						                           connectionstyle="bar,fraction=0.05"))
						ax.text(float(cptComp - cptData)/2 + cptData + 1, 0.80, dataStats[cptData][cptComp],
						       horizontalalignment='center',
						       verticalalignment='center')

				cptComp += 1
			cptData += 1


		plt.savefig(outputData + "/boxplotTest2.png", bbox_inches = 'tight', dpi = DPI)
		plt.savefig(outputData + "/boxplotTest2.svg", bbox_inches = 'tight', dpi = DPI)
		plt.close()


def drawFitComp(directories) :
	hashData = {}

	print('Reading data...')
	for d in directories :
		if os.path.isdir(d) :
			if RECURSIVE :
				tuplesWalk = os.walk(d)

				for t in tuplesWalk :
					if 'avg.txt' in t[2] :
						print('\t->' + t[0])
						with open(os.path.join(t[0], 'avg.txt'), 'r') as fileRead :
							fileRead = fileRead.readlines()

							dictResults = {}

							for line in fileRead :
								dictTmp = {}
								lineSplit = line.split(' ')

								if len(lineSplit) > 0 :
									run = int(lineSplit[0])
									avgFit = float(lineSplit[1])
									bestFit = float(lineSplit[2])

									fitCompCtrl = float(lineSplit[9])
									fitCompSigFixed = float(lineSplit[10])
									fitCompVelFixed = float(lineSplit[11])

									dictTmp['avgFit'] = avgFit
									dictTmp['bestFit'] = bestFit

									dictTmp['fitCompCtrl'] = fitCompCtrl
									dictTmp['fitCompSigFixed'] = fitCompSigFixed
									dictTmp['fitCompVelFixed'] = fitCompVelFixed

									dictResults[run] = dictTmp

						hashData[os.path.basename(t[0])] = dictResults
			else :
				listFiles = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]

				if 'avg.txt' not in listFiles :
					print("Not avg.txt in " + d)
					continue
				else :
					print('\t->' + d)
					with open(os.path.join(d, 'avg.txt'), 'r') as fileRead :
						fileRead = fileRead.readlines()

						dictResults = {}

						for line in fileRead :
							dictTmp = {}
							lineSplit = line.split(' ')

							if len(lineSplit) > 0 :
								run = int(lineSplit[0])
								avgFit = float(lineSplit[1])
								bestFit = float(lineSplit[2])

								fitCompCtrl = float(lineSplit[9])
								fitCompSigFixed = float(lineSplit[10])
								fitCompVelFixed = float(lineSplit[11])

								dictTmp['avgFit'] = avgFit
								dictTmp['bestFit'] = bestFit

								dictTmp['fitCompCtrl'] = fitCompCtrl
								dictTmp['fitCompSigFixed'] = fitCompSigFixed
								dictTmp['fitCompVelFixed'] = fitCompVelFixed

								dictResults[run] = dictTmp

				hashData[os.path.basename(d)] = dictResults


	# ----- FITNESS COMP BOXPLOTS -----
	print('Drawing fitness comp boxplots...')

	for exp in sorted(hashData.keys()) :
		outputData = OUTPUT_DIR + "/" + exp

		if os.path.isdir(outputData) :
			shutil.rmtree(outputData)

		os.makedirs(outputData)

		dataBoxPlot = []
		x_ticksLabels = []

		listFitCtrl = [hashData[exp][run]['fitCompCtrl'] for run in hashData[exp].keys()]
		listFitSigFixed = [hashData[exp][run]['fitCompSigFixed'] for run in hashData[exp].keys()]
		listFitVelFixed = [hashData[exp][run]['fitCompVelFixed'] for run in hashData[exp].keys()]

		dataBoxPlot.append(listFitCtrl)
		dataBoxPlot.append(listFitSigFixed)
		# dataBoxPlot.append(listFitVelFixed)

		x_ticksLabels = ['Unconstrained', 'Fixed signal']

		print("Unconstrained : mean performance at last generation : m = " + str(np.mean(listFitCtrl)) + ", var = " + str(np.var(listFitCtrl)))
		print("Fixed signal : mean performance at last generation : m = " + str(np.mean(listFitSigFixed)) + ", var = " + str(np.var(listFitSigFixed)))

		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
		plt.grid()
		# palette = sns.color_palette("husl", len(hashData))
		# palette = sns.color_palette("husl", 3)

		dataStats = {}
		index = 0
		for data in dataBoxPlot :
			curDataStats = list()
			for dataCmp in dataBoxPlot :
				U, P = stats.mannwhitneyu(data, dataCmp)
				print(str(U) + "/" + str(P))
				stars = "n.s."
				if P < 0.0001:
				   stars = "****"
				elif (P < 0.001):
				   stars = "***"
				elif (P < 0.01):
				   stars = "**"
				elif (P < 0.05):
				   stars = "*"

				curDataStats.append(stars)
			dataStats[index] = curDataStats
			index += 1

		# for i in range(0, len(dataStats)) :
		# 	print("Data " + str(i) + " : ")
		# 	for j in range(0, len(dataStats[i])) :
		# 		print("Data " + str(j) + " : P = " + str(dataStats[i][j]))

		bp = ax.boxplot(dataBoxPlot)

		# ax.spines['top'].set_visible(False)
		# ax.spines['right'].set_visible(False)
		# ax.spines['left'].set_visible(False)
		# ax.get_xaxis().tick_bottom()
		# ax.get_yaxis().tick_left()
		# ax.tick_params(axis='x', direction='out')
		# ax.tick_params(axis='y', length=0)

		ax.set_xlabel("Setting")
		ax.set_ylabel("Performance")
		ax.set_ylim(0, 1)

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

		# y_max = np.max(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
		# y_min = np.min(np.concatenate((dataHash['SF'][:, 0], dataHash['TT'][:, 0])))
		# y_min = 0

		# print y_max

		# Print stats
		cptData = 0
		while cptData < len(dataStats) :
			cptComp = cptData + 1
			while cptComp < len(dataStats[cptData]) :
				if dataStats[cptData][cptComp] != "-" :
					if cptData == 0 and cptComp == 2 :
						ax.annotate("", xy=(cptData + 1, 0.85), xycoords='data',
						           xytext=(cptComp + 1, 0.85), textcoords='data',
						           arrowprops=dict(arrowstyle="-", ec='#000000',
						                           connectionstyle="bar,fraction=0.05"))
						ax.text((cptComp - cptData)/2 + cptData + 1, 0.90, dataStats[cptData][cptComp],
						       horizontalalignment='center',
						       verticalalignment='center')
					else :
						ax.annotate("", xy=(cptData + 1, 0.75), xycoords='data',
						           xytext=(cptComp + 1, 0.75), textcoords='data',
						           arrowprops=dict(arrowstyle="-", ec='#000000',
						                           connectionstyle="bar,fraction=0.05"))
						ax.text(float(cptComp - cptData)/2 + cptData + 1, 0.80, dataStats[cptData][cptComp],
						       horizontalalignment='center',
						       verticalalignment='center')

				cptComp += 1
			cptData += 1


		plt.savefig(outputData + "/boxplotCompExp.png", bbox_inches = 'tight', dpi = DPI)
		plt.savefig(outputData + "/boxplotCompExp.svg", bbox_inches = 'tight', dpi = DPI)
		plt.close()



def drawSignalComp(directories) :
	hashData = {}
	regRep = re.compile(r"^(\d+)$")

	print('Reading data...')
	for d in directories :
		if os.path.isdir(d) :
			if RECURSIVE :
				tuplesWalk = os.walk(d)

				for t in tuplesWalk :
					listReplicates = [replicate for replicate in t[1] if regRep.match(replicate)]

					if len(listReplicates) > 0 :
						dictReplicates = {}
						for replicate in listReplicates :
							m = regRep.match(replicate)

							replicateDir = os.path.join(t[0], replicate)
							numReplicate = int(m.group(1))

							listInd = list()
							if os.path.isfile(os.path.join(replicateDir, 'comp.pop.txt')) :
								with open(os.path.join(replicateDir, 'comp.pop.txt'), 'r') as fileRead :
									fileRead = fileRead.readlines()

									for line in fileRead :
										dictInd = {}
										lineSplit = line.split(' ')

										if len(lineSplit) > 0 :
											dictInd['fitCompCtrl'] = float(lineSplit[1])
											dictInd['fitCompOnset'] = float(lineSplit[4])
											dictInd['fitCompLength'] = float(lineSplit[5])
											dictInd['fitCompLength2'] = float(lineSplit[6])
											dictInd['fitCompLength3'] = float(lineSplit[7])
											dictInd['fitCompLength4'] = float(lineSplit[8])

											listInd.append(dictInd)

							dictReplicates[numReplicate] = listInd

						hashData[os.path.basename(t[0])] = dictReplicates
			else :
				listReplicates = [replicate for replicate in os.listdir(d) if regRep.match(replicate)]

				if len(listReplicates) > 0 :
					dictReplicates = {}
					for replicate in listReplicates :
						m = regRep.match(replicate)

						replicateDir = os.path.join(d, replicate)
						numReplicate = int(m.group(1))

						listInd = list()
						if os.path.isfile(os.path.join(replicateDir, 'comp.pop.txt')) :
							with open(os.path.join(replicateDir, 'comp.pop.txt'), 'r') as fileRead :
								fileRead = fileRead.readlines()

								for line in fileRead :
									dictInd = {}
									lineSplit = line.split(' ')

									if len(lineSplit) > 0 :
										dictInd['fitCompCtrl'] = float(lineSplit[1])
										dictInd['fitCompOnset'] = float(lineSplit[4])
										dictInd['fitCompLength'] = float(lineSplit[5])
										dictInd['fitCompLength2'] = float(lineSplit[6])
										dictInd['fitCompLength3'] = float(lineSplit[7])
										dictInd['fitCompLength4'] = float(lineSplit[8])

										listInd.append(dictInd)

						dictReplicates[numReplicate] = listInd

					hashData[os.path.basename(d)] = dictReplicates

				listFiles = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]


	# ----- FITNESS COMP SIGNAL BOXPLOTS -----
	print('Drawing fitness comp signal boxplots...')

	for exp in sorted(hashData.keys()) :
		outputData = OUTPUT_DIR + "/" + exp

		# if os.path.isdir(outputData) :
		# 	shutil.rmtree(outputData)

		if not os.path.isdir(outputData) :
			os.makedirs(outputData)

		dataBoxPlot = []
		x_ticksLabels = []

		listDataFitUncomp = []
		listFitCtrlUncomp = {run : [hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitOnsetUncomp = {run : [hashData[exp][run][ind]['fitCompOnset'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitLengthUncomp = {run : [hashData[exp][run][ind]['fitCompLength'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitLength2Uncomp = {run : [hashData[exp][run][ind]['fitCompLength2'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitLength3Uncomp = {run : [hashData[exp][run][ind]['fitCompLength3'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitLength4Uncomp = {run : [hashData[exp][run][ind]['fitCompLength4'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}

		listDataFitUncomp.append(listFitOnsetUncomp)
		listDataFitUncomp.append(listFitLengthUncomp)
		listDataFitUncomp.append(listFitLength2Uncomp)
		listDataFitUncomp.append(listFitLength3Uncomp)
		listDataFitUncomp.append(listFitLength4Uncomp)

		# STATS
		dataStats = {}
		index = 0
		for run in hashData[exp].keys() :
			dataStats[run] = {}

			for bar in BAR_TO_SHOW :
				U, P = stats.mannwhitneyu(listFitCtrlUncomp[run], listDataFitUncomp[NUM_BAR[bar]][run])
				stars = "n.s."
				if P < 0.0001:
				   stars = "****"
				elif (P < 0.001):
				   stars = "***"
				elif (P < 0.01):
				   stars = "**"
				elif (P < 0.05):
				   stars = "*"

				dataStats[run][bar] = stars

		# listDiffFitOnset =  [np.mean([hashData[exp][run][ind]['fitCompOnset'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listDiffFitOnsetStd =  [np.std([hashData[exp][run][ind]['fitCompOnset'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listDiffFitOnsetSte =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][run])), listDiffFitOnsetStd)
		listDiffFitOnset =  [np.mean([(hashData[exp][run][ind]['fitCompOnset'] - hashData[exp][run][ind]['fitCompCtrl'])/hashData[exp][run][ind]['fitCompCtrl'] if hashData[exp][run][ind]['fitCompCtrl'] > 0 else 0 for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitOnsetStd =  [np.std([(hashData[exp][run][ind]['fitCompOnset'] - hashData[exp][run][ind]['fitCompCtrl'])/hashData[exp][run][ind]['fitCompCtrl'] if hashData[exp][run][ind]['fitCompCtrl'] > 0 else 0 for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitOnsetSte =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][run])), listDiffFitOnsetStd)
		listDiffFitLength =  [np.mean([hashData[exp][run][ind]['fitCompLength'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitLengthStd =  [np.std([hashData[exp][run][ind]['fitCompLength'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitLengthSte =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][run])), listDiffFitLengthStd)
		listDiffFitLength2 =  [np.mean([hashData[exp][run][ind]['fitCompLength2'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitLength2Std =  [np.std([hashData[exp][run][ind]['fitCompLength2'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitLength2Ste =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][run])), listDiffFitLength2Std)
		listDiffFitLength3 =  [np.mean([hashData[exp][run][ind]['fitCompLength3'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitLength3Std =  [np.std([hashData[exp][run][ind]['fitCompLength3'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitLength3Ste =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][run])), listDiffFitLength3Std)
		# listDiffFitLength4 =  [np.mean([hashData[exp][run][ind]['fitCompLength4'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listDiffFitLength4Std =  [np.std([hashData[exp][run][ind]['fitCompLength4'] - hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listDiffFitLength4Ste =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][run])), listDiffFitLength4Std)
		listDiffFitLength4 =  [np.mean([(hashData[exp][run][ind]['fitCompLength4'] - hashData[exp][run][ind]['fitCompCtrl'])/hashData[exp][run][ind]['fitCompCtrl'] if hashData[exp][run][ind]['fitCompCtrl'] > 0 else 0 for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitLength4Std =  [np.std([(hashData[exp][run][ind]['fitCompLength4'] - hashData[exp][run][ind]['fitCompCtrl'])/hashData[exp][run][ind]['fitCompCtrl'] if hashData[exp][run][ind]['fitCompCtrl'] > 0 else 0 for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		listDiffFitLength4Ste =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][run])), listDiffFitLength4Std)
		# listRatioFitOnset =  [np.mean([hashData[exp][run][ind]['fitCompOnset']/hashData[exp][run][ind]['fitCompCtrl'] if hashData[exp][run][ind]['fitCompCtrl'] > 0 else 0 for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listRatioFitOnsetStd =  [np.std([hashData[exp][run][ind]['fitCompOnset']/hashData[exp][run][ind]['fitCompCtrl'] if hashData[exp][run][ind]['fitCompCtrl'] > 0 else 0 for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listRatioFitOnsetSte =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][run])), listRatioFitOnsetStd)

		listDataFit = []
		listDataStd = []
		listDataSte = []
		listDataFit.append(listDiffFitOnset)
		listDataStd.append(listDiffFitOnsetStd)
		listDataSte.append(listDiffFitOnsetSte)
		listDataFit.append(listDiffFitLength)
		listDataStd.append(listDiffFitLengthStd)
		listDataSte.append(listDiffFitLengthSte)
		listDataFit.append(listDiffFitLength2)
		listDataStd.append(listDiffFitLength2Std)
		listDataSte.append(listDiffFitLength2Ste)
		listDataFit.append(listDiffFitLength3)
		listDataStd.append(listDiffFitLength3Std)
		listDataSte.append(listDiffFitLength3Ste)
		listDataFit.append(listDiffFitLength4)
		listDataStd.append(listDiffFitLength4Std)
		listDataSte.append(listDiffFitLength4Ste)

		figsize = (4096/DPI, 1092/DPI)

		matplotlib.rcParams['font.size'] = 34
		matplotlib.rcParams['font.weight'] = 'bold'
		matplotlib.rcParams['axes.labelsize'] = 34
		matplotlib.rcParams['axes.labelweight'] = 'bold'
		matplotlib.rcParams['xtick.labelsize'] = 34
		matplotlib.rcParams['ytick.labelsize'] = 34
		matplotlib.rcParams['legend.fontsize'] = 34
		matplotlib.rcParams['mathtext.default'] = 'regular'

		nbPlots = len(listDiffFitOnset)/int(NB_REPLICATE_PER_PLOT)
		listReplicatesToPlot = list()
		for i in range(0, int(nbPlots)) :
			listReplicatesToPlot.append(list(range(i*NB_REPLICATE_PER_PLOT, (i+1)*NB_REPLICATE_PER_PLOT)))

		cptList = 1
		# for curList in listReplicatesToPlot :
		# 	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, dpi = DPI)
		# 	plt.grid()

		# 	listBars = []
		# 	cpt = 0
		# 	for bar in BAR_TO_SHOW :
		# 		# listBars.append(ax.bar([x + MARGIN_BARS_DIFF + cpt*WIDTH_BARS_DIFF for x in range(len(curList))], [listDataFit[NUM_BAR[bar]][replicate] for replicate in curList], width = WIDTH_BARS_DIFF, color = palette[cpt+1], yerr = [listDataSte[NUM_BAR[bar]][replicate] for replicate in curList]))
		# 		listBars.append(ax.bar([x for x in range(len(curList))], [listDataFit[NUM_BAR[bar]][replicate] for replicate in curList], width = WIDTH_BARS_DIFF, color = palette[cpt+1], yerr = [listDataSte[NUM_BAR[bar]][replicate] for replicate in curList]))
		# 		cpt += 1

		# 	# xticks = [x + 0.5 for x in range(len(curList))]
		# 	# xticksLabels = [x + 1 for x in curList]
		# 	xticks = [x for x in range(len(curList))]
		# 	xticksLabels = [x + 1 for x in curList]
		# 	ax.set_xticks(xticks)
		# 	ax.set_xticklabels(xticksLabels)
		# 	ax.set_xlabel("Population line")

		# 	ax.set_ylabel("Performance difference")
		# 	ax.set_xlim(-1, 40)
		# 	ax.set_ylim(-0.7, 0.2)

		# 	# legend = plt.legend(['Sender', 'Receiver'], loc = 1, frameon=True)
		# 	# # legend = plt.legend(['Sender', 'Receiver'], loc = 4, frameon=True)
		# 	# frame = legend.get_frame()
		# 	# frame.set_facecolor('0.9')
		# 	# frame.set_edgecolor('0.9')

		# 	cptRun = 0
		# 	for run in curList :
		# 		cptBar = 0
		# 		for bar in dataStats[run].keys() :
		# 			# print(str(cptBar) + " - " + str(cptRun) + " : " + dataStats[run][bar])
		# 			# ax.annotate(dataStats[run][bar], xy = (cptRun + MARGIN_BARS_DIFF + WIDTH_BARS_DIFF/2., 0.15), fontsize = 28, fontweight = "bold", horizontalalignment = 'center', verticalalignment = 'center') #, zorder = 10)
		# 			ax.annotate(dataStats[run][bar], xy = (cptRun, 0.15), fontsize = 28, fontweight = "bold", horizontalalignment = 'center', verticalalignment = 'center') #, zorder = 10)
		# 			cptBar += 1

		# 		cptRun += 1

		# 	plt.savefig(outputData + "/barsDiffFit" + str(cptList) + ".png", bbox_inches = 'tight', dpi = DPI)
		# 	plt.savefig(outputData + "/barsDiffFit" + str(cptList) + ".svg", bbox_inches = 'tight', dpi = DPI)
		# 	plt.close()

		# 	cptList += 1


	# ----- FITNESS COMP SIGNAL BOXPLOTS -----
	print('Drawing fitness comp signal boxplots...')

	for exp in sorted(hashData.keys()) :
		outputData = OUTPUT_DIR + "/" + exp

		# if os.path.isdir(outputData) :
		# 	shutil.rmtree(outputData)

		# os.makedirs(outputData)

		dataBoxPlot = []
		x_ticksLabels = []

		listDataFitUncomp = []
		listFitCtrlUncomp = {run : [hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitOnsetUncomp = {run : [hashData[exp][run][ind]['fitCompOnset'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitLengthUncomp = {run : [hashData[exp][run][ind]['fitCompLength'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitLength2Uncomp = {run : [hashData[exp][run][ind]['fitCompLength2'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitLength3Uncomp = {run : [hashData[exp][run][ind]['fitCompLength3'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}
		listFitLength4Uncomp = {run : [hashData[exp][run][ind]['fitCompLength4'] for ind in range(0, len(hashData[exp][run]))] for run in hashData[exp].keys()}

		listDataFitUncomp.append(listFitOnsetUncomp)
		listDataFitUncomp.append(listFitLengthUncomp)
		listDataFitUncomp.append(listFitLength2Uncomp)
		listDataFitUncomp.append(listFitLength3Uncomp)
		listDataFitUncomp.append(listFitLength4Uncomp)

		# STATS
		dataStats = {}
		index = 0
		nbSignificantOnset = 0
		nbSignificantLength = 0
		nbSignificantBoth = 0
		for run in hashData[exp].keys() :
			dataStats[run] = {}

			for bar in BAR_TO_SHOW :
				U, P = stats.mannwhitneyu(listFitCtrlUncomp[run], listDataFitUncomp[NUM_BAR[bar]][run])
				stars = "n.s."
				if P < 0.0001:
				   stars = "****"
				elif (P < 0.001):
				   stars = "***"
				elif (P < 0.01):
				   stars = "**"
				elif (P < 0.05):
				   stars = "*"

				dataStats[run][bar] = stars

			both = False
			if dataStats[run]["onset"] != "n.s." and dataStats[run]["onset"] != "*" :
				nbSignificantOnset += 1
				both = True

			if dataStats[run]["length4"] != "n.s." and dataStats[run]["length4"] != "*"  :
				nbSignificantLength += 1

				if both :
					nbSignificantBoth += 1

		figsize = FIGSIZE
		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, dpi = DPI)
		plt.grid()

		listDataSignificant = [nbSignificantOnset, nbSignificantLength, nbSignificantBoth]
		listDataSignificant = [float(nbPop)/40. * 100 for nbPop in listDataSignificant]

		barWidth = 0.3
		print(listDataSignificant)
		# listBars = ax.bar([x + 0.5 for x in range(len(listDataSignificant))], listDataSignificant, width = 0.3, color = palette[0])
		listBars = ax.bar([x for x in range(len(listDataSignificant))], listDataSignificant, width = barWidth, color = palette[0])

		cptBar = 0
		for bar in listBars :
			bar.set_color(palette[cptBar])

			cptBar += 1

		# xticks = [x + 0.5 for x in range(len(listDataSignificant))]
		xticks = [x for x in range(len(listDataSignificant))]
		xticksLabels = ["Length", "Onset", "Both"]
		# xticks = [x + 0.5 for x in range(len(curList))]
		# xticksLabels = [x + 1 for x in curList]
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticksLabels)
		ax.set_xlabel("Mode of communication")

		ax.set_ylabel("Percentage of populations \nwith decreased performance (%)")
		# ax.set_ylabel("%populations with decreased performance")
		# ax.set_ylim(0, 40)
		ax.set_ylim(0, 100)

		# legend = plt.legend(['Sender', 'Receiver'], loc = 1, frameon=True)
		# # legend = plt.legend(['Sender', 'Receiver'], loc = 4, frameon=True)
		# frame = legend.get_frame()
		# frame.set_facecolor('0.9')
		# frame.set_edgecolor('0.9')

		plt.savefig(outputData + "/barsSigPerf.png", bbox_inches = 'tight', dpi = DPI)
		plt.savefig(outputData + "/barsSigPerf.svg", bbox_inches = 'tight', dpi = DPI)
		plt.close()

		cptList += 1

		# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = FIGSIZE, dpi = DPI)
		# plt.grid()

		# # barRatioOnset = ax.bar(range(len(listRatioFitOnset)), listRatioFitOnset, width = WIDTH_BARS, color = palette[0], yerr = listRatioFitOnsetSte)

		# # xticks = range(len(listRatioFitOnset))
		# # xticksLabels = xticks
		# # ax.set_xticks(xticks)
		# # ax.set_xticklabels(xticksLabels)
		# # ax.set_xlabel("Replicate")

		# # ax.set_ylabel("Fitness ratio")

		# # plt.savefig(outputData + "/barsRatioFit.png", bbox_inches = 'tight', dpi = DPI)
		# # plt.savefig(outputData + "/barsRatioFit.svg", bbox_inches = 'tight', dpi = DPI)
		# # plt.close()

		# listFitCtrl =  [np.mean([hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitCtrlStd =  [np.std([hashData[exp][run][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitCtrlSte =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][0])), listFitCtrlStd)

		# listFitOnset =  [np.mean([hashData[exp][run][ind]['fitCompOnset'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitOnsetStd =  [np.std([hashData[exp][run][ind]['fitCompOnset'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitOnsetSte =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][0])), listFitOnsetStd)
		# listFitLength =  [np.mean([hashData[exp][run][ind]['fitCompLength'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitLengthStd =  [np.std([hashData[exp][run][ind]['fitCompLength'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitLengthSte =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][0])), listFitLengthStd)
		# listFitLength2 =  [np.mean([hashData[exp][run][ind]['fitCompLength2'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitLength2Std =  [np.std([hashData[exp][run][ind]['fitCompLength2'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitLength2Ste =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][0])), listFitLength2Std)
		# listFitLength3 =  [np.mean([hashData[exp][run][ind]['fitCompLength3'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitLength3Std =  [np.std([hashData[exp][run][ind]['fitCompLength3'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitLength3Ste =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][0])), listFitLength3Std)
		# listFitLength4 =  [np.mean([hashData[exp][run][ind]['fitCompLength4'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitLength4Std =  [np.std([hashData[exp][run][ind]['fitCompLength4'] for ind in range(0, len(hashData[exp][run]))]) for run in hashData[exp].keys()]
		# listFitLength4Ste =  map(lambda x : float(x)/math.sqrt(len(hashData[exp][0])), listFitLength4Std)

		# listDataFit = []
		# listDataStd = []
		# listDataSte = []
		# listDataFit.append(listFitOnset)
		# listDataStd.append(listFitOnsetStd)
		# listDataSte.append(listFitOnsetSte)
		# listDataFit.append(listFitLength)
		# listDataStd.append(listFitLengthStd)
		# listDataSte.append(listFitLengthSte)
		# listDataFit.append(listFitLength2)
		# listDataStd.append(listFitLength2Std)
		# listDataSte.append(listFitLength2Ste)
		# listDataFit.append(listFitLength3)
		# listDataStd.append(listFitLength3Std)
		# listDataSte.append(listFitLength3Ste)
		# listDataFit.append(listFitLength4)
		# listDataStd.append(listFitLength4Std)
		# listDataSte.append(listFitLength4Ste)

		# # print([hashData[exp][8][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))])
		# # print(np.mean([hashData[exp][8][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]))
		# # print(np.std([hashData[exp][8][ind]['fitCompCtrl'] for ind in range(0, len(hashData[exp][run]))]))

		# figsize = (1920/DPI, 1536/DPI)
		# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, dpi = DPI)
		# # plt.grid()

		# barFitCtrl = ax.bar([x + MARGIN_BARS_RATIO for x in range(len(listFitCtrl))], listFitCtrl, width = WIDTH_BARS_RATIO, color = palette[0], yerr = listFitCtrlSte)#, align = 'edge')

		# listBars = [barFitCtrl]
		# cpt = 0
		# for bar in BAR_TO_SHOW :
		# 	listBars.append(ax.bar([x + MARGIN_BARS_RATIO + (cpt + 1)*WIDTH_BARS_RATIO for x in range(len(listDataFit[NUM_BAR[bar]]))], listDataFit[NUM_BAR[bar]], width = WIDTH_BARS_RATIO, color = palette[cpt + 1], yerr = listDataSte[NUM_BAR[bar]]))
		# 	cpt += 1

		# xticks = [x + 0.5 for x in range(len(listFitOnset))]
		# xticksLabels = range(1, len(hashData[exp].keys()) + 1)
		# ax.set_xticks(xticks)
		# ax.set_xticklabels(xticksLabels)
		# ax.set_xlabel("Replicate")

		# ax.set_ylabel("Fitness")
		# ax.set_ylim(0, 1.0)

		# cptRun = 0
		# for run in dataStats.keys() :
		# 	cptBar = 0
		# 	for bar in dataStats[run].keys() :
		# 		# print(str(cptBar) + " - " + str(cptRun) + " : " + dataStats[run][bar])
		# 		# ax.annotate(dataStats[run][bar], xy = (cptRun, 0.8)) #, zorder = 10)
		# 		ax.annotate(dataStats[run][bar], xy = (cptRun + MARGIN_BARS_RATIO + (cptBar + 1)*WIDTH_BARS_RATIO, listDataFit[NUM_BAR[bar]][cptRun] + 0.05)) #, zorder = 10)
		# 		cptBar += 1

		# 	cptRun += 1

  # #   plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
  # #   # draw the text with a bounding box covering up the line
  # #   plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)

		# # pvals = [0.001,0.1,0.00001]
		# # offset  =1
		# # for i,p in enumerate(pvals):
		# #     if p>=0.05:
		# #         displaystring = r'n.s.'
		# #     elif p<0.0001:
		# #         displaystring = r'***'
		# #     elif p<0.001:
		# #         displaystring = r'**'
		# #     else:
		# #         displaystring = r'*'

		# #     height = offset +  max(cell_lysate_avg[i],media_avg[i])
		# #     bar_centers = index[i] + numpy.array([0.5,1.5])*bar_width
		# #     significance_bar(bar_centers[0],bar_centers[1],height,displaystring)

		# # # Print stats
		# # cptData = 0
		# # while cptData < len(dataStats) :
		# # 	cptComp = cptData + 1
		# # 	while cptComp < len(dataStats[cptData]) :
		# # 		if dataStats[cptData][cptComp] != "-" :
		# # 			if cptData == 0 and cptComp == 2 :
		# # 				ax.annotate("", xy=(cptData + 1, 7300), xycoords='data',
		# # 				           xytext=(cptComp + 1, 7300), textcoords='data',
		# # 				           arrowprops=dict(arrowstyle="-", ec='#000000',
		# # 				                           connectionstyle="bar,fraction=0.05"))
		# # 				ax.text((cptComp - cptData)/2 + cptData + 1, 7725, dataStats[cptData][cptComp],
		# # 				       horizontalalignment='center',
		# # 				       verticalalignment='center')
		# # 			else :
		# # 				ax.annotate("", xy=(cptData + 1, 7000), xycoords='data',
		# # 				           xytext=(cptComp + 1, 7000), textcoords='data',
		# # 				           arrowprops=dict(arrowstyle="-", ec='#000000',
		# # 				                           connectionstyle="bar,fraction=0.05"))
		# # 				ax.text(float(cptComp - cptData)/2 + cptData + 1, 7250, dataStats[cptData][cptComp],
		# # 				       horizontalalignment='center',
		# # 				       verticalalignment='center')

		# # 		cptComp += 1
		# # 	cptData += 1


		# # legend = plt.legend(['Sender', 'Receiver'], loc = 1, frameon=True)
		# # # legend = plt.legend(['Sender', 'Receiver'], loc = 4, frameon=True)
		# # frame = legend.get_frame()
		# # frame.set_facecolor('0.9')
		# # frame.set_edgecolor('0.9')

		# plt.savefig(outputData + "/barsFit.png", bbox_inches = 'tight', dpi = DPI)
		# plt.savefig(outputData + "/barsFit.svg", bbox_inches = 'tight', dpi = DPI)
		# plt.close()



def drawBehavior(directories, info, ind) :
	hashData = {}
	regRep = re.compile(r"^(\d+)$")
	regFileInfo = re.compile(r"^Individual (\d+) : $")

	suffix = ""
	if info is not None :
		if info == "onset" or info == "length" or info == "length2" or info == "length3" or info == "length4" :
			suffix = info
		else :
			print("drawBehavior: info " + str(info) + " unknown")

	fileBehavior = None
	if ind == None :
		fileBehavior = "tmp.txt"
		if suffix != "" :
			fileBehavior = "tmp." +suffix + ".txt"
	else :
		if ind > -1 :
			fileBehavior = "behavior." + str(ind) + ".txt"
		else :
			fileBehavior = "tmp.txt"

		# if suffix != "" :
		# 	fileBehavior = "tmp." +suffix + ".txt"

	print('Reading data...')
	for d in directories :
		if os.path.isdir(d) :
			if RECURSIVE :
				tuplesWalk = os.walk(d)

				for t in tuplesWalk :
					listReplicates = [replicate for replicate in t[1] if regRep.match(replicate)]

					if len(listReplicates) > 0 :
						dictReplicates = {}
						for replicate in listReplicates :
							m = regRep.match(replicate)

							replicateDir = os.path.join(t[0], replicate)
							numReplicate = int(m.group(1))

							listTrajectories = list()

							if ind == -1 :
								numInd = -1
								dirInfo = os.path.join(FOLDER_INFO, os.path.join(os.path.basename(d), "Dir" + str(numReplicate)))
								print("DirInfo = " + dirInfo)

								if os.path.isdir(dirInfo) :
									if os.path.isfile(os.path.join(dirInfo, "fileInfo.txt")) :
										with open(os.path.join(dirInfo, "fileInfo.txt"), "r") as fileRead :
											fileRead = fileRead.readlines()
											s = regFileInfo.search(fileRead[1].rstrip('\n'))

											if s :
												numInd = int(s.group(1))

								if numInd > -1 :
									print("\t-> Best individual is " + str(numInd))
									fileBehavior = "behavior." + str(numInd) + ".txt"
								else :
									print('Couldn\'t find best individual info in ' + dirInfo)

							if os.path.isfile(os.path.join(replicateDir, fileBehavior)) :
								with open(os.path.join(replicateDir, fileBehavior), 'r') as fileRead :
									fileRead = fileRead.readlines()

									for line in fileRead :
										dictTime = {}
										lineSplit = line.split(' ')

										if len(lineSplit) > 0 :
											dictTime['time'] = int(lineSplit[0])

											dictTime['fitness'] = float(lineSplit[4])

											dictTime['targetPos'] = float(lineSplit[1])
											dictTime['senderPos'] = float(lineSplit[5])
											dictTime['receiverPos'] = float(lineSplit[6])

											dictTime['senderSignal'] = float(lineSplit[8])
											dictTime['receiverSignal'] = float(lineSplit[9])

											listTrajectories.append(dictTime)

							dictReplicates[numReplicate] = listTrajectories

						hashData[os.path.basename(t[0])] = dictReplicates
			else :
				listReplicates = [replicate for replicate in os.listdir(d) if regRep.match(replicate)]

				if len(listReplicates) > 0 :
					dictReplicates = {}
					for replicate in listReplicates :
						m = regRep.match(replicate)

						replicateDir = os.path.join(d, replicate)
						numReplicate = int(m.group(1))

						listTrajectories = list()

						if ind == -1 :
							numInd = -1
							dirInfo = os.path.join(FOLDER_INFO, os.path.join(os.path.basename(d), "Dir" + str(numReplicate)))
							print("DirInfo = " + dirInfo)
							if os.path.isdir(dirInfo) :
								if os.path.isfile(os.path.join(dirInfo, "fileInfo.txt")) :
									with open(os.path.join(dirInfo, "fileInfo.txt"), "r") as fileRead :
										fileRead = fileRead.readlines()
										s = regFileInfo.search(fileRead[1].rstrip('\n'))

										if s :
											numInd = int(s.group(1))

							if numInd > -1 :
								print("\t-> Best individual is " + str(numInd))
								fileBehavior = "behavior." + str(numInd) + ".txt"
							else :
								print('Couldn\'t find best individual info in ' + dirInfo)

						if os.path.isfile(os.path.join(replicateDir, fileBehavior)) :
							with open(os.path.join(replicateDir, fileBehavior), 'r') as fileRead :
								fileRead = fileRead.readlines()

								for line in fileRead :
									dictTime = {}
									lineSplit = line.split(' ')

									if len(lineSplit) > 0 :
										dictTime['time'] = int(lineSplit[0])

										dictTime['fitness'] = float(lineSplit[4])

										dictTime['targetPos'] = float(lineSplit[1])
										dictTime['senderPos'] = float(lineSplit[5])
										dictTime['receiverPos'] = float(lineSplit[6])

										dictTime['senderSignal'] = float(lineSplit[8])
										dictTime['receiverSignal'] = float(lineSplit[9])

										listTrajectories.append(dictTime)

						dictReplicates[numReplicate] = listTrajectories

					hashData[os.path.basename(d)] = dictReplicates

				listFiles = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]


	print("Drawing plots...")
	for exp in sorted(hashData.keys()) :
		outputData = OUTPUT_DIR + "/" + exp

		if os.path.isdir(outputData) :
			shutil.rmtree(outputData)

		os.makedirs(outputData)

		dataExp = hashData[exp]

		listSelectionReplicates = dataExp.keys()
		if REPLICATES != None :
			listSelectionReplicates = REPLICATES

		# figsize = (880/DPI, 640/DPI)
		figsize = FIGSIZE


		for replicate in listSelectionReplicates :
			# ------------ TRAJECTORY PLOT ------------
			fig, axe1 = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, dpi = DPI)
			# plt.axes(frameon=0)
			plt.grid()

			indexToPlot = range(len(dataExp[replicate]))
			if len(TRIALS) < 5 :
				indexToPlot = list()
				for numTrial in TRIALS :
					indexToPlot += list(range((numTrial - 1)*100, numTrial*100))

			# Sender and receiver positions
			dataToPlotSender = [float(dataExp[replicate][step]['senderPos']) for step in indexToPlot]
			dataToPlotReceiver = [dataExp[replicate][step]['receiverPos'] for step in indexToPlot]

			# print(dataToPlotSender)

			axe1.plot(range(len(dataToPlotSender)), dataToPlotSender, color=palette[1], linestyle='none', marker='o', markersize=12)
			axe1.plot(range(len(dataToPlotReceiver)), dataToPlotReceiver, color=palette[2], linestyle='none', marker='o', markersize=12)

			# Comm area position
			commAreaLow = [-math.pi/4 for step in indexToPlot]
			commAreaHigh = [math.pi/4 for step in indexToPlot]
			axe1.plot(range(len(commAreaLow)), commAreaLow, color=COMM_AREA_COLOR, linestyle='--', linewidth=linewidth)
			axe1.plot(range(len(commAreaHigh)), commAreaHigh, color=COMM_AREA_COLOR, linestyle='--', linewidth=linewidth)
			axe1.fill_between(range(len(commAreaLow)), commAreaLow, commAreaHigh, alpha=0.10, linewidth=0, color=COMM_AREA_COLOR)

			# Target position
			targetPosLow1 = [dataExp[replicate][step]['targetPos'] - (math.pi/8) for step in indexToPlot]
			targetPos = [dataExp[replicate][step]['targetPos'] for step in indexToPlot]
			targetPosHigh1 = targetPos

			targetPosHigh2 = [normalizeAngle(dataExp[replicate][step]['targetPos'] + (math.pi/8)) for step in indexToPlot]
			targetPosLow2 = list(map(lambda x : x - math.pi/8., targetPosHigh2))

			axe1.plot(range(len(targetPosLow1)), targetPosLow1, color=TARGET_COLOR, linestyle='--', linewidth=linewidth)
			# axe1.plot(range(len(targetPos)), targetPos, color=TARGET_COLOR, linestyle='-', linewidth=3)
			axe1.plot(range(len(targetPosHigh2)), targetPosHigh2, color=TARGET_COLOR, linestyle='--', linewidth=linewidth)

			axe1.fill_between(range(len(targetPos)), targetPosLow1, targetPosHigh1, alpha=0.10, linewidth=0, color=TARGET_COLOR)
			axe1.fill_between(range(len(targetPos)), targetPosLow2, targetPosHigh2, alpha=0.10, linewidth=0, color=TARGET_COLOR)



			# print(dataExp[replicate][0]['targetPos'])
			# if dataExp[replicate][0]['targetPos'] > 3.*math.pi/4. :
			# 	print("beh ?")
			# 	targetPosLow1 = [dataExp[replicate][step]['targetPos'] - (math.pi/8) for step in indexToPlot]
			# 	targetPos = [dataExp[replicate][step]['targetPos'] for step in indexToPlot]
			# 	targetPosHigh1 = targetPos

			# 	targetPosLow2 = [dataExp[replicate][step]['targetPos'] - 2.*math.pi for step in indexToPlot]
			# 	targetPosHigh2 = [dataExp[replicate][step]['targetPos'] - 2.*math.pi + (math.pi/8) for step in indexToPlot]

			# 	axe1.plot(range(len(targetPosLow1)), targetPosLow, color=TARGET_COLOR, linestyle='--', linewidth=linewidth)
			# 	axe1.plot(range(len(targetPos)), targetPos, color=TARGET_COLOR, linestyle='-', linewidth=3)
			# 	axe1.plot(range(len(targetPosHigh2)), targetPosHigh, color=TARGET_COLOR, linestyle='--', linewidth=linewidth)

			# 	axe1.fill_between(range(len(targetPos)), targetPosLow1, targetPosHigh1, alpha=0.10, linewidth=0, color=TARGET_COLOR)
			# 	axe1.fill_between(range(len(targetPos)), targetPosLow2, targetPosHigh2, alpha=0.10, linewidth=0, color=TARGET_COLOR)
			# else :
			# 	targetPosLow = [dataExp[replicate][step]['targetPos'] - (math.pi/8) for step in indexToPlot]
			# 	targetPos = [dataExp[replicate][step]['targetPos'] for step in indexToPlot]
			# 	targetPosHigh = [dataExp[replicate][step]['targetPos'] + (math.pi/8) for step in indexToPlot]

			# 	axe1.plot(range(len(targetPosLow)), targetPosLow, color=TARGET_COLOR, linestyle='--', linewidth=linewidth)
			# 	axe1.plot(range(len(targetPos)), targetPos, color=TARGET_COLOR, linestyle='-', linewidth=3)
			# 	axe1.plot(range(len(targetPosHigh)), targetPosHigh, color=TARGET_COLOR, linestyle='--', linewidth=linewidth)
			# 	axe1.fill_between(range(len(targetPos)), targetPosLow, targetPosHigh, alpha=0.10, linewidth=0, color=TARGET_COLOR)

			xticks = list(range(0, len(indexToPlot), 20))
			if len(indexToPlot) - 1 not in xticks :
				xticks.append(len(indexToPlot) - 1)
			xticksLabels = [dataExp[replicate][t]['time'] for t in xticks]
			
			cpt = 0
			while cpt < len(xticksLabels) :
				if xticksLabels[cpt] == 0 :
					xticksLabels[cpt] = 1
				elif xticksLabels[cpt] == 99 :
					xticksLabels[cpt] = 100
				cpt += 1

			axe1.set_xticks(xticks)
			axe1.set_xticklabels(xticksLabels)
			axe1.set_xlabel("Time")

			axe1.set_ylabel("Position")
			axe1.set_yticks([-math.pi, -math.pi/2., 0., math.pi/2., math.pi])
			axe1.set_yticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", "$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

			# axe1.set_yticks([-math.pi, -3.*math.pi/4., -math.pi/2., -math.pi/4., 0., math.pi/4., math.pi/2., 3.*math.pi/4., math.pi])
			# axe1.set_yticklabels([r"$-\pi$", r"$\frac{-3\pi}{4}$", r"$\frac{-\pi}{2}$", r"$\frac{-\pi}{4}$", "$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
			# axe1.set_yticklabels([r"$\mathrm{-\pi}$", r"$\mathrm{\frac{-3\pi}{4}}$", r"$\mathrm{\frac{-\pi}{2}}$", r"$\mathrm{\frac{-\pi}{4}}$", "$\mathrm{0}$", r"$\mathrm{\frac{\pi}{4}}$", r"$\mathrm{\frac{\pi}{2}}$", r"$\mathrm{\frac{3\pi}{4}}$", r"$\mathrm{\pi}$"])
			# axe1.set_yticklabels([r"$\mathdefault{-\pi}$", r"$\mathdefault{\frac{-3\pi}{4}}$", r"$\mathdefault{\frac{-\pi}{2}}$", r"$\mathdefault{\frac{-\pi}{4}}$", "$\mathdefault{0}$", r"$\mathdefault{\frac{\pi}{4}}$", r"$\mathdefault{\frac{\pi}{2}}$", r"$\mathdefault{\frac{3\pi}{4}}$", r"$\mathdefault{\pi}$"])
			axe1.set_ylim(-math.pi, math.pi)
			axe1.set_xlim(0, xticks[-1])

			legend = plt.legend(['Sender', 'Receiver'], loc = 1, frameon=True)
			# legend = plt.legend(['Sender', 'Receiver'], loc = 4, frameon=True)
			frame = legend.get_frame()
			frame.set_facecolor('0.9')
			frame.set_edgecolor('0.9')

			plt.savefig(outputData + "/trajectory" + suffix + "Run" + str(replicate) + ".png", bbox_inches = 'tight', dpi = DPI)
			plt.savefig(outputData + "/trajectory" + suffix + "Run" + str(replicate) + ".svg", bbox_inches = 'tight', dpi = DPI)
			plt.close()

			# sys.exit()
	
			# ---------- SIGNAL AMPLITUDE PLOT ------------
			fig, axe1 = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, dpi = DPI)
			# plt.axes(frameon=0)
			plt.grid()

			indexToPlot = range(len(dataExp[replicate]))
			if len(TRIALS) < 5 :
				indexToPlot = list()
				for numTrial in TRIALS :
					indexToPlot += range((numTrial - 1)*100, numTrial*100)

			# Sender and receiver positions
			dataToPlotSender = [dataExp[replicate][step]['senderSignal'] for step in indexToPlot]
			# dataToPlotSender = [dataExp[replicate][step]['receiverSignal'] for step in indexToPlot]
			dataSenderPos = [dataExp[replicate][step]['senderPos'] for step in indexToPlot]
			dataReceiverPos = [dataExp[replicate][step]['receiverPos'] for step in indexToPlot]

			axe1.plot(range(len(dataToPlotSender)), dataToPlotSender, color=palette[1], linestyle='none', marker='o', markersize='12')

			# Steps when both sender and receiver are in comm area
			senderInCom = [step for step in range(0, len(dataSenderPos)) if (dataSenderPos[step] >= -math.pi/4) and (dataSenderPos[step] <= math.pi/4)]
			receiverInCom = [step for step in range(0, len(dataReceiverPos)) if (dataReceiverPos[step] >= -math.pi/4) and (dataReceiverPos[step] <= math.pi/4)]
			bothInComm = [step for step in senderInCom if step in receiverInCom]

			listZones = list()
			previous = None
			curZone = None
			cpt = 0
			for step in bothInComm :
				if previous == None :
					previous = step
					curZone = [step]
				else :
					if step == previous + 1 :
						curZone.append(step)
					else :
						listZones.append(curZone)
						curZone = [step]

					previous = step

				if cpt == len(bothInComm) - 1 :
					listZones.append(curZone)

				cpt += 1

			for zone in listZones :
				axe1.fill_between(zone, 0, 1, color=COMM_AREA_COLOR, alpha=0.10, linewidth=0)
				# axe1.fill_betweenx(range(0, len(zone)), [zone[0] for step in zone], [zone[1] for step in zone], color=COMM_AREA_COLOR, alpha=0.10, linewidth=0)

			xticks = range(0, len(indexToPlot), 20)
			if len(indexToPlot) - 1 not in xticks :
				xticks.append(len(indexToPlot) - 1)
			xticksLabels = [dataExp[replicate][t]['time'] for t in xticks]
			
			cpt = 0
			while cpt < len(xticksLabels) :
				if xticksLabels[cpt] == 0 :
					xticksLabels[cpt] = 1
				elif xticksLabels[cpt] == 99 :
					xticksLabels[cpt] = 100
				cpt += 1

			axe1.set_xticks(xticks)
			axe1.set_xticklabels(xticksLabels)
			axe1.set_xlabel("Time")

			axe1.set_ylabel("Amplitude")
			axe1.set_ylim(0, 1)

			# legend = plt.legend(['Sender', 'Receiver'], loc = 1, frameon=True)
			# frame = legend.get_frame()
			# frame.set_facecolor('0.9')
			# frame.set_edgecolor('0.9')

			plt.savefig(outputData + "/signalAmplitude" + suffix + "Run" + str(replicate) + ".png", bbox_inches = 'tight', dpi = DPI)
			plt.savefig(outputData + "/signalAmplitude" + suffix + "Run" + str(replicate) + ".svg", bbox_inches = 'tight', dpi = DPI)
			plt.close()

def add_text(image, text, position, font_type, font_size, color):
	# Convert the image to RGB (OpenCV uses BGR)  
	cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
	 
	# Pass the image to PIL  
	pil_im = Image.fromarray(cv2_im_rgb)  
	 
	draw = ImageDraw.Draw(pil_im)

	# use a truetype font  
	font = ImageFont.truetype(font_type, font_size)  
	 
	# Draw the text  
	draw.text((position[0], position[1] - math.ceil(font_size / 2)), text, font=font, fill = (color[2], color[1], color[0]), align = "left")  
	 
	# Get back the image to OpenCV  
	image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  

	return image

def draw_environment(
	image, 
	circle_color, 
	circle_thickness, 
	radius, 
	communication_area, 
	communication_area_color,
	com_thickness,
	drawing_size,
	agent_radius,
	sender_color,
	sender_outer,
	receiver_color,
	receiver_outer,
	**drawing_arguments
	):
	# Circle
	image = cv2.circle(image, (drawing_size[0]//2, drawing_size[1]//2), radius, circle_color, circle_thickness)

	# Communication area
	image = cv2.ellipse(image , (drawing_size[0]//2, drawing_size[1]//2) , (radius, radius), 0, communication_area[1], communication_area[0], communication_area_color, com_thickness)

	# Legend
	# Sender
	image = cv2.circle(image, (drawing_size[0] + 25, 75), agent_radius, sender_color, -1)
	image = cv2.circle(image, (drawing_size[0] + 25, 75), agent_radius, sender_outer, 2)
	image = add_text(image, 'SENDER', (drawing_size[0] + 25 + agent_radius * 2, 75), "arialbd.ttf", 25, sender_outer) 

	# Receiver
	image = cv2.circle(image, (drawing_size[0] + 25, 125), agent_radius, receiver_color, -1)
	image = cv2.circle(image, (drawing_size[0] + 25, 125), agent_radius, receiver_outer, 2)
	image = add_text(image, 'RECEIVER', (drawing_size[0] + 25 + agent_radius * 2, 125), "arialbd.ttf", 25, receiver_outer) 

	# Communication area
	image = cv2.ellipse(image , (drawing_size[0] + 25 - (radius // 10), 175) , (radius // 10, radius // 10), 0, communication_area[1], communication_area[0], communication_area_color, com_thickness // 2)
	image = add_text(image, 'COM. AREA', (drawing_size[0] + 25 + agent_radius * 2, 175), "arialbd.ttf", 25, communication_area_color) 

	# foraging site
	image = cv2.ellipse(image , (drawing_size[0] + 25 - (radius // 10), 225) , (radius // 10, radius // 10), 0, communication_area[1], communication_area[0], drawing_arguments['foraging_site_color'], drawing_arguments['foraging_thickness'] // 2)
	image = add_text(image, 'FOOD SITE', (drawing_size[0] + 25 + agent_radius * 2, 225), "arialbd.ttf", 25, drawing_arguments['foraging_site_color']) 

	return image

def draw_foraging_site(
	image, 
	targetPos,
	trial,
	radius, 
	foraging_thickness,
	foraging_site_color,
	drawing_size,
	**drawing_arguments):
	target_low = math.degrees(-targetPos + math.pi / 8)
	target_high = math.degrees(-targetPos - math.pi / 8)

	# Foraging site
	image = cv2.ellipse(image, (drawing_size[0]//2, drawing_size[1]//2) , (radius, radius), 0, target_low, target_high, foraging_site_color, foraging_thickness);

	# Foraging site position
	image = add_text(image, f'TRIAL {trial}', (5, 15), 'arialbd.ttf', 35, (0, 0, 0))
	# image = cv2.putText(image, f'TRIAL {trial}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness = 3)

	return image

def draw_step(
	image,
	step_number,
	drawing_size,
	**drawing_arguments):

	# Step label
	image = add_text(image, f'STEP', (drawing_size[0] - 150, 15), 'arialbd.ttf', 35, (0, 0, 0), )
	# image = cv2.putText(image, f'STEP', (drawing_size[0] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness = 3)

	# Step number
	image = add_text(image, f'{step_number + 1}', (drawing_size[0] - 45, 15), 'arialbd.ttf', 35, (0, 0, 0))
	# image = cv2.putText(image, f'{step_number + 1}', (drawing_size[0] - 65, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness = 2)

	return image

def draw_agent(
	image, 
	position,
	type,
	radius,
	agent_radius,
	sender_color,
	sender_outer,
	receiver_color,
	receiver_outer,
	drawing_size,
	**drawing_arguments):
	pos_x = int(math.cos(position) * radius + drawing_size[1]//2)
	pos_y = int(drawing_size[0]//2 - math.sin(position) * radius)

	if type == "sender":
		# Inside
		image = cv2.circle(image, (pos_x, pos_y), agent_radius, sender_color, -1)

		# Outer
		image = cv2.circle(image, (pos_x, pos_y), agent_radius, sender_outer, 2)
	else:
		# Inside
		image = cv2.circle(image, (pos_x, pos_y), agent_radius, receiver_color, -1)

		# Outer
		image = cv2.circle(image, (pos_x, pos_y), agent_radius, receiver_outer, 2)

	return image


def createVideo(directories, info, ind) :
	hashData = {}
	regRep = re.compile(r"^(\d+)$")
	regFileInfo = re.compile(r"^Individual (\d+) : ")

	suffix = ""
	if info is not None :
		if info == "onset" or info == "length" or info == "length2" or info == "length3" or info == "length4" :
			suffix = info
		else :
			print("drawBehavior: info " + str(info) + " unknown")

	fileBehavior = None
	if ind == None :
		fileBehavior = "tmp.txt"
		if suffix != "" :
			fileBehavior = "tmp." +suffix + ".txt"
	else :
		if ind > -1 :
			fileBehavior = "behavior." + str(ind) + ".txt"
		else :
			fileBehavior = "tmp.txt"

		# if suffix != "" :
		# 	fileBehavior = "tmp." +suffix + ".txt"

	print('Reading data...')
	for d in directories :
		if os.path.isdir(d) :
			if RECURSIVE :
				tuplesWalk = os.walk(d)

				for t in tuplesWalk :
					listReplicates = [replicate for replicate in t[1] if regRep.match(replicate)]

					if len(listReplicates) > 0 :
						dictReplicates = {}
						for replicate in listReplicates :
							m = regRep.match(replicate)

							replicateDir = os.path.join(t[0], replicate)
							numReplicate = int(m.group(1))

							listTrajectories = list()

							if ind == -1 :
								numInd = -1
								# dirInfo = os.path.join(FOLDER_INFO, os.path.join(os.path.basename(d), "Dir" + str(numReplicate)))
								dirInfo = os.path.join(FOLDER_INFO, os.path.join(os.path.basename(d), str(numReplicate)))
								print("DirInfo = " + dirInfo)

								if os.path.isdir(dirInfo) :
									if os.path.isfile(os.path.join(dirInfo, "fileInfo.txt")) :
										with open(os.path.join(dirInfo, "fileInfo.txt"), "r") as fileRead :
											fileRead = fileRead.readlines()
											s = regFileInfo.search(fileRead[1].rstrip('\n'))

											if s :
												numInd = int(s.group(1))

								if numInd > -1 :
									print("\t-> Best individual is " + str(numInd))
									fileBehavior = "behavior." + str(numInd) + ".txt"
								else :
									print('Couldn\'t find best individual info in ' + dirInfo)

							if os.path.isfile(os.path.join(replicateDir, fileBehavior)) :
								with open(os.path.join(replicateDir, fileBehavior), 'r') as fileRead :
									fileRead = fileRead.readlines()

									for line in fileRead :
										dictTime = {}
										lineSplit = line.split(' ')

										if len(lineSplit) > 0 :
											dictTime['time'] = int(lineSplit[0])

											dictTime['fitness'] = float(lineSplit[4])

											dictTime['targetPos'] = float(lineSplit[1])
											dictTime['senderPos'] = float(lineSplit[5])
											dictTime['receiverPos'] = float(lineSplit[6])

											dictTime['senderSignal'] = float(lineSplit[8])
											dictTime['receiverSignal'] = float(lineSplit[9])

											listTrajectories.append(dictTime)

							dictReplicates[numReplicate] = listTrajectories

						hashData[os.path.basename(t[0])] = dictReplicates
			else :
				listReplicates = [replicate for replicate in os.listdir(d) if regRep.match(replicate)]

				if len(listReplicates) > 0 :
					dictReplicates = {}
					for replicate in listReplicates :
						m = regRep.match(replicate)

						replicateDir = os.path.join(d, replicate)
						numReplicate = int(m.group(1))

						listTrajectories = list()

						if ind == -1 :
							numInd = -1
							# dirInfo = os.path.join(FOLDER_INFO, os.path.join(os.path.basename(d), "Dir" + str(numReplicate)))
							dirInfo = os.path.join(FOLDER_INFO, os.path.join(os.path.basename(d), str(numReplicate)))
							print("DirInfo = " + dirInfo)
							if os.path.isdir(dirInfo) :
								if os.path.isfile(os.path.join(dirInfo, "fileInfo.txt")) :
									with open(os.path.join(dirInfo, "fileInfo.txt"), "r") as fileRead :
										fileRead = fileRead.readlines()
										s = regFileInfo.search(fileRead[1].rstrip('\n'))

										if s :
											numInd = int(s.group(1))

							if numInd > -1 :
								print("\t-> Best individual is " + str(numInd))
								fileBehavior = "behavior." + str(numInd) + ".txt"
							else :
								print('Couldn\'t find best individual info in ' + dirInfo)

						if os.path.isfile(os.path.join(replicateDir, fileBehavior)) :
							with open(os.path.join(replicateDir, fileBehavior), 'r') as fileRead :
								fileRead = fileRead.readlines()

								for line in fileRead :
									dictTime = {}
									lineSplit = line.split(' ')

									if len(lineSplit) > 0 :
										dictTime['time'] = int(lineSplit[0])

										dictTime['fitness'] = float(lineSplit[4])

										dictTime['targetPos'] = float(lineSplit[1])
										dictTime['senderPos'] = float(lineSplit[5])
										dictTime['receiverPos'] = float(lineSplit[6])

										dictTime['senderSignal'] = float(lineSplit[8])
										dictTime['receiverSignal'] = float(lineSplit[9])

										listTrajectories.append(dictTime)

						dictReplicates[numReplicate] = listTrajectories

					hashData[os.path.basename(d)] = dictReplicates

				listFiles = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]


	output_directory = Path(OUTPUT_DIR)

	print("Drawing plots...")
	for exp in tqdm(sorted(hashData.keys())) :
		outputData = output_directory / exp
		outputData.mkdir(exist_ok = True, parents = True)

		dataExp = hashData[exp]

		listSelectionReplicates = dataExp.keys()
		if REPLICATES != None :
			listSelectionReplicates = REPLICATES

		# figsize = (880/DPI, 640/DPI)
		figsize = FIGSIZE

		drawing_arguments = {
			"drawing_size" : (500, 500),
			"circle_color" : (0, 0, 0),
			"circle_thickness" : 6,
			"radius" : 200,
			"communication_area" : (-45, 45),
			"communication_area_color" : (255, 0, 0),
			"com_thickness" : 8,
			"foraging_thickness": 8,
			"foraging_site_color": (0, 0, 255),
			"agent_radius": 10,
			"sender_color": (0, 0, 255),
			"sender_outer": (0, 0, 200),
			"receiver_color": (255, 255, 0),
			"receiver_outer": (200, 200, 0)
		}

		environment_image = np.ones((VIDEO_SIZE[0], VIDEO_SIZE[1], 3), np.uint8) * 255
		# environment_image = np.ones((750, 750, 3), np.uint8) * 255
		environment_image = draw_environment(environment_image, **drawing_arguments)

		for replicate in tqdm(listSelectionReplicates, total = len(listSelectionReplicates)) :
			img_list = []

			# indexToPlot = list()
			# if len(TRIALS) > 1:
			# 	print('Can only make a video from a single trial. Selecting the first one')

			# numTrial = list(TRIALS).pop(0)
			# indexToPlot += range((numTrial - 1)*100, numTrial*100)

			# indexToPlot = range(len(dataExp[replicate]))
			# if len(TRIALS) < 5 :
			# 	indexToPlot = list()
			# 	for numTrial in TRIALS :
			# 		indexToPlot += list(range((numTrial - 1)*100, numTrial*100))

			# video_out = outputData / f"video_rep{replicate}.avi"
			video_out = outputData / f"video_rep{replicate}.mp4"
			video = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*'DIVX'), FPS, VIDEO_SIZE)
			for trial in TRIALS:
				indexToPlot = list(range((trial - 1)*100, trial*100))

				targetPos = dataExp[replicate][indexToPlot[0]]['targetPos']
				trial_image = draw_foraging_site(np.copy(environment_image), targetPos, trial, **drawing_arguments)

				# Sender and receiver positions
				dataToPlotSender = [float(dataExp[replicate][step]['senderPos']) for step in indexToPlot]
				dataToPlotReceiver = [dataExp[replicate][step]['receiverPos'] for step in indexToPlot]

				# Wait frames
				WAIT = 20
				for _ in range(WAIT):
					step_image = draw_step(np.copy(trial_image), -1, **drawing_arguments)
					step_image = draw_agent(step_image, 0, "sender", **drawing_arguments)
					step_image = draw_agent(step_image, 0, "receiver", **drawing_arguments)
					video.write(step_image)

				for step in range(len(dataToPlotSender)):
					step_image = draw_step(np.copy(trial_image), step, **drawing_arguments)

					sender_pos = dataToPlotSender[step]
					receiver_pos = dataToPlotReceiver[step]

					step_image = draw_agent(step_image, sender_pos, "sender", **drawing_arguments)
					step_image = draw_agent(step_image, receiver_pos, "receiver", **drawing_arguments)

					video.write(step_image)

			video.release()


			# ------------ TRAJECTORY & SIGNAL AMPLITUDE PLOT ------------
			# fig, axe1 = plt.subplots(nrows = 2, ncols = len(TRIALS), figsize = (1600 / DPI, 840 / DPI), dpi = DPI, sharex = 'col', sharey = 'row', squeeze = False)
			fig, axe1 = plt.subplots(nrows = 1, ncols = len(TRIALS), figsize = (1600 / DPI, 440 / DPI), dpi = DPI, sharex = 'col', sharey = 'row', squeeze = False)
			# plt.axes(frameon=0)
			plt.grid()
			for index_trial, trial in enumerate(TRIALS):
				indexToPlot = list(range((trial - 1)*100, trial*100))

				# Sender and receiver positions
				dataToPlotSender = [float(dataExp[replicate][step]['senderPos']) for step in indexToPlot]
				dataToPlotReceiver = [dataExp[replicate][step]['receiverPos'] for step in indexToPlot]

				# MAX_DIST = math.pi
				# diff_sender = [dataToPlotSender[i] - dataToPlotSender[i - 1] for i in range(1, len(dataToPlotSender))]
				# print(dataToPlotSender)
				# discontinued = np.where(np.abs(np.array(diff_sender)) > MAX_DIST)

				# last_index = 0
				# for index_discontinuity in discontinued[0].flatten():
				# 	dataPlot = dataToPlotSender[last_index:index_discontinuity + 1]
				# 	axe1[index_trial].plot(range(last_index, last_index + len(dataPlot)), dataPlot, color=palette[1], linestyle='-', linewidth = 3.0)
				# 	last_index = index_discontinuity + 1

				axe1[0, index_trial].plot(range(len(dataToPlotSender)), dataToPlotSender, color=palette[1], linestyle='none', marker='o', markersize=12)
				axe1[0, index_trial].plot(range(len(dataToPlotReceiver)), dataToPlotReceiver, color=palette[2], linestyle='none', marker='o', markersize=12)
				# axe1[0, index_trial].plot(range(len(dataToPlotSender)), dataToPlotSender, color=palette[1], linestyle='-', linewidth = 3.0)
				# axe1[0, index_trial].plot(range(len(dataToPlotReceiver)), dataToPlotReceiver, color=palette[2], linestyle='-', linewidth = 3.0)

				# Comm area position
				commAreaLow = [-math.pi/4 for step in indexToPlot]
				commAreaHigh = [math.pi/4 for step in indexToPlot]
				axe1[0, index_trial].plot(range(len(commAreaLow)), commAreaLow, color=COMM_AREA_COLOR, linestyle='--', linewidth=linewidth)
				axe1[0, index_trial].plot(range(len(commAreaHigh)), commAreaHigh, color=COMM_AREA_COLOR, linestyle='--', linewidth=linewidth)
				axe1[0, index_trial].fill_between(range(len(commAreaLow)), commAreaLow, commAreaHigh, alpha=0.10, linewidth=0, color=COMM_AREA_COLOR)

				# Target position
				targetPosLow1 = [dataExp[replicate][step]['targetPos'] - (math.pi/8) for step in indexToPlot]
				targetPos = [dataExp[replicate][step]['targetPos'] for step in indexToPlot]
				targetPosHigh1 = targetPos

				targetPosHigh2 = [normalizeAngle(dataExp[replicate][step]['targetPos'] + (math.pi/8)) for step in indexToPlot]
				targetPosLow2 = list(map(lambda x : x - math.pi/8., targetPosHigh2))

				axe1[0, index_trial].plot(range(len(targetPosLow1)), targetPosLow1, color=TARGET_COLOR, linestyle='--', linewidth=linewidth)
				# axe1[0, index_trial].plot(range(len(targetPos)), targetPos, color=TARGET_COLOR, linestyle='-', linewidth=3)
				axe1[0, index_trial].plot(range(len(targetPosHigh2)), targetPosHigh2, color=TARGET_COLOR, linestyle='--', linewidth=linewidth)

				axe1[0, index_trial].fill_between(range(len(targetPos)), targetPosLow1, targetPosHigh1, alpha=0.10, linewidth=0, color=TARGET_COLOR)
				axe1[0, index_trial].fill_between(range(len(targetPos)), targetPosLow2, targetPosHigh2, alpha=0.10, linewidth=0, color=TARGET_COLOR)

				xticks = list(range(0, len(indexToPlot), 20))
				if len(indexToPlot) - 1 not in xticks :
					xticks.append(len(indexToPlot) - 1)
				xticksLabels = [dataExp[replicate][t]['time'] for t in xticks]
				
				cpt = 0
				while cpt < len(xticksLabels) :
					if xticksLabels[cpt] == 0 :
						xticksLabels[cpt] = 1
					elif xticksLabels[cpt] == 99 :
						xticksLabels[cpt] = 100
					cpt += 1

				axe1[0, index_trial].set_xticks(xticks)
				axe1[0, index_trial].set_xticklabels(xticksLabels)
				axe1[0, index_trial].set_xlabel("Time")

				axe1[0, index_trial].set_xlim(xticks[0], xticks[-1])

				if index_trial == 0:
					axe1[0, index_trial].set_ylabel("Position")
					axe1[0, index_trial].set_yticks([-math.pi, -math.pi/2., 0., math.pi/2., math.pi])
					axe1[0, index_trial].set_yticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", "$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

					axe1[0, index_trial].set_ylim(-math.pi, math.pi)

				axe1[0, index_trial].grid(True)

			# legend = plt.legend(['Sender', 'Receiver'], loc = 1, frameon=True)
			# # legend = plt.legend(['Sender', 'Receiver'], loc = 4, frameon=True)
			# frame = legend.get_frame()
			# frame.set_facecolor('0.9')
			# frame.set_edgecolor('0.9')

			# verif_out = outputData / f"verif_rep{replicate}.png"
			# plt.savefig(str(verif_out), bbox_inches = 'tight', dpi = DPI)
			# plt.close()

			# ---------- SIGNAL AMPLITUDE PLOT ------------
			# fig, axe1 = plt.subplots(nrows = 1, ncols = len(TRIALS), figsize = figsize, dpi = DPI, sharey = True)
			# # plt.axes(frameon=0)
			# plt.grid()
			# for index_trial, trial in enumerate(TRIALS):
				# indexToPlot = list(range((trial - 1)*100, trial*100))

				# Sender and receiver positions
				# dataToPlotSender = [dataExp[replicate][step]['senderSignal'] for step in indexToPlot]
				# dataToPlotSender = [dataExp[replicate][step]['receiverSignal'] for step in indexToPlot]
				# dataSenderPos = [dataExp[replicate][step]['senderPos'] for step in indexToPlot]
				# dataReceiverPos = [dataExp[replicate][step]['receiverPos'] for step in indexToPlot]

				# axe1[1, index_trial].plot(range(len(dataToPlotSender)), dataToPlotSender, color=palette[1], linestyle='-', linewidth = 3.0) #, marker='o', markersize='12')

				# # Steps when both sender and receiver are in comm area
				# senderInCom = [step for step in range(0, len(dataSenderPos)) if (dataSenderPos[step] >= -math.pi/4) and (dataSenderPos[step] <= math.pi/4)]
				# receiverInCom = [step for step in range(0, len(dataReceiverPos)) if (dataReceiverPos[step] >= -math.pi/4) and (dataReceiverPos[step] <= math.pi/4)]
				# bothInComm = [step for step in senderInCom if step in receiverInCom]

				# listZones = list()
				# previous = None
				# curZone = None
				# cpt = 0
				# for step in bothInComm :
				# 	if previous == None :
				# 		previous = step
				# 		curZone = [step]
				# 	else :
				# 		if step == previous + 1 :
				# 			curZone.append(step)
				# 		else :
				# 			listZones.append(curZone)
				# 			curZone = [step]

				# 		previous = step

				# 	if cpt == len(bothInComm) - 1 :
				# 		listZones.append(curZone)

				# 	cpt += 1

				# for zone in listZones :
				# 	axe1[1, index_trial].fill_between(zone, 0, 1, color=COMM_AREA_COLOR, alpha=0.10, linewidth=0)
				# 	# axe1[1, index_trial].fill_betweenx(range(0, len(zone)), [zone[0] for step in zone], [zone[1] for step in zone], color=COMM_AREA_COLOR, alpha=0.10, linewidth=0)

				# xticks = list(range(0, len(indexToPlot), 20))
				# if len(indexToPlot) - 1 not in xticks :
				# 	xticks.append(len(indexToPlot) - 1)
				# xticksLabels = [dataExp[replicate][t]['time'] for t in xticks]
				
				# cpt = 0
				# while cpt < len(xticksLabels) :
				# 	if xticksLabels[cpt] == 0 :
				# 		xticksLabels[cpt] = 1
				# 	elif xticksLabels[cpt] == 99 :
				# 		xticksLabels[cpt] = 100
				# 	cpt += 1

				# axe1[1, index_trial].set_xticks(xticks)
				# axe1[1, index_trial].set_xticklabels(xticksLabels)
				# axe1[1, index_trial].set_xlabel("Time")

				# axe1[1, index_trial].set_xlim(xticks[0], xticks[-1])

				# if index_trial == 0:
				# 	axe1[1, index_trial].set_ylabel("Amplitude")
				# 	axe1[1, index_trial].set_ylim(0, 1)

				# axe1[1, index_trial].grid(True)

			# legend = plt.legend(['Sender', 'Receiver'], loc = 1, frameon=True)
			# frame = legend.get_frame()
			# frame.set_facecolor('0.9')
			# frame.set_edgecolor('0.9')

			verif_out = outputData / f"behavior_rep{replicate}.png"
			plt.tight_layout()
			plt.savefig(str(verif_out), dpi = DPI)
			# plt.savefig(str(verif_out), bbox_inches = 'tight', dpi = DPI)
			# plt.savefig(outputData + "/signalAmplitude" + suffix + "Run" + str(replicate) + ".png", bbox_inches = 'tight', dpi = DPI)
			# plt.savefig(outputData + "/signalAmplitude" + suffix + "Run" + str(replicate) + ".svg", bbox_inches = 'tight', dpi = DPI)
			plt.close()


def main(args) :
	if args.recursive :
		global RECURSIVE
		RECURSIVE = True

	if args.output is not None :
		global OUTPUT_DIR
		OUTPUT_DIR = args.output

	if args.trials is not None :
		global TRIALS
		TRIALS = args.trials

	if args.precision is not None :
		global PRECISION
		PRECISION = args.precision

	if args.selection is not None :
		global REPLICATES
		REPLICATES = args.selection

	if args.max is not None :
		global MAX
		MAX = args.max

	if args.avg :
		drawAvgFit(args.directories)
	elif args.comp :
		drawFitComp(args.directories)
	elif args.behavior :
		drawBehavior(args.directories, args.info, args.ind)
	elif args.video:
		createVideo(args.directories, args.info, args.ind)
	elif args.performance :
		drawAvgPerf(args.directories)
	elif args.signal :
		drawSignalComp(args.directories)
	elif args.threshold is not None :
		drawGenThreshold(args.directories, args.threshold)


def normalizeAngle(angle) :
	while(angle > math.pi) :
		angle -= 2.*math.pi
	while(angle < -math.pi) :
		angle += 2.*math.pi

	return angle


if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument('directories', help = "Directories to plot", type=str, nargs='+')
	parser.add_argument('-r', '--recursive', help = "Draw for all subdirectories too", default = False, action = 'store_true')

	action = parser.add_mutually_exclusive_group(required = True)
	action.add_argument('-a', '--avg', help = "Draw average performance", default = False, action = "store_true")
	action.add_argument('-f', '--performance', help = "Draw average performance plots", default = False, action = "store_true")
	action.add_argument('-c', '--comp', help = "Draw comparative performance", default = False, action = "store_true")
	action.add_argument('-s', '--signal', help = "Draw comparative signal behaviors", default = False, action = "store_true")
	action.add_argument('-b', '--behavior', help = "Draw behavior", default = False, action = "store_true")
	action.add_argument('-e', '--threshold', help = "Draw time before threshold of performance crossed", default = None, type = float)
	action.add_argument('-v', '--video', help = "Generate behaviour video", default = False, action = 'store_true')

	parser.add_argument('-t', '--trials', help = "Trials to plot", default=None, type=int, nargs='+')
	parser.add_argument('-p', '--precision', help = "Precision of generation to draw", default=None, type=int)
	parser.add_argument('-m', '--max', help = "Maximum of generations to draw", default=None, type=int)
	parser.add_argument('-o', '--output', help = "Output directory", type = str)
	parser.add_argument('-i', '--info', help = "Supplementary info", type = str)
	parser.add_argument('-n', '--ind', help = "Individual whose behavior we want to see", type = int, default = None)
	parser.add_argument('-l', '--selection', help = "Selected runs", default=None, type=int, nargs='+')
	# parser.add_argument('-e', '--exclusion', help = "Excluded runs", default=None, type=int, nargs='+')

	args = parser.parse_args()
	main(args)