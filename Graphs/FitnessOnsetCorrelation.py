#!/usr/bin/env python

import os
import re
import math
import argparse

from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns
from pylab import Polygon
from scipy import stats
import brewer2mpl
import numpy as np
import pandas as pd

# SEABORN
sns.set()
sns.set_style('white')
sns.set_context('paper')

# BREWER
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
palette = bmap.mpl_colors

# MATPLOTLIB PARAMS
rcParams['font.size'] = 24
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 24
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 24
rcParams['mathtext.default'] = 'regular'

# VALUES
if os.path.isfile('onsetPerf.csv'):
	df = pd.read_csv('onsetPerf.csv')

	dictGenValues = dict()
	for column in df:
		dictGenValues[int(df[column].name)] = [df[column][0], df[column][1]]

# GRAPHS GLOBAL VARIABLES
linewidth = 2
linestyles = ['-', '-', '-', '-']
markers = [None, None, None, None] #['o', '+', '*']

MARGIN_BARS_DIFF = 0.2
WIDTH_BARS_DIFF = (1.-2.*MARGIN_BARS_DIFF)/float(2 + 1)
MARGIN_BARS_RATIO = 0.3
WIDTH_BARS_RATIO = (1.-2.*MARGIN_BARS_RATIO)/float(2)

DPI = 96
FIGSIZE = (1280/DPI, 1024/DPI)
FLOAT_REG = "[-+]?\d*\.\d+|\d+"
LOW_GEN = 10
HIGH_GEN = 1000
OUTPUT_DIR = "OutputGraphsCorrAnalysis"

regLineReplicate = re.compile(r"^Replicate (\d+) : (.+)$")

def main() :
	# ----- GRADIENTS BARS FIXE VALUES -----
	print('Drawing gradients bars with fix values...')

	outputData = OUTPUT_DIR

	if not os.path.isdir(outputData) :
		os.makedirs(outputData)

	listGradientsOnset = [dictGenValues[gen][0] for gen in sorted(dictGenValues.keys())]
	listGradientsFitness = [dictGenValues[gen][1] for gen in sorted(dictGenValues.keys())]

	figsize = (4096/DPI, 1092/DPI)

	rcParams['font.size'] = 34
	rcParams['font.weight'] = 'bold'
	rcParams['axes.labelsize'] = 34
	rcParams['axes.labelweight'] = 'bold'
	rcParams['xtick.labelsize'] = 34
	rcParams['ytick.labelsize'] = 34
	rcParams['legend.fontsize'] = 34
	rcParams['mathtext.default'] = 'regular'

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, dpi = DPI)
	plt.grid()

	listBars = []
	listBars.append(ax.bar([x + MARGIN_BARS_DIFF for x in range(len(listGradientsOnset))], listGradientsOnset, width = WIDTH_BARS_DIFF, color = palette[0]))
	listBars.append(ax.bar([x + MARGIN_BARS_DIFF + WIDTH_BARS_DIFF for x in range(len(listGradientsFitness))], listGradientsFitness, width = WIDTH_BARS_DIFF, color = palette[1]))

	xticks = [x + 0.25 for x in range(len(listGradientsFitness))]
	xticksLabels = [x + 1 for x in range(len(listGradientsFitness))]
	ax.set_xticks(xticks)
	ax.set_xticklabels(xticksLabels)
	ax.set_xlabel("Population")
	ax.set_xlim(0, 40)

	ax.set_ylabel("Number of generations")
	ax.set_ylim(0, 30)

	legend = plt.legend(['Onset variation', 'Performance'], loc = 1, frameon=True)
	# # legend = plt.legend(['Sender', 'Receiver'], loc = 4, frameon=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.9')
	frame.set_edgecolor('0.9')

	# plt.tight_layout()
	plt.savefig(outputData + "/barsGradientFixed.png", bbox_inches = 'tight', dpi = DPI)
	plt.savefig(outputData + "/barsGradientFixed.svg", bbox_inches = 'tight', dpi = DPI)
	plt.close()


	# ----- STATS -----
	listDiffGen = [abs(fitness - onset) for fitness, onset in zip(listGradientsFitness, listGradientsOnset)]
	mean = np.mean(listDiffGen)
	std = np.std(listDiffGen)

	print('Mean generation difference between fitness and onset : ' + str(mean) + ' +/- ' + str(std))

	corr = stats.pearsonr(listGradientsFitness, listGradientsOnset)

	print('Correlation : ' + str(corr))




if __name__ == '__main__':
	main()