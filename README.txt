*** Installation of .cpp experiments ***

Prerequisites (folder lib)
- install the library libconfig-1.5.tar (./configure && make && make install)
- install galib247.zip (./configure && make && make install)
- install epsmerge-2.2.0.zip (sudo ./install), only needed for plotting results

NOTE: the file GAGenome.h in the galib247 is patched to incorporate additional 
information in the GA genome structure.

To compile and run the refComm2 experiments:
- import the project into Eclipse
- build and run from Release or Debug

NOTE: both config1.cfg and ga.cfg must be copied into the run folder.


*** Generation of graphs ***

The packages necessary for generating the graphs can be found in the requirements.txt file.

Each of the figures can be generated as follows:
- Fig 2: python generateGraphs.py g25p1kt5 noCom -f
- Fig 3: python analyseBehaviours.py -d g25p1kt5 & python analyseBehaviours.py -d noCom
- Fig 5: python generateGraphs.py g25p1kt5 -s
- Fig 6: python FitnessOnsetCorrelation.py
- Fig 7: python generateGraphs.py g25p1kt5 g25p1kt5_cV -f -m 5000


