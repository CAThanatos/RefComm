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