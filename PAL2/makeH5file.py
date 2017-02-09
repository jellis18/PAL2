#!/usr/bin/env python

from __future__ import division
from PAL2 import PALdatafile
import os, glob

import argparse

parser = argparse.ArgumentParser(description = 'Create HDF5 file for use in PAL2')

# options
parser.add_argument('--pardir', dest='pardir', action='store', type=str, required=True,
                   help='Full path to par files')
parser.add_argument('--timdir', dest='timdir', action='store', type=str, required=True,
                   help='Full path to tim files')
parser.add_argument('--h5File', dest='h5file', action='store', type=str, required=True,
                   help='output hdf5 file')
parser.add_argument('--iter', dest='iter', action='store', type=int, default=0,
                   help='Number of iterations in fit [default=0]')
parser.add_argument('--maxobs', dest='maxobs', action='store', type=int, default=30000,
                   help='Maximum number of TOAs [default=30000]')
parser.add_argument('--ephem', dest='ephem', action='store', type=str, default=None,
                   help='Ephemeris [default=None, goes with value in par file]')


# parse arguments
args = parser.parse_args()

try:
    os.remove(args.h5file)
except OSError:
    pass


parFiles = glob.glob(args.pardir + '/*.par')
timFiles = glob.glob(args.timdir + '/*.tim')

# sort
parFiles.sort()
timFiles.sort()


df = PALdatafile.DataFile(args.h5file)
for t,p in zip(timFiles, parFiles):
    print '\n', t, p, '\n'
    df.addTempoPulsar(p, t, iterations=args.iter, maxobs=args.maxobs, 
                      ephem=args.ephem)

