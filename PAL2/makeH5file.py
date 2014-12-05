#!/usr/bin/env python

from __future__ import division
import PALdatafile
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


# parse arguments
args = parser.parse_args()

try:
    os.remove(args.h5file)
except OSError:
    pass


parFiles = glob.glob(args.pardir + '/*.par')
timFiles = glob.glob(args.timdir + '/*.tim')
print parFiles, timFiles


df = PALdatafile.DataFile(args.h5file)
for t,p in zip(timFiles, parFiles):
    print t, p
    df.addTempoPulsar(p, t, iterations=3, sigma=10000)

