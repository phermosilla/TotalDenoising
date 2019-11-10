'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file GenerateRueMadameDataSet.py

    \brief Script to generate the rue madame dataset.

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import math
import argparse
import time
import os
from os import listdir
from os.path import isfile, join
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCCNN_DIR = os.path.join(BASE_DIR, 'MCCNN')
sys.path.append(os.path.join(MCCNN_DIR, 'utils'))

from ply_reader import read_points_binary_ply, save_model

def process_node(pts, nodeIter):
    if len(pts) > 500000:
        ptMax = np.amax(pts, axis = 0)
        ptMin = np.amin(pts, axis = 0)
        aabbSize = ptMax - ptMin
        midPt = (ptMax + ptMin)*0.5
        axis = np.where(aabbSize == np.amax(aabbSize))
        axis = axis[0][0]
        ptsLeft = np.array([pt for pt in pts if pt[axis] < midPt[axis]])
        ptsRight = np.array([pt for pt in pts if pt[axis] >= midPt[axis]])
        print("Processing intermediate node")
        print(len(pts))
        print(len(ptsLeft))
        print(len(ptsRight))
        print()
        maxPtLeft = ptMax
        maxPtLeft[axis] = midPt[axis]
        minPtRight = ptMin
        minPtRight[axis] = midPt[axis]
        nodeIter = process_node(ptsLeft, nodeIter)
        return process_node(ptsRight, nodeIter)
    else:
        save_model(args.srcFolder+"/RueMadame_"+str(nodeIter), pts)
        print("Saving leaf node")
        print(nodeIter)
        print()
        nodeIter += 1
        return nodeIter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate the rue madame dataset.')
    parser.add_argument('--srcFolder', default='RueMadame', help='Source folder with the models (default: RueMadame)')
    args = parser.parse_args()

    plyFiles = [join(args.srcFolder+"/", f) for f in listdir(args.srcFolder+"/") if isfile(join(args.srcFolder+"/", f)) and f.endswith('.ply')]
    
    nodeIter = 0
    for currPly in plyFiles:
        print(currPly)
        currPts = read_points_binary_ply(currPly)
        print(len(currPts))
        
        currPts = np.array([[cPt[0], cPt[1], cPt[2]] for cPt in currPts])

        nodeIter = process_node(currPts, nodeIter)