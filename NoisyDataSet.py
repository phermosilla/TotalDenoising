'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file NoisyDataSet.py

    \brief NoisyDataSet class.

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
from os import listdir
from os.path import isfile, join
import math
import numpy as np

class NoisyDataSet:
    """NoisyDataSet.
    """
    
    def __init__(self, dataset, train, seed=None, fileList=None):
        """Constructor.

        Args:
            
        """

        self.train_ = train
        self.datasetId_ = dataset

        # Dataset folder.
        if  dataset == 3:
            self.dataset_ = "RueMadame/"
            self.instanceList_ = [f[:-4] for f in listdir(self.dataset_) if isfile(join(self.dataset_, f)) and f.endswith(".txt")]
            self.modelList_ = self.instanceList_
        else:
            if dataset == 0:
                self.dataset_ = "NoisyDataSets/NoisyColoredGaussianDataSet/"
            elif dataset == 1:
                self.dataset_ = "NoisyDataSets/NoisyColoredGaussianDataSet/"
            elif dataset == 2:
                self.dataset_ = "NoisyDataSets/BlensorColored/"

            # Clean/Noisy dataset.
            if train:
                DSFilePath = "NoisyDataSets/noisy_dataset.txt"
            else:
                DSFilePath = "NoisyDataSets/test_dataset.txt"
            if not(fileList is None):
                DSFilePath = fileList
            
            # Model list.
            modelTypes = {}
            self.modelList_ = []
            self.instanceList_ = []
            with open(DSFilePath, 'r') as DSFile:        
                for line in DSFile:
                    modelName = line.rstrip()
                    self.instanceList_.append((len(self.modelList_), "005"))
                    self.instanceList_.append((len(self.modelList_), "01"))
                    self.instanceList_.append((len(self.modelList_), "015"))
                    self.modelList_.append(modelName)
                    
        # Iterator. 
        self.randomState_ = np.random.RandomState(seed)
        self.iterator_ = 0
        self.randList_ = self.randomState_.permutation(self.instanceList_)

        # Cache.
        self.cache_ = {}


    def begin_epoch(self):
        self.iterator_ = 0
        self.randList_ = self.randomState_.permutation(self.instanceList_)

    def next(self):
        self.iterator_+=1
        
    def get_num_models(self):
        return len(self.modelList_)
        
    def get_num_instances(self):
        return len(self.instanceList_)
        
    def end_epoch(self):
        return self.iterator_ >= len(self.randList_)
    
    def get_current_model(self, clean=False):
        if self.iterator_ < len(self.randList_):
            if self.datasetId_ == 3:
                currInstance = "0"
                currModel = self.randList_[self.iterator_]
                if currModel in self.cache_:
                    points = self.cache_[currModel]
                else:
                    points = np.loadtxt(self.dataset_+currModel+".txt", delimiter=',')
                    coordMax = np.amax(points, axis=0)
                    coordMin = np.amin(points, axis=0)
                    center = (coordMax+coordMin)*0.5
                    points = (points - center)/5.0
                    self.cache_[currModel] = points
            else:
                currModel = self.modelList_[int(self.randList_[self.iterator_][0])]
                if not clean:
                    currInstance = self.randList_[self.iterator_][1]
                    currModelPath = currModel+"_"+currInstance
                    if currModelPath in self.cache_:
                        points = self.cache_[currModelPath]
                    else:
                        points = np.loadtxt(self.dataset_+currModelPath+".txt", delimiter=',')
                        self.cache_[currModelPath] = points
                else:
                    currInstance = ""
                    if currModel in self.cache_:
                        points = self.cache_[currModel]
                    else:
                        points = np.loadtxt(self.dataset_+currModel+".txt", delimiter=',')
                        self.cache_[currModel] = points
            return points, currModel, currInstance
        return -1, "", ""
