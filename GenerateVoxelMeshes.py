'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file GenerateVocelMeshes.py

    \brief Script to generate the voxelization of the meshes.

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import argparse
import os
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCCNN_DIR = os.path.join(BASE_DIR, 'MCCNN')
sys.path.append(os.path.join(MCCNN_DIR, 'utils'))
from PyUtils import visualize_progress

def load_off_model(modelPath, normalizeDiagonal=True):
    numPts = 0
    numFaces = 0
    ptArray = []
    facesArray = []
    with open(modelPath, 'r') as modelFile:        
        for line in modelFile:
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            if line != "OFF" and not(line.startswith("#")):
                currLineInfo = line.split(' ')
                if numPts == 0  and numFaces == 0:
                    numPts = int(currLineInfo[0])
                    numFaces = int(currLineInfo[1])
                else:
                    if numPts > 0:
                        ptArray.append([float(currLineInfo[0]), float(currLineInfo[1]), float(currLineInfo[2])])
                        numPts -= 1
                    else:
                        facesArray.append([int(currLineInfo[1]), int(currLineInfo[2]), int(currLineInfo[3])])

    ptArray = np.array(ptArray)
    facesArray = np.array(facesArray)

    if normalizeDiagonal:
        indexSet = np.array(list(set(facesArray.flatten())))
        auxPt = ptArray[indexSet]
        coordMax = np.amax(auxPt, axis=0)
        coordMin = np.amin(auxPt, axis=0)
        aabbCenter = (coordMax + coordMin)*0.5
        diagonal = np.linalg.norm(coordMax - coordMin)
        ptArray = (ptArray - aabbCenter)/diagonal

    return ptArray, facesArray


def triangle_aabb_intersection(triangle, aabbCenter, aabbHalfSize):

    centeredTrian = triangle - aabbCenter
    trianEdges = np.array([triangle[1] - triangle[0],triangle[2] - triangle[1],triangle[0] - triangle[2]])
    absEdges = np.absolute(trianEdges)

    #Edge 0
    #X01
    p0 = trianEdges[0,2]*centeredTrian[0,1] - trianEdges[0,1]*centeredTrian[0,2]
    p2 = trianEdges[0,2]*centeredTrian[2,1] - trianEdges[0,1]*centeredTrian[2,2]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[0,2] + absEdges[0,1])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False
    
    #Y02
    p0 = -trianEdges[0,2]*centeredTrian[0,0] + trianEdges[0,0]*centeredTrian[0,2]
    p2 = -trianEdges[0,2]*centeredTrian[2,0] + trianEdges[0,0]*centeredTrian[2,2]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[0,2] + absEdges[0,0])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False

    #Z12
    p0 = trianEdges[0,1]*centeredTrian[1,0] - trianEdges[0,0]*centeredTrian[1,1]
    p2 = trianEdges[0,1]*centeredTrian[2,0] - trianEdges[0,0]*centeredTrian[2,1]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[0,1] + absEdges[0,0])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False

    #Edge 1
    #X01
    p0 = trianEdges[1,2]*centeredTrian[0,1] - trianEdges[1,1]*centeredTrian[0,2]
    p2 = trianEdges[1,2]*centeredTrian[2,1] - trianEdges[1,1]*centeredTrian[2,2]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[1,2] + absEdges[1,1])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False
    
    #Y02
    p0 = -trianEdges[1,2]*centeredTrian[0,0] + trianEdges[1,0]*centeredTrian[0,2]
    p2 = -trianEdges[1,2]*centeredTrian[2,0] + trianEdges[1,0]*centeredTrian[2,2]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[1,2] + absEdges[1,0])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False
        
    #Z0
    p0 = trianEdges[1,1]*centeredTrian[0,0] - trianEdges[1,0]*centeredTrian[0,1]
    p2 = trianEdges[1,1]*centeredTrian[1,0] - trianEdges[1,0]*centeredTrian[1,1]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[1,1] + absEdges[1,0])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False


    #Edge 2  
    #X2
    p0 = trianEdges[2,2]*centeredTrian[0,1] - trianEdges[2,1]*centeredTrian[0,2]
    p2 = trianEdges[2,2]*centeredTrian[1,1] - trianEdges[2,1]*centeredTrian[1,2]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[2,2] + absEdges[2,1])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False
    
    #Y1
    p0 = -trianEdges[2,2]*centeredTrian[0,0] + trianEdges[2,0]*centeredTrian[0,2]
    p2 = -trianEdges[2,2]*centeredTrian[1,0] + trianEdges[2,0]*centeredTrian[1,2]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[2,2] + absEdges[2,0])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False
    
    #Z12
    p0 = trianEdges[2,1]*centeredTrian[1,0] - trianEdges[2,0]*centeredTrian[1,1]
    p2 = trianEdges[2,1]*centeredTrian[2,0] - trianEdges[2,0]*centeredTrian[2,1]
    minVal = min(p0, p2)
    maxVal = max(p0, p2)
    rad = (absEdges[2,1] + absEdges[2,0])*aabbHalfSize
    if (minVal > rad) or (maxVal<-rad):
        return False

    #Directions
    minVal = np.amin(centeredTrian, axis=0)
    maxVal = np.amax(centeredTrian, axis=0)
    halSizeArray = np.array([aabbHalfSize,aabbHalfSize,aabbHalfSize])
    if np.any(np.logical_or(np.greater(minVal, halSizeArray), np.less(maxVal, halSizeArray*-1.0))):
        return False

    #Overlap
    normal = np.cross(trianEdges[0], trianEdges[1])
    vmin = -1.0*halSizeArray - centeredTrian[0]
    vmax = halSizeArray - centeredTrian[0]
    realMin = np.array([vmin[auxIter] if normal[auxIter] > 0.0 else vmax[auxIter] for auxIter in range(3)])
    realMax = np.array([vmax[auxIter] if normal[auxIter] > 0.0 else vmin[auxIter] for auxIter in range(3)])
    if np.dot(normal, realMin) > 0.0:
        return False
    if np.dot(normal, realMax) >= 0.0:
        return True
        
    return False
    
  
def compute_mesh_voxelization(vertices, faces, cellSize = 0.05):
    
    indexSet = np.array(list(set(faces.flatten())))
    auxPt = vertices[indexSet]
    coordMax = np.amax(auxPt, axis=0)
    coordMin = np.amin(auxPt, axis=0)
    vertices = vertices - coordMin
    diffCoords = coordMax - coordMin
    diagonal = np.linalg.norm(diffCoords)
    currCellSize = diagonal*cellSize
    
    numCells = np.ceil(diffCoords/currCellSize)
    intNumCells = numCells.astype(int)
    cellIndexsDic = {}
    
    for faceIter, currFace in enumerate(faces):
        trianVertices = np.array([vertices[currFace[0]], vertices[currFace[1]], vertices[currFace[2]]])
        trianMax = np.amax(trianVertices, axis=0)
        trianMin = np.amin(trianVertices, axis=0)
        maxCellIndex = np.floor(trianMax/currCellSize).astype(int)
        minCellIndex = np.floor(trianMin/currCellSize).astype(int)
        for xCoord in range(minCellIndex[0], maxCellIndex[0]+1):
            for yCoord in range(minCellIndex[1], maxCellIndex[1]+1):
                for zCoord in range(minCellIndex[2], maxCellIndex[2]+1):
                    voxelCenter = np.array([float(xCoord)*currCellSize + currCellSize*0.5,
                                            float(yCoord)*currCellSize + currCellSize*0.5,
                                            float(zCoord)*currCellSize + currCellSize*0.5])
                    if triangle_aabb_intersection(trianVertices, voxelCenter, currCellSize*0.5):
                        cellIndex = xCoord*intNumCells[1]*intNumCells[2] + yCoord*intNumCells[2] + zCoord
                        if cellIndex in cellIndexsDic:
                            cellIndexsDic[cellIndex].append(faceIter)
                        else:
                            cellIndexsDic[cellIndex] = [faceIter]
                            
    faceIndexes = []
    startVoxelIndexs = np.array([[[[-1,-1] for zIter in range(intNumCells[2])] \
        for yIter in range(intNumCells[1])] \
        for xIter in range(intNumCells[0])])
    for xCoord in range(intNumCells[0]):
        for yCoord in range(intNumCells[1]):
            for zCoord in range(intNumCells[2]):
                cellIndex = xCoord*intNumCells[1]*intNumCells[2] + yCoord*intNumCells[2] + zCoord
                if cellIndex in cellIndexsDic:
                    startVoxelIndexs[xCoord, yCoord, zCoord, 0] = len(faceIndexes)
                    for currIndex in cellIndexsDic[cellIndex]:
                        faceIndexes.append(currIndex)
                    startVoxelIndexs[xCoord, yCoord, zCoord, 1] = len(faceIndexes)
                    
    return faceIndexes, startVoxelIndexs, intNumCells, currCellSize
    
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate the voxelization of meshes')
    parser.add_argument('--srcFolder', default='NoisyDataSets/TestMeshes', help='Source folder with the meshes (default: NoisyDataSets/TestMeshes)')
    args = parser.parse_args()

    meshList = [f for f in listdir(args.srcFolder+"/") if isfile(join(args.srcFolder+"/", f)) and f.endswith(".off")]

    for meshIter, currMesh in enumerate(meshList):
        objName = currMesh.split('.')[0]
        vertexs, faces = load_off_model(join(args.srcFolder+"/", currMesh))
        faceIndexes, startVoxelIndexs, intNumCells, currCellSize = compute_mesh_voxelization(vertexs, faces, 0.05)
        pickle.dump((vertexs, faces, faceIndexes, startVoxelIndexs, intNumCells, currCellSize), open(join(args.srcFolder+"/", objName+".vox"),"wb"))
        visualize_progress(meshIter, len(meshList), objName+" "+str(intNumCells))
        