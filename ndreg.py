#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import SimpleITK as sitk
import ndio.remote.OCP as ocp
import sys, os, math, glob, subprocess, shutil
import numpy as np
from numpy import mat, array, dot
from time import time
from itertools import product
from landmarks import *
scriptDirPath = os.path.dirname(os.path.realpath(__file__))+"/"

dimension = 3
vectorComponentType = sitk.sitkFloat32
vectorType = sitk.sitkVectorFloat32
identityAffine = [1,0,0,0,1,0,0,0,1,0,0,0]
identityDirection = [1,0,0,0,1,0,0,0,1]
zeroOrigin = [0]*dimension
zeroIndex = [0]*dimension
ndreg=True
def run(command, checkReturnValue=True, quiet=True):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    outText = ""
    for line in iter(process.stdout.readline, ''):
        if not quiet: sys.stdout.write(line)
        outText += line
    
    process.communicate()[0]
    returnValue = process.returncode
    if checkReturnValue and (returnValue != 0): raise Exception(outText)
    return (returnValue, outText)

"""
def runMatlab(command, dirPath="", checkReturnValue=True, quiet=True):
    origDirPath = os.getcwd()
    if dirPath != "": os.chdir(dirPath)
    matlabCommand = 'matlab -nodesktop -nosplash -r "try; {0}; catch, err = lasterror; disp(err.message); exit(1), end; exit(0);"'.format(command)
    (returnValue, outText) = run(matlabCommand, checkReturnValue, quiet)
    os.chdir(origDirPath)
    return (returnValue, outText)
"""

def getOutPaths(inPath, outPath):
    if (os.path.splitext(outPath)[1] == "") or os.path.isdir(outPath):
        # outPath is a directory
        outDirPath = os.path.normpath(outPath)+"/"
        if inPath == "":
            outFileName = ""
        else:
            outFileName = os.path.basename(inPath)

    else:
        # outPath is a file
        outDirPath = os.path.dirname(os.path.normpath(outPath))+"/"
        outFileName = os.path.basename(outPath)
    if not os.path.exists(outDirPath): os.makedirs(outDirPath)
    return (outDirPath+outFileName, outDirPath)

def txtWrite(text, path, mode="w"):
    textFile = open(path, mode)
    print(text, file=textFile)
    textFile.close()

def txtRead(path):
    textFile = open(path,"r")
    text = textFile.read()
    textFile.close()
    return text

def txtReadParameterList(path):
    return map(float,txtRead(path).split())

def txtWriteParameterList(parameterList, path):
    txtWrite(" ".join(map(str,parameterList)), path)

def imgCopy(inPath, outPath):
    """
    Copies Analyze image and header at inPath to outPath.
    """
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    for ext in [".img",".hdr"]: shutil.copy(os.path.splitext(inPath)[0]+ext, os.path.splitext(outPath)[0]+ext)
    return outPath

def imgUpload(path, token, channel="", resolution=0, server="openconnecto.me"):
    """
    Upload image with given token from given server at given resolution.
    If channel isn't specified image is uploaded to default channel
    
    TODO: Check bounds
    """
    # Read image
    img = imgRead(path)
    array = sitk.GetArrayFromImage(img).transpose([1,2,0])

    # Create OCP instance
    oo = ocp()
    oo.hostname = server

    # Get size and spacing of image data
    metadata = oo.get_proj_info(token)
    #size = metadata[u'dataset'][u'imagesize'][unicode(str(resolution))]
    size = array.shape
    
    # Download all image data from specified channel
    if(channel == ""): channel = metadata[u'channels'].keys()[0] # If channel isn't specified use first one.
    oo.post_cutout(token, channel, 0, size[0], 0, size[1], 0, size[2], data=array, resolution=resolution)

def imgDownload(token, path, channel="", resolution=0, server="openconnecto.me"):
    """
    Download image with given token from given server at given resolution.
    If channel isn't specified the default channel is downloaded.
    """
    
    # Create OCP instance
    oo = ocp()
    oo.hostname = server

    # Get size and spacing of image data
    metadata = oo.get_proj_info(token)
    size = metadata[u'dataset'][u'imagesize'][unicode(str(resolution))]
    spacingNm = metadata[u'dataset'][u'voxelres'][unicode(str(resolution))] # Returns spacing in nanometers
    spacing = [x * 1e-6 for x in spacingNm] # Convert spacing to mm

    # Download all image data from specified channel
    channelList = metadata[u'channels'].keys()
    
    if len(channelList) == 0:
        raise Exception("No channels defined for given token.")
    elif channel == "": 
        channel = channelList[0] # If channel isn't specified use first one.
    elif not(channel in channelList):
        raise Exception("Channel '{0}' does not exist for given token.".format(channel))

    array = oo.get_cutout(token, channel,0,size[0],0,size[1],0,size[2],resolution)
    img = sitk.GetImageFromArray(array) # convert numpy array to sitk image
    img.SetSpacing(spacing)

    # Write image to file
    path = getOutPaths("",path)[0]
    imgWrite(img, path)

    return path

def imgRead(path):
    """
    Alias for sitk.ReadImage
    """
    return sitk.ReadImage(path)


def imgWrite(img, path):
    """
    Write sitk image to path.
    """
    path = getOutPaths("",path)[0]
    sitk.WriteImage(img, path)
    #sitk.ImageFileWriter().Execute(img, path, False)

    # Reformat files to be compatible with CIS Software
    ext = os.path.splitext(path)[1].lower()
    if ext in [".img",".hdr"]:
        ###imgReformat(path, path)
        pass
    elif ext == ".vtk":
        mapReformat(path, path)
        
    return path

def mapReformat(inPath, outPath):
    """
    Reformats map so that it can be read by CIS software.
    """
    # Get size of map
    inFile = open(inPath,"rb")
    lineList = inFile.readlines()
    for line in lineList:
        if line.lower().strip().startswith("dimensions"):
            size = map(int,line.split(" ")[1:dimension+1])
            break
    inFile.close()

    outFile = open(outPath,"wb")
    for (i,line) in enumerate(lineList):
        if i == 1:
            newline = line.lstrip(line.rstrip("\n"))
            line = "lddmm 8 0 0 {0} {0} 0 0 {1} {1} 0 0 {2} {2}".format(size[2]-1, size[1]-1, size[0]-1) + newline
        outFile.write(line)

def imgReformat(inPath, outPath):
    """
    Reformates Analyze Image so that it can be read by CIS software
    """
    (outPath, outDirPath) = getOutPaths(inPath, outPath)

    if inPath != outPath: imgCopy(inPath, outPath)
    rootPath = os.path.splitext(outPath)[0]
    imgPath = rootPath + ".img"
    hdrPath = rootPath + ".hdr"

    hdrFile = open(hdrPath,"rb")
    hdrData = hdrFile.read()
    hdrData = hdrData[0:348]
    hdrFile.close()

    hdrFile = open(hdrPath,"wb")
    hdrFile.write(hdrData)
    hdrFile.close()

    return outPath

def imgThreshold(inPath, outPath, threshold=0):
    """
    Thresholds image at inPath at given threshold and writes result to outPath.
    """
    inImg = imgRead(inPath)
    outImg = sitk.BinaryThreshold(inImg, 0, threshold, 0, 1)

    # Write output image
    (outPath, outDirPath) = getOutPaths(inPath, outPath)    
    imgWrite(outImg, outPath)

    return outPath

def imgMask(inPath, maskPath, outPath):
    """
    Masks image at inPath with mask at maskPath and writes result to outPath.
    """
    inImg = imgRead(inPath)
    maskImg = imgRead(maskPath)
    outImg = sitk.MaskImageFilter().Execute(inImg, maskImg)

    # Write output image
    (outPath, outDirPath) = getOutPaths(inPath, outPath)    
    imgWrite(outImg, outPath)

    return  outPath

def imgGenerateMask(inPath, outPath, threshold=None, forgroundValue=1):
    """
    Generates morphologically smooth mask with given forground value from input image.
    If the threshold is given, the binary mask is initialzed using the given threshold...
    ...Otherwise it is initialized using Otsu's Method.
    """
    inImg = imgRead(inPath)
    
    if threshold is None:
        # Initialize binary mask using otsu threshold
        inMask = sitk.BinaryThreshold(inImg, 0, 0, 0, forgroundValue) # Mask of non-zero voxels
        otsuThresholder = sitk.OtsuThresholdImageFilter()
        otsuThresholder.SetInsideValue(0)
        otsuThresholder.SetOutsideValue(forgroundValue)
        otsuThresholder.SetMaskValue(forgroundValue)
        tmpMask = otsuThresholder.Execute(inImg, inMask)
    else:
        # initialzie binary mask using given threshold
        tmpMask = sitk.BinaryThreshold(inImg, 0, threshold, 0, forgroundValue)

    # Assuming input image is has isotropic resolution...
    # ... compute size of morphological kernels in voxels.
    spacing = min(list(inImg.GetSpacing()))
    openingRadiusMM = 0.05 # In mm
    closingRadiusMM = 0.125 # In mm
    openingRadius = max(1, int(round(openingRadiusMM / spacing))) # In voxels
    closingRadius = max(1, int(round(closingRadiusMM / spacing))) # In voxels

    # Morphological open mask remove small background objects
    opener = sitk.GrayscaleMorphologicalOpeningImageFilter()
    opener.SetKernelType(sitk.sitkBall)
    opener.SetKernelRadius(openingRadius)
    tmpMask = opener.Execute(tmpMask)

    # Morphologically close mask to fill in any holes
    closer = sitk.GrayscaleMorphologicalClosingImageFilter()
    closer.SetKernelType(sitk.sitkBall)
    closer.SetKernelRadius(closingRadius)
    outMask = closer.Execute(tmpMask)

    # Write output mask
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(outMask, outPath)

    return outPath

def imgCheckerBoard(inPath, refPath, outPath, useHistogramMatching=True):
    """
    Checkerboards input image with reference image
    """
    # Read input and reference images
    inImg = imgRead(inPath)
    refImg = imgRead(refPath)
    
    inSize = list(inImg.GetSize())
    refSize = list(refImg.GetSize())
    
    if(inSize != refSize):
        sourceSize = np.array([inSize, refSize]).min(0)
        tmpImg = sitk.Image(refSize,refImg.GetPixelID()) # Empty image with same size as reference image
        tmpImg.CopyInformation(refImg)
        inImg = sitk.PasteImageFilter().Execute(tmpImg, inImg, sourceSize, zeroIndex, zeroIndex)

    if useHistogramMatching:
        numBins = 64
        numMatchPoints = 8
        inImg = sitk.HistogramMatchingImageFilter().Execute(inImg, refImg, numBins, numMatchPoints, False)

    outImg = sitk.CheckerBoardImageFilter().Execute(inImg, refImg,[8]*dimension)
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(outImg, outPath)

    return outPath

def imgHM(inPath, refPath, outPath, numMatchPoints=8, numBins=64):
    """
    Histogram matches input image to reference image and writes result to output image
    """
    inImg = imgRead(inPath)
    refImg = imgRead(refPath)

    outImg = sitk.HistogramMatchingImageFilter().Execute(inImg, refImg, numBins, numMatchPoints, False)
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(outImg, outPath)

    return outPath


def imgResample(inPath, outPath, spacing, size=[], useNearestNeighborInterpolation=False):
    """
    Resamples image at inPath to given spacing and writes result to outPath.
    """
    if len(spacing) != dimension: raise Exception("len(spacing) != " + str(dimension))

    # Read input image
    inImg = imgRead(inPath)

    # Set Size
    if size == []:
        inSpacing = inImg.GetSpacing()
        inSize = inImg.GetSize()
        size = [int(math.ceil(inSize[i]*(inSpacing[i]/spacing[i]))) for i in range(dimension)]
    else:
        if len(size) != dimension: raise Exception("len(size) != " + str(dimension))
    
    # Resample input image
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearestNeighborInterpolation]
    identityTransform = sitk.Transform()
    outImg = sitk.Resample(inImg, size, identityTransform, interpolator, zeroOrigin, spacing)

    # Write output image
    (outPath, outDirPath) = getOutPaths(inPath, outPath)    
    imgWrite(outImg, outPath)
    
    return outPath

def sizeOut(inImg, transform, outSpacing):
    outCornerPointList = []
    inSize = inImg.GetSize()
    for corner in product((0,1), repeat=dimension):
        inCornerIndex = array(corner)*array(inSize)
        inCornerPoint = inImg.TransformIndexToPhysicalPoint(inCornerIndex)
        outCornerPoint = transform.GetInverse().TransformPoint(inCornerPoint)
        outCornerPointList += [list(outCornerPoint)]

    size = np.ceil(array(outCornerPointList).max(0) / outSpacing).astype(int)
    return size
    

def imgApplyAffine(inPath, outPath, affine, useNearestNeighborInterpolation=False, size=[], spacing=[]):
    # Read input image
    inImg = sitk.ReadImage(inPath)

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearestNeighborInterpolation]

    # Set affine parameters
    affineTransform = sitk.AffineTransform(dimension)
    numParameters = len(affineTransform.GetParameters())
    if (len(affine) != numParameters): raise Exception("affine must have length {0}.".format(numParameters))
    affineTransform = sitk.AffineTransform(dimension)
    affineTransform.SetParameters(affine)

    # Set Spacing
    if spacing == []:
        spacing = inImg.GetSpacing()
    else:
        if len(spacing) != dimension: raise Exception("spacing must have length {0}.".format(dimension))

    # Set size
    if size == []:
        # Compute size to contain entire output image
        """
        outCornerPointList = []
        inSize = inImg.GetSize()
        for corner in product((0,1), repeat=dimension):
            inCornerIndex = array(corner)*array(inSize)
            inCornerPoint = inImg.TransformIndexToPhysicalPoint(inCornerIndex)
            outCornerPoint = affineTransform.GetInverse().TransformPoint(inCornerPoint)
            outCornerPointList += [list(outCornerPoint)]

        size = np.ceil(array(outCornerPointList).max(0) / spacing).astype(int)
        """
        size = sizeOut(inImg, affineTransform, spacing)
    else:
       if len(size) != dimension: raise Exception("size must have length {0}.".format(dimension))
    
    # Apply affine transform
    outImg = sitk.Resample(inImg, size, affineTransform, interpolator, zeroOrigin, spacing)

    # Write output image
    (outPath, outDirPath) = getOutPaths(inPath, outPath)    
    imgWrite(outImg, outPath)

    return outPath

def fieldApplyField(inPath, fieldPath, outPath):
    inField = sitk.Cast(imgRead(inPath), sitk.sitkVectorFloat64)
    field = sitk.Cast(imgRead(fieldPath), sitk.sitkVectorFloat64)
    
    size = list(inField.GetSize())
    spacing = list(inField.GetSpacing())

    # Create transform for input field
    inTransform = sitk.DisplacementFieldTransform(dimension)
    inTransform.SetDisplacementField(inField)
    inTransform.SetInterpolator(sitk.sitkLinear)

    # Create transform for field
    transform = sitk.DisplacementFieldTransform(dimension)
    transform.SetDisplacementField(field)
    transform.SetInterpolator(sitk.sitkLinear)
    
    # Combine thransforms
    outTransform = sitk.Transform()
    outTransform.AddTransform(transform)
    outTransform.AddTransform(inTransform)

    # Get output displacement field
    outField = sitk.TransformToDisplacementFieldFilter().Execute(outTransform, vectorType, size, zeroOrigin, spacing, identityDirection)        

    # Write output field
    (outPath, outDirPath) = getOutPaths(inPath, outPath)    
    imgWrite(outField, outPath)

    return outPath

def imgApplyField(inPath, fieldPath, outPath, useNearestNeighborInterpolation=False, size=[], spacing=[]):
    """
    img \circ field
    """
    inImg = imgRead(inPath)
    inImg.SetDirection(identityDirection)
    field = sitk.Cast(imgRead(fieldPath), sitk.sitkVectorFloat64)
    field.SetDirection(identityDirection)

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearestNeighborInterpolation]

    # Set transform field
    transform = sitk.DisplacementFieldTransform(dimension)
    transform.SetInterpolator(sitk.sitkLinear)
    transform.SetDisplacementField(field)


    # Set size
    if size == []:
        size = inImg.GetSize()
    else:
        if len(size) != dimension: raise Exception("size must have length {0}.".format(dimension))

    # Set Spacing
    if spacing == []:
        spacing = inImg.GetSpacing()
    else:
        if len(spacing) != dimension: raise Exception("spacing must have length {0}.".format(dimension))
    
    # Apply displacement transform
    outImg = sitk.Resample(inImg, size, transform, interpolator, zeroOrigin, spacing)

    # Write output image
    (outPath, outDirPath) = getOutPaths(inPath, outPath)    
    imgWrite(outImg, outPath)

    return outPath
    
def affineToField(affine, size, spacing, outPath):
    """
    Generates displacement field with given size and spacing based on affine parameters.
    """
    if len(size) != dimension: raise Exception("size must have length {0}.".format(dimension))
    if len(spacing) != dimension: raise Exception("spacing must have length {0}.".format(dimension))

    # Set affine parameters
    affineTransform = sitk.AffineTransform(dimension)
    numParameters = len(affineTransform.GetParameters())
    if len(affine) != numParameters: raise Exception("affine must have length {0}.".format(numParameters))
    affineTransform = sitk.AffineTransform(dimension)
    affineTransform.SetParameters(affine)

    # Convert affine transform to field
    outField = sitk.TransformToDisplacementFieldFilter().Execute(affineTransform, vectorType, size, zeroOrigin, spacing, identityDirection)

    # Write output displacement field
    (outPath, outDirPath) = getOutPaths("", outPath)    
    imgWrite(outField, outPath)

    return outPath

def fieldToMap(inPath, outPath):
    """
    Convert input displacement field into CIS compatable map
    """
    inField = imgRead(inPath)
    inSpacing = inField.GetSpacing()
    inSize = inField.GetSize()
    idMap = mapCreateIdentity(inSize)
    idMap.CopyInformation(inField)

    outMapComponentList = []
    for i in range(dimension):
        idMapComponent = sitk.VectorIndexSelectionCastImageFilter().Execute(idMap, i, vectorComponentType)
        inFieldComponent = sitk.VectorIndexSelectionCastImageFilter().Execute(inField, i, vectorComponentType)
        outMapComponent = idMapComponent + (inFieldComponent / inSpacing[i])
        outMapComponentList += [outMapComponent]

    outMap = sitk.ComposeImageFilter().Execute(outMapComponentList)

    # Write output map
    (outPath, outDirPath) = getOutPaths("", outPath)    
    imgWrite(outMap, outPath)

    return outPath

def mapToField(inPath, outPath, spacing=[]):
    #TODO
    return

def mapCreateIdentity(size):
    """
    Generates an identity map of given size
    """
    spacing = [1,1,1]
    return sitk.PhysicalPointImageSource().Execute(vectorType, size, zeroOrigin, spacing, identityDirection)

def lmkApplyAffine(inPath, outPath, affine, spacing=[1.0,1.0,1.0]):
    if (not(type(spacing)) is list) or (len(spacing) != 3): raise Exception("spacing must be a list of length 3.")
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    
    # Apply affine to landmarks
    lmk = landmarks(inPath, spacing)
    transformedLmk = lmk.Affine(affine)
    transformedLmk.Write(outPath)
    return outPath

def affineInverse(affine):
    # x0 = A0*x1 + b0
    # x1 = (A0.I)*x0 + (-A0.I*b0) = A1*x0 + b1
    A0 = mat(affine[0:9]).reshape(3,3)
    b0 = mat(affine[9:12]).reshape(3,1)

    A1 = A0.I
    b1 = -A1*b0
    return A1.flatten().tolist()[0] + b1.flatten().tolist()[0]

def affineApplyAffine(inAffine, affine):
    """ A_{outAffine} = A_{inAffine} \circ A_{affine} """
    if (not(type(inAffine)) is list) or (len(inAffine) != 12): raise Exception("inAffine must be a list of length 12.")
    if (not(type(affine)) is list) or (len(affine) != 12): raise Exception("affine must be a list of length 12.")
    A0 = array(affine[0:9]).reshape(3,3)
    b0 = array(affine[9:12]).reshape(3,1)
    A1 = array(inAffine[0:9]).reshape(3,3)
    b1 = array(inAffine[9:12]).reshape(3,1)

    # x0 = A0*x1 + b0
    # x1 = A1*x2 + b1
    # x0 = A0*(A1*x2 + b1) + b0 = (A0*A1)*x2 + (A0*b1 + b0)
    A = dot(A0,A1)
    b = dot(A0,b1) + b0

    outAffine = A.flatten().tolist() + b.flatten().tolist()
    return outAffine

def affineToMap(affine, size, outPath, spacing=[1,1,1]):
    # Copy affine matrix to direction matrix
    direction = itk.Matrix[itk.D,dimension,dimension]()
    for col in range(dimension):
        for row in range(dimension):            
            direction.GetVnlMatrix().put(col,row,affine[col*dimension+row])

    # Set origin based on offset
    origin = list(array(affine[9:12])/array(spacing))
    
    # Create map based on direction and origin
    ImageSourceType = itk.PhysicalPointImageSource[MapType]
    imageSource = ImageSourceType.New(Size=size,Origin=origin,Direction=direction)
    imageSource.Update()

    outMap = imageSource.GetOutput()
    outMap.SetOrigin(zeroOrigin)
    outMap.SetDirection(identityDirection)

    # Write map to file
    (outPath, outDirPath) = getOutPaths("", outPath)
    imgWrite(outMap, outPath)
    return outPath

def imgAffine(inPath, refPath, outPath, useNearestNeighborInterpolation=False, useRigid=False, verbose=False):
    """
    Perform Affine Registration between input image and reference image
    """
    # Read input and reference images
    inImg = imgRead(inPath)
    refImg = imgRead(refPath)
    (outPath, outDirPath) = getOutPaths(inPath, outPath)

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearestNeighborInterpolation]

    # Set transform
    transform = [sitk.AffineTransform(dimension), sitk.Similarity3DTransform()][useRigid]

    # Do registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(interpolator)
    registration.SetInitialTransform(transform)
    registration.SetOptimizerAsGradientDescent(learningRate=0.000001, numberOfIterations=1000)
    #registration_method.SetOptimizerScalesFromPhysicalShift()
    if(verbose): registration.AddCommand(sitk.sitkIterationEvent, lambda: print("Iteration = {0}, Value = {1}".format(registration.GetOptimizerIteration(),registration.GetMetricValue())))
    #imgWrite(sitk.SmoothingRecursiveGaussian(inImg,0.25),outDirPath+"in.img")
    #imgWrite(sitk.SmoothingRecursiveGaussian(refImg,0.25),outDirPath+"ref.img")

    registration.Execute(sitk.SmoothingRecursiveGaussian(refImg,0.25),
                         sitk.SmoothingRecursiveGaussian(inImg,0.25) )


    # Write final transform parameters to text file
    affine = list(transform.GetMatrix()) + list(transform.GetTranslation())
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    txtWriteParameterList(affine, outDirPath+"affine.txt")

    # Apply final transform parameters to input image
    imgApplyAffine(inPath, outPath, affine, useNearestNeighborInterpolation, refImg.GetSize())

    return affine

def imgMI(inPath, refPath, numBins=64):
    #inImg = imgRead(inPath)
    #refImg = imgRead(refPath)
    
    inImg = sitk.Normalize(imgRead(inPath))
    refImg = sitk.Normalize(imgRead(refPath))

    # Evaluate Metric
    # In SimpleITK the metric can't be accessed directly.
    # Therefore we create a do-nothing registration method which uses an identity transform to get the metric value
    identityTransform = sitk.Transform()
    tmpRegistration = sitk.ImageRegistrationMethod()

    tmpRegistration.SetMetricSamplingStrategy(tmpRegistration.RANDOM)
    tmpRegistration.SetMetricSamplingPercentage(0.01)
    #tmpRegistration.SetMetricAsJointHistogramMutualInformation(numBins)
    #tmpRegistration.SetMetricAsMeanSquares()
    tmpRegistration.SetMetricAsMattesMutualInformation(numBins)
    tmpRegistration.SetInterpolator(sitk.sitkNearestNeighbor)
    tmpRegistration.SetInitialTransform(identityTransform)
    tmpRegistration.SetOptimizerAsGradientDescent(learningRate=0.000001, numberOfIterations=1)
    tmpRegistration.Execute( sitk.Cast(refImg,sitk.sitkFloat32),sitk.Cast(inImg, sitk.sitkFloat32) )

    return tmpRegistration.GetMetricValue()




def imgMetamorphosis(inPath, refPath, outPath, alpha=0.01, beta=0.05, useNearestNeighborInterpolation=False, useBiasCorrection=False, verbose=True):
    """
    Performs Metamorphic LDDMM between input and refereence images
    """
    if(not useBiasCorrection): beta = -1.0

    (outPath, outDirPath) = getOutPaths(inPath, outPath)    
    fieldPath = outDirPath+"field.vtk"
    command = scriptDirPath+"metamorphosis/bin/metamorphosis --input {0} --reference {1} --output {2} --alpha {3} --beta {4} --displacement {5} --iterations 100 --sigma 1 --steps 4".format(inPath, refPath, outPath, alpha, beta, fieldPath)
    if(verbose): command += " --verbose"
    os.system(command)


    return fieldPath


def imgRegistration(inImgPath, refImgPath, outPath, useNearestNeighborInterpolation=False, initialAffine=identityAffine):
    (outPath, outDirPath) = getOutPaths(inImgPath, outPath)

    inImgFileName = os.path.basename(inImgPath)
    refImgFileName = os.path.basename(refImgPath)

    # Get original data
    print("--- Getting Original Data")
    origDirPath = outDirPath + "0_orig/"

    refImg = imgRead(refImgPath)
    refSize = refImg.GetSize()
    refSpacing = refImg.GetSpacing()
    refImg.SetDirection(identityDirection)
    imgWrite(refImg,origDirPath+refImgFileName)

    inImg = imgRead(inImgPath)
    inSize = inImg.GetSize()
    inSpacing = inImg.GetSpacing()
    inImg.SetDirection(identityDirection)
    imgWrite(inImg,origDirPath+inImgFileName)

    # Apply initial affine
    print("--- Applying initial affine")
    initialDirPath = outDirPath + "1_initial/"
    imgApplyAffine(origDirPath+inImgFileName, initialDirPath, initialAffine, useNearestNeighborInterpolation)
    imgCheckerBoard(initialDirPath+inImgFileName, origDirPath+refImgFileName, initialDirPath+"checkerboard.img")
    
    # Do rigid registration
    print("--- Rigid Registration")
    rigidDirPath = outDirPath + "2_rigid/"
    rigid = imgAffine(initialDirPath+inImgFileName, origDirPath+refImgFileName, rigidDirPath, useNearestNeighborInterpolation, useRigid=True, verbose=False)
    imgCheckerBoard(rigidDirPath+inImgFileName, origDirPath+refImgFileName, rigidDirPath+"checkerboard.img")

    # Combine initial and rigid transforms and write result to file
    combinedAffine = affineApplyAffine(rigid, initialAffine)
    txtWriteParameterList(combinedAffine, rigidDirPath+"combinedAffine.txt") # Write combined transform to file

    # Do affine registration
    print("--- Affine Registration")
    affineDirPath = outDirPath + "3_affine/"
    affine = imgAffine(rigidDirPath+inImgFileName, origDirPath+refImgFileName, affineDirPath, useNearestNeighborInterpolation, useRigid=False, verbose=False) # Do affine registration
    #imgCheckerBoard(affineDirPath+inImgFileName, origDirPath+refImgFileName, affineDirPath+"checkerboard.img") ###

    # Combine initial and rigid transforms and write result to file
    combinedAffine = affineApplyAffine(affine, combinedAffine)                         # Conbine rigid and affine transforms
    txtWriteParameterList(combinedAffine, affineDirPath+"combinedAffine.txt") # Write combined transform to file

    # Convert combined affine to displacement field
    fieldPath = affineDirPath+"field.vtk"
    affineToField(combinedAffine, refSize, inSpacing, fieldPath)

    # Convert inverse combined affine to displacement field
    invFieldPath = affineDirPath+"fieldInv.vtk"
    affineToField(affineInverse(combinedAffine), inSize, refSpacing, invFieldPath)

    # Apply combined transform to original input image and generate checkerboard image between original input image and reference image
    imgApplyField(origDirPath+inImgFileName, fieldPath, affineDirPath, useNearestNeighborInterpolation, refSize)
    imgCheckerBoard(affineDirPath+inImgFileName, origDirPath+refImgFileName, affineDirPath+"checkerboard.img")

    # Do diffeomorphic registration
    print("...Diffeomorphic Registrration")
    diffeoDirPath = outDirPath + "3_diffeo/"
    alpha = 0.05
    beta  = 0.05
    imgMetamorphosis(affineDirPath+inImgFileName, origDirPath+refImgFileName, diffeoDirPath, alpha, beta, useNearestNeighborInterpolation, False)
    fieldApplyField(diffeoDirPath+"field.vtk", affineDirPath+"field.vtk", diffeoDirPath+"combinedField.vtk")
    imgApplyField(origDirPath+inImgFileName, diffeoDirPath+"combinedField.vtk", diffeoDirPath, useNearestNeighborInterpolation, refSize)
    imgCheckerBoard(diffeoDirPath+inImgFileName, origDirPath+refImgFileName, diffeoDirPath+"checkerboard.img")

    # Apply displacemet field to input 
    fieldPath = outDirPath+"field.vtk"
    shutil.copyfile(diffeoDirPath+"combinedField.vtk", fieldPath)
    imgApplyField(origDirPath+inImgFileName, fieldPath, outPath, useNearestNeighborInterpolation)
    
    return (fieldPath, None)

def maskPipeline(inPath, atlasPath, atlasLabelPath, outPath, spacing=[0.1,0.1,0.1], initialAffine=identityAffine, inThreshold=None, atlasThreshold=None):
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    """
    Mask Pipeline aligns atlas to input image using their masks
    If available an initial affine transform can be provided to aid in the alignment.
    """
    if len(spacing) != dimension: raise Exception("spacing must have length {0}.".format(dimension))
    if len(initialAffine) != len(identityAffine): raise Exception("affine must have length {0}.".format(numParameters))
    inImg=imgRead(inPath)
    inSize=inImg.GetSize()
    inSpacing=inImg.GetSpacing()
    inFileName = os.path.basename(inPath)
    atlasFileName = os.path.basename(atlasPath)
    atlasLabelFileName = os.path.basename(atlasLabelPath)

    # Downsample input and atlas images to given spacing
    print("Downsampling")
    dsDirPath = outDirPath + "1_ds/"
    imgResample(inPath, dsDirPath, spacing)
    imgResample(atlasPath, dsDirPath, spacing)

    # Generate masks of input and atlas images
    print("Generating Masks")
    maskDirPath = outDirPath + "2_mask/"

    foregroundValue = 255
    imgGenerateMask(dsDirPath+inFileName, maskDirPath, inThreshold, foregroundValue)
    imgGenerateMask(dsDirPath+atlasFileName, maskDirPath, atlasThreshold, foregroundValue)

    print("Registering Masks")
    # Run mask registration
    regDirPath = outDirPath + "3_reg/"
    imgRegistration(maskDirPath+atlasFileName, maskDirPath+inFileName, regDirPath, True, initialAffine)
    
    print("Applying Deformation Field")
    atlasLabelImg = imgRead(atlasLabelPath)
    imgWrite(sitk.Cast(atlasLabelImg, sitk.sitkInt16), outDirPath+atlasLabelFileName)
    imgApplyField(atlasPath, regDirPath+"field.vtk", outDirPath, False, inSize, inSpacing)
    imgApplyField(outDirPath+atlasLabelFileName, regDirPath+"field.vtk", outDirPath, True, inSize, inSpacing)

    return outPath

def lmkApplyAffine(inPath, outPath, affine, spacing=[1.0,1.0,1.0]):
    if (not(type(spacing)) is list) or (len(spacing) != 3): raise Exception("spacing must be a list of length 3.")
    (outPath, outDirPath) = getOutPaths(inPath, outPath)

    # Apply affine to landmarks
    lmk = landmarks()
    lmk.Read(inPath, spacing)    
    transformedLmk = lmk.Affine(affine)
    transformedLmk.Write(outPath)
    return outPath

def lmkApplyField(inPath, outPath, fieldPath, spacing=[1.0, 1.0, 1.0]):
    inLmk = landmarks(inPath,  spacing)

    # Read field
    field = imgRead(fieldPath)
    field.SetDirection(identityDirection)

    # Create transform
    transform = sitk.DisplacementFieldTransform(dimension)
    transform.SetInterpolator(sitk.sitkLinear)
    transform.SetDisplacementField(sitk.Cast(field, sitk.sitkVectorFloat64))

    for lmk in inLmk.GetLandmarks():
        pass
    
    print(dir(transform))


    
