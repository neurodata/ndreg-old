#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import ndio.remote.neurodata as neurodata
import os, math, sys, subprocess, tempfile, shutil
from itertools import product

dimension = 3
vectorComponentType = sitk.sitkFloat32
vectorType = sitk.sitkVectorFloat32
affine = sitk.AffineTransform(dimension)
identityAffine = list(affine.GetParameters())
identityDirection = list(affine.GetMatrix())
zeroOrigin = [0]*dimension
zeroIndex = [0]*dimension

ndToSitkDataTypes = {'uint8': sitk.sitkUInt8,
                     'uint16': sitk.sitkUInt16,
                     'uint32': sitk.sitkUInt32,
                     'float32': sitk.sitkFloat32}

ndregDirPath = os.path.dirname(os.path.realpath(__file__))+"/"

def isIterable(obj):
    """
    Returns True if obj is a list, tuple or any other iterable object
    """
    return hasattr([],'__iter__')

def isNumber(variable):
    try:
        float(variable)
    except TypeError:
        return False
    return True

def run(command, checkReturnValue=True, quiet=True):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    outText = ""

    for line in iter(process.stdout.readline, ''):
        if not quiet:  sys.stdout.write(line)
        outText += line
    #process.wait()
    process.communicate()[0]
    returnValue = process.returncode
    if checkReturnValue and (returnValue != 0): raise Exception(outText)

    return (returnValue, outText)

def txtWrite(text, path, mode="w"):
    """
    Conveinence function to write text to a file at specified path
    """
    dirMake(os.path.dirname(path))
    textFile = open(path, mode)
    print(text, file=textFile)
    textFile.close()

def txtRead(path):
    """
    Conveinence function to read text from file at specified path
    """
    textFile = open(path,"r")
    text = textFile.read()
    textFile.close()
    return text

def txtReadList(path):
    return map(float,txtRead(path).split())

def txtWriteList(parameterList, path):
    txtWrite(" ".join(map(str,parameterList)), path)

def dirMake(dirPath):
    if not os.path.exists(dirPath): os.makedirs(dirPath)
    return os.path.normpath(dirPath) + "/"


def imgHM(inImg, refImg, numMatchPoints=8, numBins=64):
    """
    Histogram matches input image to reference image and writes result to output image
    """
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    return  sitk.HistogramMatchingImageFilter().Execute(inImg, refImg, numBins, numMatchPoints, False)

def imgRead(path):
    """
    Alias for sitk.ReadImage
    """
    inImg = sitk.ReadImage(path)
    inImg.SetDirection(identityDirection)
    inImg.SetOrigin(zeroOrigin)

    return inImg


def imgDownload(token, channel="", resolution=0, server="openconnecto.me"):
    """
    Download image with given token from given server at given resolution.
    If channel isn't specified the first channel is downloaded.
    """
    # Create neurodata instance
    nd = neurodata(suppress_warnings=True)
    nd.hostname = server

    # If channel isn't specified use first one
    channelList = nd.get_channels(token).keys()
    if len(channelList) == 0:
        raise Exception("No channels defined for given token.")
    elif channel == "": 
        channel = channelList[0] 
    elif not(channel in channelList):
        raise Exception("Channel '{0}' does not exist for given token.".format(channel))

    # Get image spacing, size, offset and data type from server
    metadata = nd.get_proj_info(token)
    spacingNm = metadata[u'dataset'][u'voxelres'][unicode(str(resolution))] # Returns spacing in nanometers 
    spacing = [x * 1e-6 for x in spacingNm] # Convert spacing to mm

    size = nd.get_image_size(token, resolution)
    offset = nd.get_image_offset(token, resolution)
    dataType = metadata['channels'][channel]['datatype']

    # Download all image data from specified channel
    array = nd.get_cutout(token, channel, offset[0], size[0], offset[1], size[1], offset[2], size[2], resolution)
    
    # Cast downloaded image to server's data type
    img = sitk.Cast(sitk.GetImageFromArray(array),ndToSitkDataTypes[dataType]) # convert numpy array to sitk image

    # Reverse axes order
    
    img = sitk.PermuteAxesImageFilter().Execute(img,range(dimension-1,-1,-1))
    img.SetDirection(identityDirection)
    img.SetSpacing(spacing)

    return img


def imgUpload(img, token, channel="", resolution=0, server="openconnecto.me"):
    """
    Upload image with given token from given server at given resolution.
    If channel isn't specified image is uploaded to default channel
    
    TODO: Check bounds
    """
    # Create neurodata instance
    nd = neurodata()
    nd.hostname = server

    # If channel isn't specified use first one
    channelList = nd.get_channels(token).keys()
    if len(channelList) == 0:
        raise Exception("No channels defined for token {0}.".format(token))
    elif channel == "": 
        channel = channelList[0]
    elif not(channel in channelList):
        raise Exception("Channel '{0}' does not exist for given token.".format(channel))

    # Get image data type from server
    metadata = nd.get_proj_info(token)
    dataType = metadata['channels'][channel]['datatype']

    # Cast input image to server's data type
    castImg = sitk.Cast(img,ndToSitkDataTypes[dataType])

    # Reverse axis order
    array = sitk.GetArrayFromImage(castImg).transpose(range(dimension-1,-1,-1))

    # Upload all image data from specified channel
    nd.post_cutout(token, channel, 0, 0, 0, data=array, resolution=resolution)


def imgWrite(img, path):
    """
    Write sitk image to path.
    """
    dirMake(os.path.dirname(path))
    sitk.WriteImage(img, path)

    # Reformat files to be compatible with CIS Software
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtk": vtkReformat(path, path)
        
    return path

def vtkReformat(inPath, outPath):
    """
    Reformats vtk file so that it can be read by CIS software.
    """
    # Get size of map
    inFile = open(inPath,"rb")
    lineList = inFile.readlines()
    for line in lineList:
        if line.lower().strip().startswith("dimensions"):
            size = map(int,line.split(" ")[1:dimension+1])
            break
    inFile.close()

    if dimension == 2: size += [0]

    outFile = open(outPath,"wb")
    for (i,line) in enumerate(lineList):
        if i == 1:
            newline = line.lstrip(line.rstrip("\n"))
            line = "lddmm 8 0 0 {0} {0} 0 0 {1} {1} 0 0 {2} {2}".format(size[2]-1, size[1]-1, size[0]-1) + newline
        outFile.write(line)


def imgResample(img, spacing, size=[], useNearestNeighborInterpolation=False):
    """
    Resamples image to given spacing and size.
    """
    if len(spacing) != dimension: raise Exception("len(spacing) != " + str(dimension))

    # Set Size
    if size == []:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [int(math.ceil(inSize[i]*(inSpacing[i]/spacing[i]))) for i in range(dimension)]
    else:
        if len(size) != dimension: raise Exception("len(size) != " + str(dimension))
    
    # Resample input image
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearestNeighborInterpolation]
    identityTransform = sitk.Transform()
    
    return sitk.Resample(img, size, identityTransform, interpolator, zeroOrigin, spacing)

def imgLargestMaskObject(maskImg):
    ccFilter = sitk.ConnectedComponentImageFilter()
    labelImg = ccFilter.Execute(maskImg)
    numberOfLabels = ccFilter.GetObjectCount()
    labelArray = sitk.GetArrayFromImage(labelImg)
    largestLabel = 0
    largestSize = 0
    for i in range(1,numberOfLabels+1):
        size = sum(labelArray == i)
        if size > largestSize:
            largestSize = size
            largestLabel = i

    outImg = sitk.GetImageFromArray((labelArray==largestLabel).astype(np.int16))
    outImg.CopyInformation(maskImg) # output image should have same metadata as input mask image
    return outImg

def imgMakeMask(inImg, threshold=None, forgroundValue=1):
    """
    Generates morphologically smooth mask with given forground value from input image.
    If the threshold is given, the binary mask is initialzed using the given threshold...
    ...Otherwise it is initialized using Otsu's Method.
    """
    
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

    return outMask

def imgMask(img, mask):
    """
    Convenience function to apply mask to image
    """
    return  sitk.MaskImageFilter().Execute(img, mask)

def sizeOut(inImg, transform, outSpacing):
    """
    Calculates size of bounding box which encloses transformed image
    """
    outCornerPointList = []
    inSize = inImg.GetSize()
    for corner in product((0,1), repeat=dimension):
        inCornerIndex = np.array(corner)*np.array(inSize)
        inCornerPoint = inImg.TransformIndexToPhysicalPoint(inCornerIndex)
        outCornerPoint = transform.GetInverse().TransformPoint(inCornerPoint)
        outCornerPointList += [list(outCornerPoint)]

    size = np.ceil(np.array(outCornerPointList).max(0) / outSpacing).astype(int)
    return size

def affineToField(affine, size, spacing):
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
    return  sitk.TransformToDisplacementFieldFilter().Execute(affineTransform, vectorType, size, zeroOrigin, spacing, identityDirection)

def imgApplyField(img, field, useNearest=False, size=[], spacing=[],defaultValue=0):
    """
    img \circ field
    """
    field = sitk.Cast(field, sitk.sitkVectorFloat64)

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]

    # Set transform field
    transform = sitk.DisplacementFieldTransform(dimension)
    transform.SetInterpolator(sitk.sitkLinear)
    transform.SetDisplacementField(field)

    # Set size
    if size == []:
        size = img.GetSize()
    else:
        if len(size) != dimension: raise Exception("size must have length {0}.".format(dimension))

    # Set Spacing
    if spacing == []:
        spacing = img.GetSpacing()
    else:
        if len(spacing) != dimension: raise Exception("spacing must have length {0}.".format(dimension))
    
    # Apply displacement transform
    return  sitk.Resample(img, size, transform, interpolator, zeroOrigin, spacing, img.GetDirection() ,defaultValue)
    
def imgApplyAffine(inImg, affine, useNearest=False, size=[], spacing=[]):
    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]

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
        size = sizeOut(inImg, affineTransform, spacing)
    else:
       if len(size) != dimension: raise Exception("size must have length {0}.".format(dimension))
    
    # Apply affine transform
    outImg = sitk.Resample(inImg, size, affineTransform, interpolator, zeroOrigin, spacing)

    return outImg


def affineInverse(affine):
    # x0 = A0*x1 + b0
    # x1 = (A0.I)*x0 + (-A0.I*b0) = A1*x0 + b1
    A0 = np.mat(affine[0:9]).reshape(3,3)
    b0 = np.mat(affine[9:12]).reshape(3,1)

    A1 = A0.I
    b1 = -A1*b0
    return A1.flatten().tolist()[0] + b1.flatten().tolist()[0]

def affineApplyAffine(inAffine, affine):
    """ A_{outAffine} = A_{inAffine} \circ A_{affine} """
    if (not(isIterable(inAffine))) or (len(inAffine) != 12): raise Exception("inAffine must be a list of length 12.")
    if (not(isIterable(affine))) or (len(affine) != 12): raise Exception("affine must be a list of length 12.")
    A0 = np.array(affine[0:9]).reshape(3,3)
    b0 = np.array(affine[9:12]).reshape(3,1)
    A1 = np.array(inAffine[0:9]).reshape(3,3)
    b1 = np.array(inAffine[9:12]).reshape(3,1)

    # x0 = A0*x1 + b0
    # x1 = A1*x2 + b1
    # x0 = A0*(A1*x2 + b1) + b0 = (A0*A1)*x2 + (A0*b1 + b0)
    A = np.dot(A0,A1)
    b = np.dot(A0,b1) + b0

    outAffine = A.flatten().tolist() + b.flatten().tolist()
    return outAffine

def fieldApplyField(inField, field):
    """ outField = inField \circ field """
    inField = sitk.Cast(inField, sitk.sitkVectorFloat64)
    field = sitk.Cast(field, sitk.sitkVectorFloat64)
    
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
    return sitk.TransformToDisplacementFieldFilter().Execute(outTransform, vectorType, size, zeroOrigin, spacing, identityDirection)


def imgChecker(inImg, refImg, useHM=True):
    """
    Checkerboards input image with reference image
    """    
    inSize = list(inImg.GetSize())
    refSize = list(refImg.GetSize())
    
    if(inSize != refSize):
        sourceSize = np.array([inSize, refSize]).min(0)
        tmpImg = sitk.Image(refSize,refImg.GetPixelID()) # Empty image with same size as reference image
        tmpImg.CopyInformation(refImg)
        inImg = sitk.PasteImageFilter().Execute(tmpImg, inImg, sourceSize, zeroIndex, zeroIndex)

    if useHM:
        numBins = 64
        numMatchPoints = 8
        inImg = sitk.Cast(inImg, refImg.GetPixelID())
        inImg = sitk.HistogramMatchingImageFilter().Execute(inImg, refImg, numBins, numMatchPoints, False)

    return sitk.CheckerBoardImageFilter().Execute(inImg, refImg,[8]*dimension)

def imgAffine(inImg, refImg, useNearestNeighborInterpolation=False, useRigid=False, useMI=False, iterations=1000, verbose=False):
    """
    Perform Affine Registration between input image and reference image
    """

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearestNeighborInterpolation]
    
    # Set transform
    transform = [sitk.AffineTransform(dimension), sitk.Similarity3DTransform()][useRigid]

    # Do registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetInterpolator(interpolator)
    registration.SetInitialTransform(transform)

    if useMI:
        numHistogramBins = 64
        registration.SetMetricAsMattesMutualInformation(numHistogramBins)
        learningRate=0.01
    else:
        registration.SetMetricAsMeanSquares()
        learningRate=0.000001
    registration.SetOptimizerAsGradientDescent(learningRate=learningRate, numberOfIterations=iterations)
    if(verbose): registration.AddCommand(sitk.sitkIterationEvent, lambda: print("{0}.\t {1}".format(registration.GetOptimizerIteration(),registration.GetMetricValue())))

    registration.Execute(sitk.SmoothingRecursiveGaussian(refImg,0.25),
                         sitk.SmoothingRecursiveGaussian(inImg,0.25) )


    # Write final transform parameters to text file
    affine = list(transform.GetMatrix()) + list(transform.GetTranslation())
    return affine
    

def imgMetamorphosis(inImg, refImg, alpha=0.02, beta=0.05, scale=1.0, iterations=1000, useNearestNeighborInterpolation=False, useBias=True, useMI=False, verbose=True, outDirPath=""):
    """
    Performs Metamorphic LDDMM between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)
    inPath = outDirPath + "in.img"
    imgWrite(inImg, inPath)
    refPath = outDirPath + "ref.img"
    imgWrite(refImg, refPath)
    outPath = outDirPath + "out.img"

    fieldPath = outDirPath + "field.vtk"
    invFieldPath = outDirPath + "invField.vtk"

    binPath = ndregDirPath + "metamorphosis "
    command = binPath + " --in {0} --ref {1} --out {2} --alpha {3} --beta {4} --field {5} --invfield {6} --iterations {7} --scale {8} --verbose ".format(inPath, refPath, outPath, alpha, beta, fieldPath, invFieldPath, iterations, scale)
    if(not useBias): command += " --mu 0"
    if(useMI):
        command += " --cost 1 --sigma 1e-4 --epsilon 1e-2"
        
    os.system(command)
    #run(command, quiet=not(verbose))

    field = imgRead(fieldPath)
    invField = imgRead(invFieldPath)
    
    if useTempDir: shutil.rmtree(outDirPath)
    return (field, invField)


def imgMetamorphosisCascading(inImg, refImg, alphaList=0.02, betaList=0.05, scaleList=1.0, iterations=1000, useNearestNeighborInterpolation=False, useBias=True, useMI=False, verbose=True, outDirPath=""):
    """
    Performs Metamorphic LDDMM between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)

    if isNumber(alphaList): alphaList = [float(alphaList)]
    if isNumber(betaList): betaList = [float(betaList)]
    if isNumber(scaleList): scaleList = [float(scaleList)]
    
    numSteps = max(len(alphaList), len(betaList), len(scaleList))

    if len(alphaList) != numSteps:
        if len(alphaList) != 1:
            raise Exception("Legth of alphaList must be 1 or same length as betaList or scaleList")
        else:
            alphaList *= numSteps

    if len(betaList) != numSteps:
        if len(betaList) != 1:
            raise Exception("Legth of betaList must be 1 or same length as alphaList or scaleList")
        else:
            betaList *= numSteps
        
    if len(scaleList) != numSteps:
        if len(scaleList) != 1:
            raise Exception("Legth of scaleList must be 1 or same length as alphaList or betaList")
        else:
            scaleList *= numSteps

    origInImg = inImg
    for step in range(numSteps):
        alpha = alphaList[step]
        beta = betaList[step]
        scale = scaleList[step]
        stepDirPath = outDirPath + "step" + str(step) + "/"
        if(verbose): print("Step {0}: alpha={1}, beta={2}, scale={3}".format(step,alpha, beta, scale))
        
        (field, invField) = imgMetamorphosis(inImg, refImg, 
                                             alpha, 
                                             beta, 
                                             scale, 
                                             iterations, 
                                             useNearestNeighborInterpolation, 
                                             useBias, 
                                             useMI, 
                                             verbose, 
                                             outDirPath=stepDirPath)

        if step == 0:
            compositeField = field
            compositeInvField = invField
        else:
            compositeField = fieldApplyField(field, compositeField)
            compositeInvField = fieldApplyField(compositeInvField, invField)

            fieldPath = stepDirPath+"field.vtk"
            imgWrite(compositeField, fieldPath)

            invFieldPath = stepDirPath+"invField.vtk"
            imgWrite(compositeInvField, invFieldPath)

        inImg = imgApplyField(origInImg, compositeField, size=refImg.GetSize())


    # Write final results
    imgWrite(compositeField, outDirPath+"field.vtk")
    imgWrite(compositeInvField, outDirPath+"invField.vtk")
    imgWrite(inImg, outDirPath+"out.img")
    imgWrite(imgChecker(inImg,refImg), outDirPath+"checker.img")

    # Clean up if necessary
    if useTempDir: shutil.rmtree(outDirPath)
    
    return (compositeField, compositeInvField)

def imgRegistration(inImg, refImg, scale=1.0, alphaList=[0.02], iterations=1000, useMI=False, verbose=False, outDirPath=""):
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)

    initialDirPath = outDirPath + "0_initial/"
    rigidDirPath = outDirPath + "1_rigid/"
    affineDirPath = outDirPath + "2_affine/"
    lddmmDirPath = outDirPath + "3_lddmm/"
    origInImg = inImg
    origRefImg = refImg

    # Resample and histogram match in and ref images
    refSpacing = refImg.GetSpacing()
    spacing = [x / scale for x in refSpacing]
    inImg = imgResample(inImg, spacing)
    refImg = imgResample(refImg, spacing)    
    if not useMI: inImg = imgHM(inImg, refImg)
    initialInImg = inImg

    imgWrite(inImg, initialDirPath+"in.img")
    imgWrite(refImg, initialDirPath+"ref.img")

    # Rigid align in and ref images
    if(verbose): print("Rigid alignment")
    rigid = imgAffine(inImg, refImg, useRigid=True, useMI=useMI, iterations=iterations, verbose=verbose)
    inImg = imgApplyAffine(initialInImg, rigid)

    txtWriteList(rigid, rigidDirPath+"affine.txt")
    txtWriteList(affineInverse(rigid), rigidDirPath+"invAffine.txt")
    imgWrite(inImg, rigidDirPath+"in.img")


    # Affine align in and ref images
    if(verbose): print("Affine alignment")
    affine = imgAffine(inImg, refImg, useRigid=False, useMI=useMI, iterations=iterations, verbose=verbose)
    affine = affineApplyAffine(affine, rigid)
    affineField = affineToField(affine, inImg.GetSize(), inImg.GetSpacing())
    invAffine = affineInverse(affine)
    invAffineField = affineToField(invAffine, refImg.GetSize(), refImg.GetSpacing())
    inImg = imgApplyField(initialInImg, affineField, size=refImg.GetSize())

    txtWriteList(affine, affineDirPath+"affine.txt")
    txtWriteList(affine, affineDirPath+"invAffine.txt")
    imgWrite(affineField, affineDirPath+"field.vtk")
    imgWrite(invAffineField, affineDirPath+"invField.vtk")
    imgWrite(inImg, affineDirPath+"in.img")

    # Deformably align in and ref images
    if(verbose): print("Deformable alignment")
    (field, invField) = imgMetamorphosisCascading(inImg, refImg, alphaList=alphaList, scaleList=[0.5], useBias=False, useMI=useMI, verbose=verbose, iterations=iterations, outDirPath=lddmmDirPath)
                        #imgMetamorphosisCascading(inImg, refImg, alphaList=0.02, betaList=0.05, scaleList=1.0, iterations=1000, useNearestNeighborInterpolation=False, useBias=True, useMI=False, verbose=True, outDirPath="")
    field = fieldApplyField(field, affineField)
    invField = fieldApplyField(invAffineField, invField)
    inImg = imgApplyField(initialInImg, field, size=refImg.GetSize())

    imgWrite(field, lddmmDirPath+"field.vtk")
    imgWrite(invField, lddmmDirPath+"invField.vtk")
    imgWrite(inImg, lddmmDirPath+"in.img")    

    if useTempDir: shutil.rmtree(outDirPath)
    return (field, invField)
