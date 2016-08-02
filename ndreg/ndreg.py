#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import ndio.remote.neurodata as neurodata
import os, math, sys, subprocess, tempfile, shutil, requests
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


sitkToNpDataTypes = {sitk.sitkUInt8: np.uint8,
                     sitk.sitkUInt16: np.uint16,
                     sitk.sitkUInt32: np.uint32,
                     sitk.sitkInt8: np.int8,
                     sitk.sitkInt16: np.int16,
                     sitk.sitkInt32: np.int32,
                     sitk.sitkFloat32: np.float32,
                     sitk.sitkFloat64: np.float64,
                     }


ndregDirPath = os.path.dirname(os.path.realpath(__file__))+"/"
ndregTranslation = 0
ndregRigid = 1
ndregAffine = 2 

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
    if dirPath != "":
        if not os.path.exists(dirPath): os.makedirs(dirPath)
        return os.path.normpath(dirPath) + "/"
    else:
        return dirPath


def imgHM(inImg, refImg, numMatchPoints=32, numBins=256):
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
    if(inImg.GetDimension() == 2): inImg = sitk.JoinSeriesImageFilter().Execute(inImg)
    inImg.SetDirection(identityDirection)
    inImg.SetOrigin(zeroOrigin)

    return inImg


def imgDownload(token, channel="", resolution=0, server="openconnecto.me", size=[]):
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

    if size == []: size = nd.get_image_size(token, resolution)
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

def limsGetMetadata(token):
    nd = neurodata()
    r = requests.get(nd.meta_url("metadata/ocp/get/" + token))
    return r.json()

def projGetMetadata(token):
    nd = neurodata()
    return nd.get_proj_info(token)

def limsSetMetadata(token, metadata):
    nd = neurodata()
    nd.set_metadata(token, metadata)

def imgPreprocess(inToken, refToken="", inChannel="", inResolution=0, refResolution=0, outDirPath="", doSteps=[1,1,1], verbose=True):
    """
    Downloads, resamples and reorients input image to the spacing and orientation of the reference image.
    This function assumes that the input token (and reference token if used) has a corresponding LIMS token.
    The LIMS metadata should contain \"spacing\" and \"orientation\" fields.
    The \"spacing\" field should specify the x, y and z spacing of the image in millimeters and e.g. spacing=[0.025, 0.025, 0.025]
    The \"orientation\" field should be orientation strings specifying the x, y, z orientation of the image e.g. orientation="las"
    See documentation of the \"imgReorient\" function for more details on orientation strings.
    """
    if outDirPath != "": outDirPath = dirMake(outDirPath)
    inMask = None

    # Create NeuroData instance
    nd = neurodata()

    # Set reference default spacing and orientation
    refSpacing = [0.025, 0.025, 0.025] 
    refOrient = "rsa" # Coronal sliced images

    if refToken != "":
        # Get project metadata for reference token
        try:
            refProjMetadata = nd.get_proj_info(refToken)
        except:
            raise Exception("Reference project token {0} was not found on server {1}".format(refToken, nd.hostname))

        # Get lims metadata (spacing and orientation) of reference token
        refLimsMetadata = limsGetMetadata(refToken)
        
        if "spacing" in refLimsMetadata:
            refSpacing = refLimsMetadata["spacing"]

            # Scale reference spacing values of input image based on resolution level
            for i in range(0, dimension-1): refSpacing[i] *= 2**refResolution
            if refProjMetadata['dataset']['scaling'] != 'zslices': refSpacing[dimension-1] *= 2**refResolution
        else:
            if verbose: print("Warning: Reference LIMS token {0} does not have an \"spacing\" feild. Using value of {1} from project token {0}".format(refToken, refSpacing))

        if "orientation" in refLimsMetadata:
            refOrient = refLimsMetadata["orientation"]
        else:
            if verbose: print("Warning: Reference LIMS token {0} does not have an \"orientation\" feild. Usign default value of ".format(refToken, refOrient))

    # Get project metadata for input token
    try:
        inProjMetadata = nd.get_proj_info(inToken)
    except:
        raise Exception("Token {0} was not found on server {1}".format(inToken, nd.hostname))

    # Get lims metadata for input token
    inLimsMetadata = limsGetMetadata(inToken)

    # Download input image 
    if doSteps[0]:
        if verbose: print("Downloading input image")
        inImg = imgDownload(inToken, channel=inChannel, resolution=inResolution)

        # Check if downloaded image is empty
        epsilon = sys.float_info.epsilon    
        stats = sitk.StatisticsImageFilter()
        stats.Execute(inImg)

        if (stats.GetMean() < epsilon) and (stats.GetVariance() < epsilon):
            raise Exception("Error: Input image downoaded from token {0} is empty".format(inToken))

        inSpacing = inImg.GetSpacing()

        # Set input image's spacing based on lims metadata 
        if 'spacing' in inLimsMetadata.keys():
            # If lims metadata contains a spacing field then set spacing of downloaded image using it
            inSpacing = inLimsMetadata['spacing']

            # Scale spacing values of input image based on resolution level
            for i in range(0, dimension-1): inSpacing[i] *= 2**inResolution
            if inProjMetadata['dataset']['scaling'] != 'zslices': inSpacing[dimension-1] *= 2**inResolution
            inImg.SetSpacing(inSpacing)
        else:
            if verbose: print("Warning: LIMS token {0} does not have an \"spacing\" feild.  Using value of {1} from project token {0}".format(inToken, inSpacing))

        if outDirPath != "": imgWrite(inImg,outDirPath+"/0_download/in.img")

    # Resample input image to spacing of reference image
    if doSteps[1]:
        if verbose: print("Resampling input image")
        if outDirPath != "": inImg = imgRead(outDirPath+"/0_download/in.img")
        inImg = imgResample(inImg, refSpacing)
        if outDirPath != "": imgWrite(inImg,outDirPath+"/1_resample/in.img")

    # Reorient input image to orientation of reference image
    if doSteps[2]:
        if outDirPath != "": inImg = imgRead(outDirPath+"/1_resample/in.img")
        if "orientation" in inLimsMetadata.keys():
            if verbose: print("Reorienting input image")
            # If lims metadata contains a orientation field then reorient image
            inOrient = inLimsMetadata["orientation"]
            inImg = imgReorient(inImg, inOrient, refOrient)
            
            """
            # If there's there's affine info for the reference orientation then apply it
            if refOrient+"Affine" in inLimsMetadata.keys():
                affine = inLimsMetadata[refOrient+"Affine"]
                inImg = imgApplyAffine(inImg, affine, size=inImg.GetSize())
            """
        else:
            if verbose: print("Warning: LIMS token {0} does not have an \"orientation\" feild. Could not reorient image.".format(inToken))

        if outDirPath != "": imgWrite(inImg,outDirPath+"/2_reorient/in.img")

    if outDirPath != "": 
        imgWrite(inImg, outDirPath+"/in.img")

    return inImg

def imgPostprocess(inImg, refToken, outToken, useNearest=False, doSteps=[1,1], verbose=False, outDirPath=""):
    if outDirPath != "": outDirPath = dirMake(outDirPath)

    refLimsMetadata = limsGetMetadata(refToken)
    refOrient = refLimsMetadata["orientation"]

    outProjMetadata = projGetMetadata(outToken)
    inToken = outProjMetadata["dataset"]["name"] 
    inLimsMetadata = limsGetMetadata(inToken)
    inOrient = inLimsMetadata["orientation"]
    
    if doSteps[0]:
        if verbose: print("Reorienting image")
        """
        if refOrient+"Affine" in inLimsMetadata.keys():
            invAffine = affineInverse(inLimsMetadata[refOrient+"Affine"])
            inImg = imgApplyAffine(inImg, invAffine, useNearest=useNearest, size=inImg.GetSize())
        """
        inImg = imgReorient(inImg, refOrient, inOrient)
        if outDirPath !="": imgWrite(inImg, outDirPath+"0_reorient/in.img")

    if doSteps[1]:
        nd = neurodata()
        outResolution = 5
        outSpacing = inLimsMetadata["spacing"]
        outSize = list(np.array(nd.get_image_size(outToken, outResolution)) - np.array(nd.get_image_offset(outToken, outResolution)))    

        for i in range(dimension-1): outSpacing[i] *= 2**outResolution
        if outProjMetadata["dataset"]["scaling"] != "zslices": outSpacing[dimension-1] *= 2**outResolution

        inImg = imgResample(inImg, spacing=outSpacing, size=outSize, useNearest=useNearest)
        if outDirPath !="": imgWrite(inImg, outDirPath+"1_resample/in.img")

        if verbose: print("Uploading results")
        outChannel = "annotation"
        imgUpload(inImg, outToken, channel=outChannel, resolution=outResolution)


def imgUpload(img, token, channel="", resolution=0, start=[0,0,0], server="openconnecto.me",  propagate=False):
    """
    Upload image with given token from given server at given resolution.
    If channel isn't specified image is uploaded to default channel
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


    # Convert with RGB or RGBA multi-component images to uint32
    numComponents = img.GetNumberOfComponentsPerPixel()
    if numComponents in [3,4]:
        for component in range(numComponents):
            #  Extract component
            caster = sitk.VectorIndexSelectionCastImageFilter()
            caster.SetIndex(component)
            componentImg = caster.Execute(img)

            # Scale intensity to 8-bit range
            componentDatatype = sitkToNpDataTypes[componentImg.GetPixelID()]
            
            componentMin = np.iinfo(componentDatatype).min
            componentMax = np.iinfo(componentDatatype).max
            componentImg = sitk.IntensityWindowingImageFilter().Execute(componentImg, componentMin, componentMax, 0, 255)
            componentImg = sitk.Cast(componentImg, sitk.sitkUInt32)

            # Create RGBA image with red component in lowest order byte followed by green and blue (and alpha)
            if component == 0:
                uint32Img = componentImg * 256**component
            else:
                uint32Img += componentImg * 256**component

        # Fill alpha byte if no alpha component was given
        if numComponents == 3: uint32Img += 255*256**3
        img = uint32Img

    elif numComponents == 2 or numComponents > 4:
        raise Exception("Error: Expected scaler, RGB or RGBA image")

    # Get image size from server
    offset = nd.get_image_offset(token, resolution)
        
    serverSize = list(np.array(nd.get_image_size(token, resolution)) - np.array(offset))    
    imgSize = img.GetSize()

    # Raise exception if input image is too big 
    for i in range(dimension):
        if imgSize[i] > serverSize[i]: raise Exception("Input image with size {0} excedes bounds of token {1} with size {2}".format(imgSize, token, datasetSize))

    # Get image data type from server
    metadata = nd.get_proj_info(token)
    dataType = metadata['channels'][channel]['datatype']

    # Cast input image to server's data type
    castImg = sitk.Cast(img,ndToSitkDataTypes[dataType])

    # Reverse axis order
    array = sitk.GetArrayFromImage(castImg).transpose(range(dimension-1,-1,-1))

    # Upload all image data from specified channel
    nd.post_cutout(token, channel, offset[0]+start[0], offset[1]+start[1], offset[2]+start[2], data=array, resolution=resolution)    

    # Propagate
    if propagate: nd.propagate(token, channel)

def imgWrite(img, path):
    """
    Write sitk image to path.
    """
    dirMake(os.path.dirname(path))
    sitk.WriteImage(img, path)

    # Reformat files to be compatible with CIS Software
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtk": vtkReformat(path, path)

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


def imgResample(img, spacing, size=[], useNearest=False):
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
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    identityTransform = sitk.Transform()
    
    return sitk.Resample(img, size, identityTransform, interpolator, zeroOrigin, spacing)


def imgLargestMaskObject(maskImg):
    ccFilter = sitk.ConnectedComponentImageFilter()
    labelImg = ccFilter.Execute(maskImg)
    numberOfLabels = ccFilter.GetObjectCount()
    labelArray = sitk.GetArrayFromImage(labelImg)
    labelSizes = np.bincount(labelArray.flatten())
    largestLabel = np.argmax(labelSizes[1:])+1
    outImg = sitk.GetImageFromArray((labelArray==largestLabel).astype(np.int16))
    outImg.CopyInformation(maskImg) # output image should have same metadata as input mask image
    return outImg

def createTmpRegistration(inMask=None, refMask=None,samplingPercentage=0.01):
    identityTransform = sitk.Transform()
    tmpRegistration = sitk.ImageRegistrationMethod()
    tmpRegistration.SetMetricSamplingStrategy(tmpRegistration.RANDOM)
    tmpRegistration.SetMetricSamplingPercentage(samplingPercentage)
    tmpRegistration.SetInterpolator(sitk.sitkNearestNeighbor)
    tmpRegistration.SetInitialTransform(identityTransform)
    tmpRegistration.SetOptimizerAsGradientDescent(learningRate=1e-14, numberOfIterations=1)
    if(inMask): tmpRegistration.SetMetricMovingMask(inMask)
    if(refMask): tmpRregistration.SetMetricFixedMask(refMask)

    return tmpRegistration

def imgMI(inImg, refImg, inMask=None, refMask=None, numBins=64):
    """
    Compute mattes mutual information between input and reference images
    """
    
    # In SimpleITK the metric can't be accessed directly.
    # Therefore we create a do-nothing registration method which uses an identity transform to get the metric value
    tmpRegistration = createTmpRegistration(inMask, refMask)
    tmpRegistration.SetMetricAsMattesMutualInformation(numBins)
    
    tmpRegistration.Execute( sitk.Cast(refImg,sitk.sitkFloat32),sitk.Cast(inImg, sitk.sitkFloat32) )

    return -tmpRegistration.GetMetricValue()

def imgMSE(inImg, refImg, inMask=None, refMask=None):
    """
    Compute mean square error between input and reference images
    """
    
    tmpRegistration = createTmpRegistration(inMask, refMask)
    tmpRegistration.SetMetricAsMeanSquares()
    tmpRegistration.Execute( sitk.Cast(refImg,sitk.sitkFloat32),sitk.Cast(inImg, sitk.sitkFloat32) )

    return tmpRegistration.GetMetricValue()


def imgMakeRGBA(imgList, dtype=sitk.sitkUInt8):
    if len(imgList) < 3 or len(imgList) > 4: raise Exception("imgList must contain 3 ([r,g,b]) or 4 ([r,g,b,a]) channels.")
    
    inDatatype = sitkToNpDataTypes[imgList[0].GetPixelID()]
    outDatatype = sitkToNpDataTypes[dtype]
    inMin = np.iinfo(inDatatype).min
    inMax = np.iinfo(inDatatype).max
    outMin = np.iinfo(outDatatype).min
    outMax = np.iinfo(outDatatype).max

    castImgList = []
    for img in imgList:
        castImg = sitk.Cast(sitk.IntensityWindowingImageFilter().Execute(img, inMin, inMax, outMin, outMax), dtype)
        castImgList.append(castImg)

    if len(imgList) == 3:        
        channelSize = list(imgList[0].GetSize())
        alphaArray = outMax*np.ones(channelSize[::-1], dtype=outDatatype)
        alphaChannel = sitk.GetImageFromArray(alphaArray)
        alphaChannel.CopyInformation(imgList[0])
        castImgList.append(alphaChannel)

    return sitk.ComposeImageFilter().Execute(castImgList)


def imgMakeMask(inImg, threshold=None, forgroundValue=1):
    """
    Generates morphologically smooth mask with given forground value from input image.
    If a threshold is given, the binary mask is initialzed using the given threshold...
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
    openingRadiusMM = 0.05  # In mm
    closingRadiusMM = 0.2   # In mm
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

    return imgLargestMaskObject(outMask)

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

def imgReorient(inImg, inOrient, outOrient):
    """
    Reorients image from input orientation inOrient to output orientation outOrient.
    inOrient and outOrient must be orientation strings specifying the orientation of the image.
    For example an orientation string of "las" means that the ...
        x-axis increases from \"l\"eft to right
        y-axis increases from \"a\"nterior to posterior
        z-axis increases from \"s\"uperior to inferior
    Thus using inOrient = "las" and outOrient = "rpi" reorients the input image from left-anterior-superior to right-posterior-inferior orientation.
    """

    if (len(inOrient) != dimension) or not isinstance(inOrient, basestring): raise Exception("inOrient must be a string of length {0}.".format(dimension))
    if (len(outOrient) != dimension) or not isinstance(outOrient, basestring): raise Exception("outOrient must be a string of length {0}.".format(dimension))
    inOrient = str(inOrient).lower()
    outOrient = str(outOrient).lower()
    
    inDirection = ""
    outDirection = ""
    orientToDirection = {"r":"r", "l":"r", "s":"s", "i":"s", "a":"a", "p":"a"}
    for i in range(dimension):
        try:
            inDirection += orientToDirection[inOrient[i]]
        except:
            raise Exception("inOrient \'{0}\' is invalid.".format(inOrient))

        try:
            outDirection += orientToDirection[outOrient[i]]
        except:
            raise Exception("outOrient \'{0}\' is invalid.".format(outOrient))
        
    if len(set(inDirection)) != dimension: raise Exception("inOrient \'{0}\' is invalid.".format(inOrient))
    if len(set(outDirection)) != dimension: raise Exception("outOrient \'{0}\' is invalid.".format(outOrient))

    order = []
    flip = []
    for i in range(dimension):
        j = outDirection.find(inDirection[i])
        order += [j]
        flip += [inOrient[i] != outOrient[j]]

    outImg = sitk.FlipImageFilter().Execute(inImg, flip, False)
    outImg = sitk.PermuteAxesImageFilter().Execute(outImg, order)
    outImg.SetDirection(identityDirection)
    outImg.SetOrigin(zeroOrigin)

    return outImg

def imgChecker(inImg, refImg, useHM=True, pattern=[4]*dimension):
    """
    Checkerboards input image with reference image
    """    
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    inSize = list(inImg.GetSize())
    refSize = list(refImg.GetSize())

    if(inSize != refSize):
        sourceSize = np.array([inSize, refSize]).min(0)
        tmpImg = sitk.Image(refSize,refImg.GetPixelID()) # Empty image with same size as reference image
        tmpImg.CopyInformation(refImg)
        inImg = sitk.PasteImageFilter().Execute(tmpImg, inImg, sourceSize, zeroIndex, zeroIndex)

    if useHM: inImg = imgHM(inImg, refImg)

    return sitk.CheckerBoardImageFilter().Execute(inImg, refImg,pattern)

def imgAffine(inImg, refImg, method=ndregAffine, scale=1.0, useNearest=False, useMI=False, iterations=1000, inMask=None, refMask=None, verbose=False):
    """
    Perform Affine Registration between input image and reference image
    """
    # Rescale images
    refSpacing = refImg.GetSpacing()
    spacing = [x / scale for x in refSpacing]
    inImg = imgResample(inImg, spacing, useNearest=useNearest)
    refImg = imgResample(refImg, spacing, useNearest=useNearest)
    if(inMask): imgResample(inMask, spacing, useNearest=False)
    if(refMask): imgResample(refMask, spacing, useNearest=False)

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    
    # Set transform
    try:
        transform = [sitk.TranslationTransform(dimension), sitk.Similarity3DTransform(), sitk.AffineTransform(dimension)][method]
    except:
        raise Exception("method is invalad")

    # Do registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetInterpolator(interpolator)
    registration.SetInitialTransform(transform)

    if(inMask): registration.SetMetricMovingMask(inMask)
    if(refMask): registration.SetMetricFixedMask(refMask)
    
    if useMI:
        numHistogramBins = 64
        registration.SetMetricAsMattesMutualInformation(numHistogramBins)
 
    else:
        registration.SetMetricAsMeanSquares()

    learningRate=0.1
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=learningRate, numberOfIterations=iterations, estimateLearningRate=registration.EachIteration,minStep=0.001)
    if(verbose): registration.AddCommand(sitk.sitkIterationEvent, lambda: print("{0}.\t {1} \t{2}".format(registration.GetOptimizerIteration(),registration.GetMetricValue(), registration.GetOptimizerLearningRate())))
                    
    registration.Execute(sitk.SmoothingRecursiveGaussian(refImg,0.25),
                         sitk.SmoothingRecursiveGaussian(inImg,0.25) )

    if method == ndregTranslation:
        affine = identityAffine[0:dimension**2] + list(transform.GetOffset())
    else:
        affine = list(transform.GetMatrix()) + list(transform.GetTranslation())

    return affine

def imgAffineComposite(inImg, refImg, scale=1.0, useNearest=False, useMI=False, iterations=1000, inAffine=identityAffine,verbose=False, inMask=None, refMask=None, outDirPath=""):
    if outDirPath != "": outDirPath = dirMake(outDirPath)

    origInImg = inImg
    origInMask = inMask
    origRefMask = refMask

    #initilize using input affine
    compositeAffine = inAffine
    inImg = imgApplyAffine(origInImg, compositeAffine)
    if(inMask): inMask = imgApplyAffine(origInMask, compositeAffine, useNearest=True)

    if outDirPath != "":
        imgWrite(inImg, outDirPath+"0_initial/in.img")
        if(inMask): imgWrite(inMask, outDirPath+"0_initial/inMask.img")
        txtWriteList(compositeAffine, outDirPath+"0_initial/affine.txt")

    methodList = [ndregTranslation, ndregRigid, ndregAffine]
    methodNameList = ["translation", "rigid", "affine"]
    for (step, method) in enumerate(methodList):
        methodName = methodNameList[step]
        stepDirPath = outDirPath + str(step+1) + "_" + methodName + "/"
        dirMake(stepDirPath)
        if(verbose): print("Step {0}:".format(methodName))

        affine = imgAffine(inImg, refImg, method=method, scale=scale, useNearest=useNearest, useMI=useMI, iterations=iterations, inMask=inMask, refMask=refMask, verbose=verbose)
        compositeAffine = affineApplyAffine(affine, compositeAffine)

        inImg = imgApplyAffine(origInImg, compositeAffine, size=refImg.GetSize())
        if(inMask): inMask = imgApplyAffine(origInMask, compositeAffine, size=refImg.GetSize(), useNearest=False)

        if outDirPath != "":
            imgWrite(inImg, stepDirPath+"in.img")
            if(inMask): imgWrite(inMask, stepDirPath+"inMask.img")
            txtWriteList(compositeAffine, stepDirPath+"affine.txt")

    # Write final results
    if outDirPath != "":
        txtWrite(compositeAffine, outDirPath+"affine.txt")
        imgWrite(inImg, outDirPath+"out.img")
        imgWrite(imgChecker(inImg, refImg), outDirPath+"checker.img")
    
    return compositeAffine    

def imgMetamorphosis(inImg, refImg, alpha=0.02, beta=0.05, scale=1.0, iterations=1000, useNearest=False, useBias=False, useMI=False, verbose=True, inMask=None, refMask=None, outDirPath=""):
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
        command += " --cost 1 --sigma 1e-4 --epsilon 1e-3"

    if(inMask):
        inMaskPath = outDirPath + "inMask.img"
        imgWrite(inMask, inMaskPath)
        command += " --inmask " + inMaskPath

    if(refMask):
        refMaskPath = outDirPath + "refMask.img"
        imgWrite(refMask, refMaskPath)
        command += " --refmask " + refMaskPath

    os.system(command)
    #run(command, quiet=not(verbose))

    field = imgRead(fieldPath)
    invField = imgRead(invFieldPath)
    
    if useTempDir: shutil.rmtree(outDirPath)
    return (field, invField)


def imgMetamorphosisComposite(inImg, refImg, alphaList=0.02, betaList=0.05, scaleList=1.0, iterations=1000, useNearest=False, useBias=True, useMI=False, inMask=None, refMask=None, verbose=True, outDirPath=""):
    """
    Performs Metamorphic LDDMM between input and reference images
    """
    if outDirPath != "": outDirPath = dirMake(outDirPath)
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)
    """
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
    origInMask = inMask
    for step in range(numSteps):
        alpha = alphaList[step]
        beta = betaList[step]
        scale = scaleList[step]
        stepDirPath = outDirPath + "step" + str(step) + "/"
        if(verbose): print("Step {0}: alpha={1}, beta={2}, scale={3}".format(step,alpha, beta, scale))
        imgWrite(inMask, stepDirPath+"inMask.img")

        (field, invField) = imgMetamorphosis(inImg, refImg, 
                                             alpha, 
                                             beta, 
                                             scale, 
                                             iterations, 
                                             useNearest, 
                                             useBias, 
                                             useMI, 
                                             verbose,
                                             inMask=inMask,
                                             refMask=refMask,
                                             outDirPath=stepDirPath)

        if step == 0:
            compositeField = field
            compositeInvField = invField
        else:
            compositeField = fieldApplyField(field, compositeField)
            compositeInvField = fieldApplyField(compositeInvField, invField)

            if outDirPath != "":
                fieldPath = stepDirPath+"field.vtk"
                invFieldPath = stepDirPath+"invField.vtk"
                imgWrite(compositeInvField, invFieldPath)
                imgWrite(compositeField, fieldPath)

        inImg = imgApplyField(origInImg, compositeField, size=refImg.GetSize())
        if(inMask): inMask=imgApplyField(origInMask, compositeField, size=refImg.GetSize(), useNearest=True)

    # Write final results
    if outDirPath != "":
        imgWrite(compositeField, outDirPath+"field.vtk")
        imgWrite(compositeInvField, outDirPath+"invField.vtk")
        imgWrite(inImg, outDirPath+"out.img")
        imgWrite(imgChecker(inImg,refImg), outDirPath+"checker.img")
    
    return (compositeField, compositeInvField)

def imgRegistration(inImg, refImg, scale=1.0, affineScale=1.0, lddmmScaleList=[1.0], lddmmAlphaList=[0.02], iterations=1000, useMI=False, useNearest=True, inAffine=identityAffine, inMask=None, refMask=None, verbose=False, outDirPath=""):
    if outDirPath != "": outDirPath = dirMake(outDirPath)

    initialDirPath = outDirPath + "0_initial/"
    affineDirPath = outDirPath + "1_affine/"
    lddmmDirPath = outDirPath + "2_lddmm/"
    origInImg = inImg
    origRefImg = refImg

    # Resample and histogram match in and ref images
    refSpacing = refImg.GetSpacing()
    spacing = [x / scale for x in refSpacing]
    inImg = imgResample(inImg, spacing, useNearest=useNearest)
    refImg = imgResample(refImg, spacing, useNearest=useNearest)
    if(inMask): inMask = imgResample(inMask, spacing, useNearest=True)
    if(refMask): refMask = imgResample(refMask, spacing, useNearest=True)

    if not useMI: inImg = imgHM(inImg, refImg)
    initialInImg = inImg
    initialInMask = inMask
    initialRefMask = refMask
    if outDirPath != "":
        imgWrite(inImg, initialDirPath+"in.img")
        imgWrite(refImg, initialDirPath+"ref.img")

    if(verbose): print("Affine alignment")    
    affine = imgAffineComposite(inImg, refImg, scale=affineScale, useMI=useMI, iterations=iterations, inAffine=inAffine, verbose=verbose, inMask=inMask, refMask=refMask, outDirPath=affineDirPath)
    affineField = affineToField(affine, refImg.GetSize(), refImg.GetSpacing())
    invAffine = affineInverse(affine)
    invAffineField = affineToField(invAffine, inImg.GetSize(), inImg.GetSpacing())
    inImg = imgApplyField(initialInImg, affineField, size=refImg.GetSize())
    if(inMask): inMask = imgApplyField(initialInMask, affineField, size=refImg.GetSize(), useNearest=True)
    if(refMask): refMask = imgApplyField(initialRefMask, affineField, size=refImg.GetSize(), useNearest=True)

    if outDirPath != "":
        imgWrite(inImg, affineDirPath+"in.img")
        if(inMask): imgWrite(inMask, affineDirPath+"inMask.img")
        if(refMask): imgWrite(refMask, affineDirPath+"refMask.img")

    # Deformably align in and ref images
    if(verbose): print("Deformable alignment")
    (field, invField) = imgMetamorphosisComposite(inImg, refImg, alphaList=lddmmAlphaList, scaleList=lddmmScaleList, useBias=False, useMI=useMI, verbose=verbose, iterations=iterations, inMask=inMask, refMask=refMask, outDirPath=lddmmDirPath)
    field = fieldApplyField(field, affineField)
    invField = fieldApplyField(invAffineField, invField)
    inImg = imgApplyField(initialInImg, field, size=refImg.GetSize())

    if outDirPath != "":
        imgWrite(field, lddmmDirPath+"field.vtk")
        imgWrite(invField, lddmmDirPath+"invField.vtk")
        imgWrite(inImg, lddmmDirPath+"in.img")    
        imgWrite(imgChecker(inImg, refImg), lddmmDirPath+"checker.img")

    return (field, invField)
