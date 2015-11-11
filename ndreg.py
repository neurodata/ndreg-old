#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO
# Add diffeomorphic registration code
# Add code to automatically generate brain masks
# Use ndio to fetch images from OCP insted of using ANALYZE files.

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import itk, sys, os, math, ctypes, tempfile, glob, subprocess, shutil, landmarks
from numpy import mat, array, dot, zeros
from copy import copy
from itertools import product

itk.ImageFileWriter
dimension = 3
PixelType = itk.Vector[itk.D, dimension]
FieldType = itk.Image[PixelType, dimension]
ComponentType = itk.Image[itk.D, dimension]
MapType = FieldType
identityAffine = [1,0,0,0,1,0,0,0,1,0,0,0]
identityDirection = itk.Matrix[itk.D,dimension,dimension]()
zeroOrigin = [0]*dimension
identityDirection.SetIdentity()    

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

def imgSize(path):
    """
    Returns size of image at path.
    """
    #ioDict = {'.img':itk.NiftiImageIO.New(), '.hdr':itk.NiftiImageIO.New(), '.nii':itk.NiftiImageIO.New(), '.vtk':itk.VTKImageIO.New() }
    #io = ioDict[os.path.splitext(path)[1].lower()]
    ext = os.path.splitext(path)[1].lower()

    if ext in ['.img','.hdr','.nii']:
        io = itk.NiftiImageIO.New()
    elif ext in ['.vtk']:
        io = itk.VTKImageIO.New()
                                             
    io.SetFileName(path)

    try:
        io.ReadImageInformation()
    except:
        print(sys.exc_info()[1])
        print("\nError: Could not read " + path)
        sys.exit(1)
    
    size = [int(io.GetDimensions(i)) for i in range(dimension)]
    return size

def imgSpacing(path):
    """
    Returns spacing of image at path.
    """
    #ioDict = {'.img':itk.NiftiImageIO.New(), '.hdr':itk.NiftiImageIO.New(), '.nii':itk.NiftiImageIO.New(), '.vtk':itk.VTKImageIO.New() }
    #io = ioDict[os.path.splitext(path)[1].lower()]
    ext = os.path.splitext(path)[1].lower()

    if ext in ['.img','.hdr','.nii']:
        io = itk.NiftiImageIO.New()
    elif ext in ['.vtk']:
        io = itk.VTKImageIO.New()
                                             
    io.SetFileName(path)

    try:
        io.ReadImageInformation()
    except:
        print(sys.exc_info()[1])
        print("\nError: Could not read " + path)
        sys.exit(1)
    
    spacing = [io.GetSpacing(i) for i in range(dimension)]
    return spacing

def imgRead(path, ImageType=None):
    """
    Read itk image from path.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".img",".hdr"]:
        io = itk.NiftiImageIO.New()
        io.SetLegacyAnalyze75Mode(True)
        #io = itk.AnalyzeImageIO.New()
        io.SetFileName(path)
        
        try:
            io.ReadImageInformation()
        except:
            print(sys.exc_info()[1])
            print("\nError: Could not read " + path)
            sys.exit(1)

        if ImageType is None: 
            typeDict = {'unsigned_char':itk.UC, 'short':itk.SS, 'float':itk.F, 'double':itk.D}
            PixelType = typeDict[io.GetComponentTypeAsString(io.GetComponentType())]
            ImageType = itk.Image[PixelType, dimension]
    
        ReaderType = itk.ImageFileReader[ImageType]
        reader = ReaderType.New(FileName=path, ImageIO=io)
    elif ext == ".vtk":
        ReaderType = itk.ImageFileReader[MapType]
        reader = ReaderType.New(FileName=path)
    else:
        ReaderType = itk.ImageFileReader[ImageType]
        reader = ReaderType.New(FileName=path)

    try:
        reader.Update()
    except:
        print(sys.exc_info()[1])
        print("\nError: Could not read " + path)
        sys.exit(1)
        
    return reader.GetOutput()

def imgWrite(img, path, ImageType=None):
    """
    Write itk image to path.
    """
    if ImageType is None:
        ImageType = type(img)
    else:
        CasterType = itk.CastImageFilter[type(img), ImageType]
        caster = CasterType.New(Input=img)
        caster.Update()
        img = caster.GetOutput()

    ext = os.path.splitext(path)[1].lower()
    if ext in [".img",".hdr"]:
        io = itk.NiftiImageIO.New()
        io.SetLegacyAnalyze75Mode(True)
        WriterType = itk.ImageFileWriter[ImageType]
        writer = WriterType.New(Input=img, FileName=path)
        writer.SetImageIO(io) 
    if ext == ".vtk":
        WriterType = itk.ImageFileWriter[ImageType]
        writer = WriterType.New(Input=img, FileName=path)

    try:
        writer.Update()
        #io = writer.GetImageIO()
        #print(io.GetComponentTypeAsString(io.GetComponentType()))
    except:
        print(sys.exc_info()[1])
        print("\nError: Could not write " + path)
        sys.exit(1)        
    
    if ext in [".img",".hdr"]:
        imgReformat(path, path)
    elif ext == ".vtk":
        mapReformat(path, path)
        
    return path

def imgClose(inPath, outPath, radius):
    """
    Morphologically closes an image with ball kernel of given radius.
    """
    ImageType = itk.Image[itk.SS, dimension]
    inImg = imgRead(inPath, ImageType)

    StructuringElementType = itk.FlatStructuringElement[dimension]
    ball = StructuringElementType.Ball(radius)

    CloserType = itk.GrayscaleMorphologicalClosingImageFilter[ImageType,ImageType,StructuringElementType]
    closer = CloserType.New(Input=inImg, Kernel=ball)
    closer.Update()

    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(closer.GetOutput(), outPath)

    return outPath

def imgOpen(inPath, outPath, radius):
    """
    Morphologically opens an image with ball kernel of given radius.
    """
    ImageType = itk.Image[itk.SS, dimension]
    inImg = imgRead(inPath, ImageType)

    StructuringElementType = itk.FlatStructuringElement[dimension]
    ball = StructuringElementType.Ball(radius)

    OpenerType = itk.GrayscaleMorphologicalOpeningImageFilter[ImageType,ImageType,StructuringElementType]
    opener = OpenerType.New(Input=inImg, Kernel=ball)
    opener.Update()

    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(opener.GetOutput(), outPath)

    return outPath

def imgDilate(inPath, outPath, radius):
    """
    Morphologically opens an image with ball kernel of given radius.
    """
    ImageType = itk.Image[itk.SS, dimension]
    inImg = imgRead(inPath, ImageType)

    StructuringElementType = itk.FlatStructuringElement[dimension]
    box = StructuringElementType.Box(radius)

    DilatorType = itk.GrayscaleDilateImageFilter[ImageType,ImageType,StructuringElementType]
    dilator = DilatorType.New(Input=inImg, Kernel=box)
    dilator.Update()

    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(dilator.GetOutput(), outPath)

    return outPath


def imgCrop(inPath, outPath, start, size):
    """
    Crops an image at inPath with giving bounding box start and size
    """
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    inImg = imgRead(inPath)
    ImageType = type(inImg)
    region = itk.ImageRegion[dimension](start, size)

    CropperType = itk.ExtractImageFilter[ImageType, ImageType]
    cropper = CropperType.New(Input=inImg, ExtractionRegion=region)
    cropper.SetDirectionCollapseToIdentity()
    cropper.Update()

    outImg=cropper.GetOutput()
    outRegion = itk.ImageRegion[dimension](size)
    outImg.SetRegions(outRegion)

    imgWrite(outImg, outPath)

    return outPath


def imgThreshold(inPath, outPath, threshold=0):
    """
    Thresholds image at inPath at given threshold and writes result to outPath.
    """
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    inImg = imgRead(inPath)
    ImageType = type(inImg)

    ThresholderType = itk.BinaryThresholdImageFilter[ImageType,ImageType]
    thresholder = ThresholderType.New(Input=inImg, UpperThreshold=threshold, InsideValue=0, OutsideValue=1)
    thresholder.Update()
    
    imgWrite(thresholder.GetOutput(), outPath)
    return outPath

def imgResample(inPath, outPath, spacing, size=[], useNearestNeighborInterpolation=False):
    """
    Resamples image at inPath to given spacing and writes result to outPath.
    """
    if len(spacing) != dimension: raise Exception("len(spacing) != " + str(dimension))
    if size != []:
        if len(size) != dimension: raise Exception("len(size) != " + str(dimension))

    inImg = imgRead(inPath)
    ImageType = type(inImg)
    inSpacing = inImg.GetSpacing()
    inSize = inImg.GetLargestPossibleRegion().GetSize()

    outSpacing = type(inSpacing)(spacing)
    outSize = type(inSize)()    
    for i in range(0, dimension):
        outSize[i] = int(math.ceil(ctypes.c_float( inSize[i]*(inSpacing[i]/outSpacing[i]) ).value))

    if useNearestNeighborInterpolation:
        InterpolatorType = itk.NearestNeighborInterpolateImageFunction[ImageType, itk.D]
    else:
        InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    interpolator = InterpolatorType.New()

    IdentityTransformType = itk.IdentityTransform[itk.D, dimension]
    identityTransform = IdentityTransformType.New()

    ResamplerType = itk.ResampleImageFilter[ImageType, ImageType]
    resampler = ResamplerType.New(Input=inImg, Transform=identityTransform, Interpolator=interpolator, Size=outSize, OutputSpacing=outSpacing)
    resampler.Update()

    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(resampler.GetOutput(), outPath)

    return outPath

def imgMask(inPath, maskPath, outPath):
    """
    Masks image at inPath with mask at maskPath and writes result to outPath.
    """
    inImg = imgRead(inPath)
    ImageType = type(inImg)

    maskImg = imgRead(maskPath)
    MaskType = type(maskImg)
    
    maskImg.CopyInformation(inImg)

    MaskerType = itk.MaskImageFilter[ImageType, MaskType, ImageType]
    masker = MaskerType.New(Input=inImg, MaskImage=maskImg)
    masker.Update()

    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(masker.GetOutput(), outPath)

    return outPath

def imgCheckerBoard(inPath, refPath, outPath, useHistogramMatching=True):

    # Read input image as floating point
    InternalImageType = itk.Image[itk.F, dimension]
    inImg = imgRead(inPath, InternalImageType)

    # Cast reference image to floating point
    refImg = imgRead(refPath)
    ImageType = type(refImg)

    CasterType = itk.CastImageFilter[ImageType, InternalImageType]
    caster = CasterType.New(Input=refImg)
    caster.Update()

    refImg = caster.GetOutput()

    if useHistogramMatching:
        HistogramMatcherType = itk.HistogramMatchingImageFilter[InternalImageType,InternalImageType]
        histogramMatcher = HistogramMatcherType.New(SourceImage=inImg, ReferenceImage=refImg, NumberOfMatchPoints=8, NumberOfHistogramLevels=64)
        histogramMatcher.Update()
        inImg = histogramMatcher.GetOutput()

    CheckerBoardType = itk.CheckerBoardImageFilter[InternalImageType]
    checkerBoard = CheckerBoardType.New(Input1=inImg, Input2=refImg)

    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(checkerBoard.GetOutput(), outPath, ImageType)
    
    return outPath


def imgHM(inPath, refPath, outPath, numMatchPoints=8, numBins=64):
    """
    Histogram matches input image to reference image and writes result to output image
    """
    # Read input image as floating point
    InternalImageType = itk.Image[itk.F, dimension]
    inImg = imgRead(inPath, InternalImageType)

    # Cast reference image to floating point
    refImg = imgRead(refPath)
    ImageType = type(refImg)

    CasterType = itk.CastImageFilter[ImageType, InternalImageType]
    caster = CasterType.New(Input=refImg)
    caster.Update()

    # Histogram match input image to reference image
    HistogramMatcherType = itk.HistogramMatchingImageFilter[InternalImageType,InternalImageType]
    histogramMatcher = HistogramMatcherType.New(SourceImage=inImg, ReferenceImage=caster.GetOutput(), NumberOfMatchPoints=numMatchPoints, NumberOfHistogramLevels=numBins)
    histogramMatcher.Update()

    # Write output image
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(histogramMatcher.GetOutput(), outPath, ImageType)
    return outPath


def imgApplyAffine(inPath, outPath, affine, useNearestNeighborInterpolation=False, size=[], spacing=[]):
    """ I_{in} \circ \phi_{map} """
    # Read input image
    inImg = imgRead(inPath)
    ImageType = type(inImg)

    # Save orignal parameters
    inOrigin = list(inImg.GetOrigin())
    inDirection = itkToNumpyMat(inImg.GetDirection())

    # Set internal parameters
    internalOrigin = [0.0]*dimension
    internalDirection = itk.Matrix[itk.D,dimension,dimension]()
    internalDirection.SetIdentity()

    # Use internal parameters
    inImg.SetOrigin(internalOrigin)
    inImg.SetDirection(internalDirection)

    # Set Size
    if size == []:
        size = inImg.GetLargestPossibleRegion().GetSize()
    else:
        if (not(type(size)) is list) or (len(size) != dimension): raise Exception("size must be a list of length {0}.".format(dimension))

    # Set Spacing
    if spacing == []:
        spacing = inImg.GetSpacing()
    else:
        if (not(type(spacing)) is list) or (len(spacing) != dimension): raise Exception("spacing must be a list of length {0}.".format(dimension))

    # Create interpolator
    if useNearestNeighborInterpolation:
        InterpolatorType = itk.NearestNeighborInterpolateImageFunction[ImageType, itk.D]
    else:
        InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    interpolator = InterpolatorType.New()

    # Create transform 
    TransformType = itk.AffineTransform[itk.D,dimension]
    transform = TransformType.New()
    numParameters = transform.GetNumberOfParameters()

    # Set affine parameters
    if (not(type(affine)) is list) or (len(affine) != numParameters): raise Exception("affine must be a list of length {0}.".format(numParameters))
    parameters = transform.GetParameters()    
    for i in range(numParameters): parameters.SetElement(i,affine[i])
    transform.SetParameters(parameters)    

    # Apply affine transform to image
    ResamplerType = itk.ResampleImageFilter[ImageType, ImageType]
    resampler = ResamplerType.New(Input=inImg, Transform=transform, Interpolator=interpolator, Size=size, OutputSpacing=spacing, OutputOrigin=internalOrigin, OutputDirection=internalDirection)
    resampler.Update()

    # Restore input parameters
    outImg = resampler.GetOutput()
    outImg.SetOrigin(inOrigin)
    outImg.SetDirection(numpyToItkMat(inDirection))
    
    # Write
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(outImg, outPath)

    return outPath

def imgApplyMap(inPath, mapPath, outPath, useNearestNeighborInterpolation=False, size=[]):
    """ I_{in} \circ \phi_{map} """
    # Read input image
    inImg = imgRead(inPath)
    ImageType = type(inImg)

    # Save input parameters
    inOrigin = list(inImg.GetOrigin())
    inSpacing = list(inImg.GetSpacing())
    inDirection = itkToNumpyMat(inImg.GetDirection())

    # Set internal parameters
    internalOrigin = [0.0]*dimension
    internalSpacing = [1.0]*dimension
    internalDirection = itk.Matrix[itk.D,dimension,dimension]()
    internalDirection.SetIdentity()
    
    # Use internal parameters
    inImg.SetOrigin(internalOrigin)
    inImg.SetSpacing(internalSpacing)
    inImg.SetDirection(internalDirection)

    # Read displacement
    displacement = mapReadAsDisplacement(mapPath)
    DisplacementType = type(displacement)
    
    # Use internal parameters
    displacement.SetOrigin(internalOrigin)
    displacement.SetSpacing(internalSpacing)
    displacement.SetDirection(internalDirection)

    # Set Size
    if size == []:
        size = inImg.GetLargestPossibleRegion().GetSize()
    else:
        if (not(type(size)) is list) or (len(size) != dimension): raise Exception("size must be a list of length {0}.".format(dimension))

    # Create interpolator
    if useNearestNeighborInterpolation:
        InterpolatorType = itk.NearestNeighborInterpolateImageFunction[ImageType, itk.D]
    else:
        InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    interpolator = InterpolatorType.New()
    
    # Create Transform
    TransformType = itk.DisplacementFieldTransform[itk.D, dimension]
    transform = TransformType.New(DisplacementField=displacement)

    # Apply map to image image    
    ResamplerType = itk.ResampleImageFilter[ImageType, ImageType]
    resampler = ResamplerType.New(Input=inImg, Transform=transform, Interpolator=interpolator, Size=size, OutputOrigin=internalOrigin, OutputSpacing=internalSpacing, OutputDirection=internalDirection)
    resampler.Update()

    # Restore original parameters
    outImg = resampler.GetOutput()
    outImg.SetOrigin(inOrigin)
    outImg.SetSpacing(inSpacing)
    outImg.SetDirection(numpyToItkMat(inDirection))

    # Write 
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    imgWrite(outImg, outPath)

    return outPath

def mapApplyMap(inPath, mapPath, outPath):
    """ \phi_{in} \circ \phi_{map} """
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    inDisplacement = mapReadAsDisplacement(inPath) 

    map = imgRead(mapPath)
    map.SetSpacing([1.0]*dimension)
    map.SetOrigin([0.0]*dimension)
    map.SetDirection(identityDirection)

    ExtractorType = itk.VectorIndexSelectionCastImageFilter[MapType, ComponentType]
    ComposerType = itk.ComposeImageFilter[ComponentType, MapType]
    composer = ComposerType.New()

    for i in range(dimension):
        extractor = ExtractorType.New(Input=map, Index=i)
        extractor.Update()

        WarperType = itk.WarpImageFilter[ComponentType, ComponentType, FieldType]
        warper = WarperType.New(Input=extractor.GetOutput(), DisplacementField=inDisplacement, OutputParametersFromImage=extractor.GetOutput())
        warper.Update()

        composer.PushBackInput(warper.GetOutput())

    composer.Update()
    imgWrite(composer.GetOutput(), outPath)

    return outPath


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

def mapToComponents(inPath, outPathList):
    inMap = imgRead(inPath)

    if (not(type(outPathList)) is list) or (len(outPathList) != dimension):
        raise Exception("outPathLit must be list of length {0}.".format(dimension))
    
    ExtractorType = itk.VectorIndexSelectionCastImageFilter[MapType, ComponentType]
    for i in range(dimension):
        extractor = ExtractorType.New(Input=inMap, Index=i)
        extractor.Update()

        outPath = outPathList[i]
        (outPath, outDirPath) = getOutPaths(inPath, outPath)
        imgWrite(extractor.GetOutput(), outPath)

    return outPathList

def mapReadAsDisplacement(inPath):
    inMap = imgRead(inPath)
    inMap.SetSpacing([1.0]*dimension)
    inMap.SetOrigin([0.0]*dimension)
    inMap.SetDirection(identityDirection)

    ImageSourceType = itk.PhysicalPointImageSource[MapType]
    imageSource = ImageSourceType.New(Size=inMap.GetLargestPossibleRegion().GetSize())
    imageSource.Update()

    identityMap = imageSource.GetOutput()
    identityMap.SetOrigin(inMap.GetOrigin());
    identityMap.SetSpacing(inMap.GetSpacing())
    identityMap.SetDirection(inMap.GetDirection())
    
    SubtractorType = itk.SubtractImageFilter[ComponentType, ComponentType, ComponentType]
    ExtractorType = itk.VectorIndexSelectionCastImageFilter[MapType, ComponentType]
    ComposerType = itk.ComposeImageFilter[ComponentType, MapType]
    composer = ComposerType.New()

    for i in range(dimension):
        inExtractor = ExtractorType.New(Input=inMap, Index=i)
        inExtractor.Update()

        identityExtractor = ExtractorType.New(Input=identityMap, Index=i)
        identityExtractor.Update()
    
        subtractor = SubtractorType.New(Input1=inExtractor.GetOutput(), Input2=identityExtractor.GetOutput())
        subtractor.Update()

        composer.PushBackInput(subtractor.GetOutput())

    composer.Update()

    return composer.GetOutput()

def mapCreateIdentity(size):
    ImageSourceType = itk.PhysicalPointImageSource[MapType]
    imageSource = ImageSourceType.New(Size=size)
    imageSource.Update()
    return imageSource.GetOutput()


def displacementWriteAsMap(displacement, outPath):
    size = displacement.GetLargestPossibleRegion().GetSize()

    ImageSourceType = itk.PhysicalPointImageSource[MapType]
    imageSource = ImageSourceType.New(Size=size)
    imageSource.Update()
    identityMap = imageSource.GetOutput()
    displacement.CopyInformation(identityMap)

    AdderType = itk.AddImageFilter[ComponentType, ComponentType, ComponentType]
    ExtractorType = itk.VectorIndexSelectionCastImageFilter[MapType, ComponentType]
    ComposerType = itk.ComposeImageFilter[ComponentType, MapType]
    composer = ComposerType.New()
    for i in range(dimension):
        displacementExtractor = ExtractorType.New(Input=displacement, Index=i)
        displacementExtractor.Update()

        identityExtractor = ExtractorType.New(Input=identityMap, Index=i)
        identityExtractor.Update()
    
        adder = AdderType.New(Input1=displacementExtractor.GetOutput(), Input2=identityExtractor.GetOutput())
        adder.Update()

        composer.PushBackInput(adder.GetOutput())
    composer.Update()

    return imgWrite(composer.GetOutput(), outPath)


def lmkApplyAffine(inPath, outPath, affine, spacing=[1.0,1.0,1.0]):
    if (not(type(spacing)) is list) or (len(spacing) != 3): raise Exception("spacing must be a list of length 3.")
    (outPath, outDirPath) = getOutPaths(inPath, outPath)

    # Apply spacing to affine
    tmpAffine = affine[:]
    for i in range(3): tmpAffine[9+i] =  tmpAffine[9+i] / spacing[i]
    
    # Apply affine to landmarks
    lmk = landmarks.landmarks(inPath)
    transformedLmk = lmk.Affine(tmpAffine)
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

def itkToNumpyMat(itkMat):
    numRows = itkMat.GetVnlMatrix().rows()
    numCols = itkMat.GetVnlMatrix().cols()
    numpyMat = zeros([numRows, numCols])
    for row in range(numRows):
        for col in range(numCols):
            numpyMat[row][col] = itkMat(row,col)

    return numpyMat
    
def numpyToItkMat(numpyMat):
    (numRows, numCols) = numpyMat.shape
    itkMat = itk.Matrix[itk.D,numRows, numCols]()
    for row in range(numRows):
        for col in range(numCols):
            itkMat.GetVnlMatrix().put(row,col, numpyMat[row,col])

    return itkMat           

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

def imgAffine(inPath, refPath, outPath, useNearestNeighborInterpolation=False, useRigid=False):
    InternalImageType = itk.Image[itk.D, dimension]
    inImg = imgRead(inPath, InternalImageType)
    inImg.SetOrigin(zeroOrigin)
    inImg.SetDirection(identityDirection)

    refImg = imgRead(refPath, InternalImageType)
    refImg.SetOrigin(zeroOrigin)
    refImg.SetDirection(identityDirection)

    # Create interpolator
    if useNearestNeighborInterpolation:
        interpolator = itk.NearestNeighborInterpolateImageFunction[InternalImageType, itk.D].New()
    else:
        interpolator = itk.LinearInterpolateImageFunction[InternalImageType, itk.D].New()

    # Create Transform
    if useRigid:
        transform = itk.Similarity3DTransform[itk.D].New() # Transform for rotation, translation and scaling
    else:
        transform = itk.AffineTransform[itk.D, dimension].New()

    # Set initial transform parameters
    M = mat(identityAffine[0:9]).reshape([3,3])
    transform.SetMatrix(numpyToItkMat(M))
    transform.SetTranslation(identityAffine[9:])
    initialParameters = transform.GetParameters()

    # Create optimizer and metrix
    optimizer = itk.GradientDescentOptimizer.New(LearningRate=0.00001, NumberOfIterations=100, Maximize=False)
    metric = itk.MeanSquaresImageToImageMetric[InternalImageType, InternalImageType].New(NumberOfSpatialSamples=1000)

    # Do registration
    registration = itk.ImageRegistrationMethod[InternalImageType, InternalImageType].New(Interpolator=interpolator, Transform=transform, InitialTransformParameters=initialParameters, Optimizer=optimizer, Metric=metric, FixedImage=refImg, MovingImage=inImg, FixedImageRegion=refImg.GetLargestPossibleRegion())
    registration.Update()

    # Copy final transform parameters into list
    #affine = [registration.GetLastTransformParameters().GetElement(i) for i in range(12)]
    affine = itkToNumpyMat(transform.GetMatrix()).reshape([1,9]).tolist()[0] + list(transform.GetTranslation())
    
    # Write final transform parameters to text file
    (outPath, outDirPath) = getOutPaths(inPath, outPath)
    txtWriteParameterList(affine,outDirPath+"affine.txt")

    # Apply final transform parameters to input image
    refSize = imgSize(refPath)
    imgApplyAffine(inPath, outPath, affine, useNearestNeighborInterpolation, refSize)

    return affine

def registration(inImgPath, refImgPath, outDirPath, useNearestNeighborInterpolation=False):
    (outPath, outDirPath) = getOutPaths(inImgPath, outDirPath)

    inImgFileName = os.path.basename(inImgPath)
    refImgFileName = os.path.basename(refImgPath)

    print("...Getting Original Data")
    origDirPath = outDirPath + "0_orig/"

    # Get original data
    imgCopy(inImgPath, origDirPath)
    imgCopy(refImgPath, origDirPath)

    # Get size and spacing of images
    refSize = imgSize(origDirPath+refImgFileName)
    refSpacing = imgSpacing(origDirPath+refImgFileName)
    inSize = imgSize(origDirPath+inImgFileName)
    inSpacing = imgSpacing(origDirPath+inImgFileName)

    # Do rigid registration
    print("...Rigid Registration")
    rigidDirPath = outDirPath + "1_rigid/"
    rigid = imgAffine(origDirPath+inImgFileName, origDirPath+refImgFileName, rigidDirPath, useNearestNeighborInterpolation, True)
    imgCheckerBoard(rigidDirPath+inImgFileName, origDirPath+refImgFileName, rigidDirPath+"checkerboard.img")
    
    # Do affine registration
    print("...Affine Registration")
    affineDirPath = outDirPath + "2_affine/"
    affine = imgAffine(rigidDirPath+inImgFileName, origDirPath+refImgFileName, affineDirPath, useNearestNeighborInterpolation, False) # Do affine registration
    combinedAffine = affineApplyAffine(affine, rigid)                         # Conbine rigid and affine transforms
    txtWriteParameterList(combinedAffine, affineDirPath+"combinedAffine.txt") # Write combined transform to file

    mapPath = affineDirPath+"map.vtk"
    affineToMap(combinedAffine, inSize, mapPath, inSpacing)                         # Convert combined affine to map
    invMapPath = affineDirPath+"mapInv.vtk"
    affineToMap(affineInverse(combinedAffine), refSize, invMapPath, refSpacing)     # Convert inverse combined affine to map

    #imgApplyAffine(origDirPath+inImgFileName, affineDirPath+"out.img", combinedAffine, useNearestNeighborInterpolation, refSize) # Apply combined transform to orignal input image
    imgApplyMap(origDirPath+inImgFileName, mapPath, affineDirPath, useNearestNeighborInterpolation) # Apply combined transform to original  input image
    imgCheckerBoard(affineDirPath+inImgFileName, origDirPath+refImgFileName, affineDirPath+"checkerboard.img")
    
    imgCopy(affineDirPath+inImgFileName, outDirPath)
    shutil.copy(mapPath, outDirPath)
    shutil.copy(invMapPath, outDirPath)

    # TODO: Do diffeomorphic registration

