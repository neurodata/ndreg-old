#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ndreg import *

inPath = "inImage.img"
refPath = "refImage.img"
outDirPath = "./"
registration(inPath, refPath, outDirPath, True)
