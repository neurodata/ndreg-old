#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
from ndreg import *
import ndio.ramon as ndramon
import ndio.remote.neurodata as neurodata
"""
Here we show how to RAMONify Allen Reference Atlas data.
First we download annotation ontology from Allen Brain Atlas API.
It returns a JSON tree in which larger parent structures are divided into smaller children regions.
For example the "corpus callosum" parent is has children "corpus callosum, anterior forceps", "genu of corpus callosum", "corpus callosum, body", etc
"""

url = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
jsonRaw = requests.get(url).content
jsonDict = json.loads(jsonRaw)

"""
Next we collect the names and ids of all of the regions.
Since our json data is a tree we can walk through it in arecursive manner.
Thus starting from the root...
"""
root = jsonDict['msg'][0]
"""
...we define a recursive function ...
"""
#leafList = []
def getChildrenNames(parent, childrenNames={}):
    #if len(parent['children']) == 0:
    #    leafList.append(parent['id'])

    for childIndex in range(len(parent['children'])):
        child = parent['children'][childIndex]
        childrenNames[child['id']] = child['name']

        childrenNames = getChildrenNames(child, childrenNames)
    return childrenNames

"""
... and collect all of the region names in a dictionary with the "id" field as keys.
"""


regionDict = getChildrenNames(root)
#print(leafList)
#for key in regionDict.keys():
#    print('{0}, "{1}"'.format(key, regionDict[key]))
#print(regionDict)
#sys.exit()

"""
Next we RAMONify the data
"""
token = "ara3_to_AutA"
channel = "annotation_draft"
nd = neurodata(hostname='synaptomes.neurodata.io/nd/')

for regionId in regionDict.keys():
    regionName = regionDict[regionId]

    kvpairs = {'name': regionName}
    ramonObj = ndramon.RAMONGeneric(id=regionId, resolution=0, kvpairs=kvpairs)
    try:
        nd.post_ramon(token, channel, ramonObj)
        print "Successfully posted ramon obj {0} for {1}".format(regionId, regionName)
    except:
        print "Failed to post ramon obj {0} for {1}".format(regionId, regionName)
