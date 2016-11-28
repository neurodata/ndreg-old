import ndio.remote.neurodata as ND
import ndio.ramon as ndramon
import csv

def readMetadata(csvfilename):
    metadata = {}

    with open(csvfilename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None) # skip headers
        for row in csvreader:
            metadata[int(row[0])] = {
                'gaba': int(row[1]),
                'postgaba': int(row[2]),
                'display': int(row[3])
            }
    return metadata

def createRamonSynapse(ridx, kvpairs, resolution, confidence=1, author=''):
    robj = ndramon.RAMONSynapse(id=ridx, resolution=resolution, confidence=confidence, kvpairs=kvpairs, author=author, segments=[0])
    return robj

def main():
    token = 'myToken'
    channel = 'annotation'
    metadata = readMetadata('cleft_class.csv')

    nd = ND(hostname='synaptomes.neurodata.io/nd/')

    for ridx in metadata.keys():
        robj = createRamonSynapse(ridx, metadata[ridx], 0, confidence=1, author='Author Name')
        nd.post_ramon(token, channel, robj)
        print "Successfully posted ramon obj {}".format(ridx)

if __name__=='__main__':
    main()
