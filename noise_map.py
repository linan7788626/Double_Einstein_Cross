#!/home/wtluo/anaconda/bin/python2.7


import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
from optparse import OptionParser

parser=OptionParser()
parser.add_option("--ImSize",dest="ImSize",default=128,
                    help="Image Size",metavar="value",
                    type="int")
parser.add_option("--NoiseVar",dest="NoiseVar",default=0.0,
                    help="Noise variance",metavar="value",
                    type="float")
parser.add_option("--NoiseType",dest="NoiseType",default=None,
                    help="Gaussian noise",metavar="value",
                    type="string")
parser.add_option("--Verbose",dest="Verbose",default=True,
                    help="Gaussian noise",metavar="value",
                    type="string")

(o,args)=parser.parse_args()
nax     =o.ImSize
nstd    =np.sqrt(o.NoiseVar)
noise   =np.zeros((nax,nax))

if o.NoiseType=='Poisson':
	if o.Verbose:
		print '## Hey! You choose Poisson noise.'
	noise=np.random.poisson(nstd,(nax,nax))-nstd
if o.NoiseType=='Gaussian':
	if o.Verbose:
		print '## Hey! You choose Gaussian noise.'
	noise=nstd*np.random.normal(0.0,1.0,(nax,nax))	

plt.imshow(noise)
plt.show()
