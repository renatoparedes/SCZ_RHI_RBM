import numpy as np
import matplotlib.pyplot as plt
from GRBM_RHI import *
from scipy import stats

pVisible=0.75
pDecoupled=0.25
NV=np.asarray([2500,150,1650,30]) #number of visible units
NH=np.asarray([1500]) #number of hidden units
neuronInfo=[structtype(),structtype(),structtype(),structtype()]

# visual stimulus
neuronInfo[0].span=np.asarray([1.8,1.8])# x,y span
neuronInfo[0].sm=np.asarray([.3,.3]) #x,y safety margin against edge effects (symmetric)
neuronInfo[0].center=np.asarray([0,.6]) #where the center of the neural population is in trunk centered coordinates;
neuronInfo[0].n=np.asarray([50,50]) #nx,ny
neuronInfo[0].tc=3 #tuning curve width (neurons)

# hand position proprioceptive
neuronInfo[1].span=np.asarray([2,1.4]) #x,y span
neuronInfo[1].sm=np.asarray([.4,.4]) #x,y safety margin against edge effects (symmetric)
neuronInfo[1].center=np.asarray([0,.3]) #where the center of the neural population is in trunk centered coordinates;
neuronInfo[1].n=np.asarray([15,10]) #nx,ny
neuronInfo[1].tc=1 #tuning curve width (neurons)

# hand position visual
neuronInfo[2].span=np.asarray([1.8,1.2]) #x,y span
neuronInfo[2].sm=np.asarray([.3,.3]) #x,y safety margin against edge effects (symmetric)
neuronInfo[2].center=np.asarray([0,.3]) #where the center of the neural population is in trunk centered coordinates;
neuronInfo[2].n=np.asarray([50,33]) #nx,ny
neuronInfo[2].tc=3 #tuning curve width (neurons)

# tactile
neuronInfo[3].n=NV[3]

N_epochs=16 #20 #160
N_batches=40 #50 #400

N_vects=100
total_annealing=1
Cp=N_epochs/8
Slope=N_epochs/20
dE=1 #plot and save every dE epochs
eta=5e-6###this value of eta hasn't been optimized, but it's more or less OK
etas=((eta/total_annealing)+eta*np.exp(-((np.arange(1,N_epochs+1))-Cp)/Slope))/(1+np.exp(-((np.arange(1,N_epochs+1))-Cp)/Slope))-np.linspace(0,eta/total_annealing,N_epochs)
tocOld=0

Names = structtype()
Names.V=['visual stimulus','hand proprioception','hand vision','tactile'] #names of populations
Names.H=['hidden']
TrainParams = structtype()
TrainParams.N_epochs=N_epochs
TrainParams.N_batches=N_batches
TrainParams.N_vects=N_vects
TrainParams.TotalAnnealing=total_annealing
TrainParams.Etas=etas
g=grbm(NV,NH,['poiss','poiss','poiss','poiss'],['bern'],Names,TrainParams,neuronInfo) #population types (bernoulli or poisson)
res=[]