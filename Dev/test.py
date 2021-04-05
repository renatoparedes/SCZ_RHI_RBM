import numpy as np
import matplotlib.pyplot as plt
from GRBM import *
from scipy import stats
import string
import matplotlib as mpl

pCouple=0
NV=np.asarray([2500,150,30]) #number of visible units
NH=np.asarray([1500]) #number of hidden units

neuronInfo=[structtype(),structtype(),structtype()]
neuronInfo[0].span=np.asarray([1.8,1.8])# x,y span
neuronInfo[0].sm=np.asarray([.3,.3]) #x,y safety margin against edge effects (symmetric)
neuronInfo[0].center=np.asarray([0,.6]) #where the center of the neural population is in trunk centered coordinates;
neuronInfo[0].n=np.asarray([50,50]) #nx,ny
neuronInfo[0].tc=3 #tuning curve width (neurons)

neuronInfo[1].span=np.asarray([2,1.4]) #x,y span
neuronInfo[1].sm=np.asarray([.4,.4]) #x,y safety margin against edge effects (symmetric)
neuronInfo[1].center=np.asarray([0,.3]) #where the center of the neural population is in trunk centered coordinates;
neuronInfo[1].n=np.asarray([15,10]) #nx,ny
neuronInfo[1].tc=1 #tuning curve width (neurons)

neuronInfo[2].n=NV[2]

N_epochs=16 #160
N_batches=40 #400

N_vects=100
total_annealing=1
Cp=N_epochs/8
Slope=N_epochs/20
dE=1 #plot and save every dE epochs
eta=5e-6###this value of eta hasn't been optimized, but it's more or less OK
#etas=((eta/total_annealing)+eta*np.exp(-((np.arange(N_epochs))-Cp)/Slope))/(1+np.exp(-((np.arange(N_epochs))-Cp)/Slope))-np.linspace(0,eta/total_annealing,N_epochs)
etas=((eta/total_annealing)+eta*np.exp(-((np.arange(1,N_epochs+1))-Cp)/Slope))/(1+np.exp(-((np.arange(1,N_epochs+1))-Cp)/Slope))-np.linspace(0,eta/total_annealing,N_epochs)
#double check etas
tocOld=0

Names = structtype()
Names.V=['body centered','hand position','tactile'] #names of populations
Names.H=['hidden']
TrainParams = structtype()
TrainParams.N_epochs=N_epochs
TrainParams.N_batches=N_batches
TrainParams.N_vects=N_vects
TrainParams.TotalAnnealing=total_annealing
TrainParams.Etas=etas
g=grbm(NV,NH,['poiss','poiss','poiss'],['bern'],Names,TrainParams,neuronInfo) #population types (bernoulli or poisson)
res=[]

###############

stims=np.zeros((np.sum(g.NV),N_vects), dtype=np.float)
hiddenState=np.zeros((g.NH[0],N_vects))

for epoch in np.arange(N_epochs):
    re=0
    g.TrainParams.TrainingCompleted=False
    countb=0
    eta=etas[epoch]
    #get weights from the rmb objects, calculations for training are done
    #outside the object for efficiency
    w=g.allW()
    bh=g.allBh()
    bv=g.allBv()
        
    for batch in np.arange(N_batches): #a batch is made by N_vects individual stimuli
        #generate stimuli
        for i in np.arange(N_vects):
            #pH=np.multiply(g.NeuronInfo[1].min+(g.NeuronInfo[1].max-g.NeuronInfo[1].min),*np.random.rand(1,2))
            pH=np.multiply(g.NeuronInfo[1].min+(g.NeuronInfo[1].max-g.NeuronInfo[1].min),np.random.rand(2))
            if np.random.rand()>pCouple:
                #pBc=g.NeuronInfo[0].min-.15+ np.multiply((.3+g.NeuronInfo[0].max-g.NeuronInfo[0].min),*np.random.rand(1,2))
                pBc=g.NeuronInfo[0].min-.15+ np.multiply((.3+g.NeuronInfo[0].max-g.NeuronInfo[0].min),np.random.rand(2))
            else:
                #pBc=pH+.15*np.random.normal(1,2)
                pBc=pH+.15*np.random.normal(2)
            #gains=4+6*np.random.rand(3,1)
            gains=4+6*np.random.rand(3)
            Bc,H,T=stimgen(pBc,pH,g.NeuronInfo,gains)
            stims[:,i]=np.concatenate((Bc, H, T), axis=None)
        
        stims=np.random.poisson(lam=stims)
   
        #one-step contrastive divergence, done for all stimuli in a batch
        #at once for efficiency
        #up
        gv=w@stims+bh
        mu=1.0 / (1.0+np.exp(-gv)) 
        mu=mu.conj().T
        rr=np.random.rand(*np.shape(mu))
        h=1.0*(rr<mu).conj().T
            
        dW=h@stims.conj().T  
        dBv=np.sum(stims,axis=1,keepdims=True)
        dBh=np.sum(h,axis=1,keepdims=True)

        #down
        gv=w.conj().T@h+bv
        mu=np.exp(gv)
        v=np.random.poisson(lam=mu)
            
        #up
        gv=w@v+bh
        mu=1.0 / (1.0+np.exp(-gv)) 
        mu=mu.conj().T
        rr=np.random.rand(*np.shape(mu))
        h=1.0*(rr<mu).conj().T
            
        dW=dW-h@v.conj().T
        dBv=dBv-np.sum(v,axis=1,keepdims=True)
        dBh=dBh-np.sum(h,axis=1,keepdims=True)
            
        #update weights and biases
        w=w+eta*dW
        bv=bv+eta*dBv
        bh=bh+eta*dBh
            
    #end of epoch
    #update weights in the rbm object
    g.setW(w)
    g.setBv(bv)
    g.setBh(bh)
        
    if np.remainder(epoch,dE)==0: #plot learning parameters
        #save weights and reconstruction errors
        #res[epoch]=re/(dE*N_batches)
        res.append(re/(dE*N_batches))
        g.TrainParams.lastEpoch=epoch
        if epoch==N_epochs:
             g.TrainParams.TrainingCompleted=True
                    
        g.showPars(0)
        plt.figure(3)
        plt.plot(res)
        plt.xlabel('epoch')
        plt.title('log likelihood')

###########################################

## hand centered vs body centered tactile evoked activity
ntrials=10000
N_vects=100
ntrials=int(np.round(ntrials/N_vects)*N_vects)

nsteps=1
pBcs=np.zeros((ntrials,2))
pHs=np.zeros((ntrials,2))
pHcs=np.zeros((ntrials,2))
Ts=np.zeros(ntrials)
nGridPoints=17
stims=np.zeros((np.sum(g.NV),N_vects),dtype=float)
count=0
w=g.allW()
bh=g.allBh()
bv=g.allBv()
for i in np.arange( int(np.round(ntrials/N_vects)) ):
    if (i%10)==0:print(i)
    
    for v in np.arange(N_vects):
        count=count+1
        #pBc=g.NeuronInfo[0].min+np.multiply(g.NeuronInfo[0].max-g.NeuronInfo[0].min,*np.random.rand(1,2))
        pBc=g.NeuronInfo[0].min+np.multiply(g.NeuronInfo[0].max-g.NeuronInfo[0].min,np.random.rand(2))
        #pH=g.NeuronInfo[1].min+np.multiply(g.NeuronInfo[1].max-g.NeuronInfo[1].min,*np.random.rand(1,2))
        pH=g.NeuronInfo[1].min+np.multiply(g.NeuronInfo[1].max-g.NeuronInfo[1].min,np.random.rand(2))
        #gains=4+6*np.random.rand(3,1)
        gains=4+6*np.random.rand(3)
        Bc,H,T=stimgen(pBc,pH,g.NeuronInfo,gains)
        T=0*T
        stims[:,v]=np.concatenate((Bc, H, T), axis=None)
        pBcs[count-1,:]=pBc
        pHs[count-1,:]=pH
        pHcs[count-1,:]=pBc-pH
    
    v=stims
    for s in np.arange(nsteps):
        #noiseless up and down passes, since we only care about averages
        #for the plots
        #up
        gv=w@v+bh
        mu=1.0/(1.0+np.exp(-gv))
        mu=mu.conj().T
        h=mu.conj().T
        #down
        gv=w.conj().T@h+bv
        mu=np.exp(gv)
        v=mu
        v[:-g.NV[2],:]=stims[:-g.NV[2],:]
        
    Ts[(i)*N_vects:(i+1)*N_vects] = np.nanmean(v[-g.NV[2]:,:],axis=0) 

######################

def extents(f):
# https://gist.github.com/fasiha/eff0763ca25777ec849ffead370dc907
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

plt.figure(3)

fig, axs = plt.subplots(2, 2, figsize=(9, 9),dpi = 200)
ax1 = plt.subplot(221)
grid,xc,yc,_,_=bindata2d(pBcs[:,0],pBcs[:,1],Ts,nGridPoints)
im1=ax1.imshow(grid, aspect='auto', interpolation='none',
           extent=extents(xc) + extents(yc), origin='lower')
ax1.set_xlabel('Visual stimulus position (m)')
ax1.set_ylabel('Visual stimulus position (m)')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
fig.colorbar(im1, cax=cax,label='Tactile evoked activity')
ax1.text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax1.transAxes, 
                size=20, weight='bold')

ax2 = plt.subplot(222)
grid,xc,yc,_,_=bindata2d(pHs[:,0],pHs[:,1],Ts,nGridPoints)
im2=ax2.imshow(grid, aspect='auto', interpolation='none',
           extent=extents(xc) + extents(yc), origin='lower')
ax2.set_xlabel('Hand position (m)')
ax2.set_ylabel('Hand position (m)')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
fig.colorbar(im2, cax=cax,label='Tactile evoked activity')
ax2.text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax2.transAxes, 
                size=20, weight='bold')

ax3 = plt.subplot(223)
grid,xc,yc,stds,ns=bindata2d(pHcs[:,0],pHcs[:,1],Ts,nGridPoints)
im3=ax3.imshow(grid, aspect='auto', interpolation='none',
           extent=extents(xc) + extents(yc), origin='lower')
ax3.set_xlabel('Hand-centered position (m)')
ax3.set_ylabel('Hand-centered position (m)')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
fig.colorbar(im3, cax=cax,label='Tactile evoked activity')
ax3.text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax3.transAxes, 
                size=20, weight='bold')

ax4=plt.subplot(224)
xgrid,ygrid=np.meshgrid(xc,yc)
distances=np.concatenate(xgrid)**2+np.concatenate(ygrid)**2
xx,yy,_,sem = bindata(distances,grid.flatten(),numbins=20)
ax4.plot(xx,yy) # TODO complete: plot(xx,yy,'LineWidth',2)
ax4.set_xlabel('Distance from the hand (m)')
ax4.set_ylabel('Activity')

minInd=np.argmin(np.abs(xx))
maxInd=np.argmin(np.abs(xx-1))
ax4.set_xlim([xx[minInd],xx[maxInd]])
ax4.text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax4.transAxes, 
                size=20, weight='bold')

fig.subplots_adjust(wspace=0.75)      
fig.subplots_adjust(hspace=0.5)        