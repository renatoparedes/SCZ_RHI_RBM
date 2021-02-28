import numpy as np
import matplotlib.pyplot as plt
# import line_profiler
# profile = line_profiler.LineProfiler()

from GRBM import *

def main():
    pCouple=0
    NV=[2500,150,30] #number of visible units
    NH=[1500] #number of hidden units

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
    etas=((eta/total_annealing)+eta*np.exp(-((np.arange(N_epochs))-Cp)/Slope))/(1+np.exp(-((np.arange(N_epochs))-Cp)/Slope))-np.linspace(0,eta/total_annealing,N_epochs)
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

    w = g.allW()
    v = g.allV()
    h,mu = g.fastUp(v)
    v,mu = g.fastDown(h)

    gains=4+6*np.random.rand(3,1)
    Bc,H,T=stimgen(0.7,0.7,g.NeuronInfo,gains)
    mystim=[Bc, H, T]




    # actual training
    stims=np.zeros((np.sum(g.NV),N_vects), dtype=np.float)
    hiddenState=np.zeros((g.NH[0],N_vects))

    for epoch in range(N_epochs):
        re=0
        g.TrainParams.TrainingCompleted=False
        countb=0
        eta=etas[epoch]
        #get weights from the rmb objects, calculations for training are done
        #outside the object for efficiency
        w=g.allW()
        bh=g.allBh()
        bv=g.allBv()
        
        for batch in range(N_batches): #a batch is made by N_vects individual stimuli
            #generate stimuli
            for i in range(N_vects):
                pH= np.multiply(g.NeuronInfo[1].min+(g.NeuronInfo[1].max-g.NeuronInfo[1].min),*np.random.rand(1,2))
                #double check H and stimgen function for H
                if np.random.rand()>pCouple:
                    pBc=g.NeuronInfo[0].min-.15+ np.multiply((.3+g.NeuronInfo[0].max-g.NeuronInfo[0].min),*np.random.rand(1,2))
                else:
                    pBc=pH+.15*np.random.normal(1,2)
                gains=4+6*np.random.rand(3,1)
                Bc,H,T=stimgen(pBc,pH,g.NeuronInfo,gains)
                # stims[:,i]=np.concatenate((np.hstack(Bc),np.hstack(H),np.hstack(T)))
                stims[:,i] = np.concatenate((Bc, H, T), axis=None) # TODO faster concatenate
            stims=np.random.poisson(stims)
            
            #one-step contrastive divergence, done for all stimuli in a batch
            #at once for efficiency
            #up
            gv=w@stims+bh
            mu=1.0 / (1.0+np.exp(-gv)) # TODO float
            mu=mu.T
            rr=np.random.rand(*np.shape(mu))
            h=1.0*(rr<mu).T # TODO keep as float by changing to 1.0
            
            dW=h@stims.T  # TODO keep everything as float
            dBv=np.sum(stims,1)
            dBh=np.sum(h,1)
            
            #down
            gv=w.T@h+bv
            mu=np.exp(gv)
            v=np.random.poisson(mu)
            
            #up
            gv=w@v+bh
            mu=1.0 / (1.0+np.exp(-gv)) # TODO float
            mu=mu.T
            rr=np.random.rand(*np.shape(mu))
            h=1.0*(rr<mu).T # TODO  float
            
            dW=dW-h@v.T # TODO float
            dBv=dBv-np.sum(v,1)
            dBh=dBh-np.sum(h,1)
            
            #update weights and biases
            w=w+eta*dW
            bv=bv+eta*dBv.reshape((2680,1))
            bh=bh+eta*dBh.reshape((1500,1))
            
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
            
    #         tt=tic
    #         mFile.WeightHistory(epoch/dE,1)=struct('W',g.allW,'Bh',g.allBh,'Bv',g.allBv,'Res',res(epoch),'Epoch',epoch);
    #         mFile.res=res;
    #         save(SaveNameTmp,'g');
    #         toc(tt)
            
    #         tCurr=toc;
    #         disp(['V ' Ver ', epoch ' num2str(epoch)])
    #         toc
    #         disp([num2str(N_batches*N_vects*dE/((tCurr-tocOld))) ' vectors per second'])
    #         tocOld=tCurr;
            
            g.showPars(0)
            
            plt.figure(3)
            plt.plot(res)
            plt.xlabel('epoch')
            plt.title('log likelihood')


main()