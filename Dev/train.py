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
            #np.random.seed(255)
            pH= g.NeuronInfo[1].min+np.multiply((g.NeuronInfo[1].max-g.NeuronInfo[1].min),np.random.rand(2))
            #np.random.seed(255)
            if np.random.rand()>pCouple:
                #np.random.seed(255)
                pBc=g.NeuronInfo[0].min-.15+ np.multiply((.3+g.NeuronInfo[0].max-g.NeuronInfo[0].min),np.random.rand(2))
            else:
                #np.random.seed(255)
                pBc=pH+.15*norm.ppf(np.random.rand(2))
            #np.random.seed(255)
            gains=4+6*np.random.rand(3)
            Bc,H,T=stimgen(pBc,pH,g.NeuronInfo,gains)
            stims[:,i]=np.concatenate((Bc, H, T), axis=None)
        
        #np.random.seed(255)
        stims=np.random.poisson(lam=stims)
        # https://hpaulkeeler.com/simulating-poisson-random-variables-survey-methods/ (not exact equivalent to Matlab)
   

        #one-step contrastive divergence, done for all stimuli in a batch
        #at once for efficiency
        #up
        gv=w@stims+bh
        mu=1.0 / (1.0+np.exp(-gv)) 
        mu=mu.conj().T
        #np.random.seed(255)
        rr=np.random.rand(np.shape(mu)[1],np.shape(mu)[0]).T
        h=1.0*(rr<mu).conj().T

        dW=h@stims.conj().T  
        dBv=np.sum(stims,axis=1,keepdims=True)
        dBh=np.sum(h,axis=1,keepdims=True)

        #down
        gv=w.conj().T@h+bv
        mu=np.exp(gv)
        #np.random.seed(255)
        v=np.random.poisson(lam=mu)
            
        #up
        gv=w@v+bh
        mu= 1.0 / (1.0+np.exp(-gv)) 
        mu=mu.conj().T
        #np.random.seed(255)
        rr=np.random.rand(np.shape(mu)[1],np.shape(mu)[0]).T
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