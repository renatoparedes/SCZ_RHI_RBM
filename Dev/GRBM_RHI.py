import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
from scipy import stats
from scipy.stats import norm

class structtype():
    pass
# Matlab to Python conversion> https://stackoverflow.com/questions/11637045/complex-matlab-like-data-structure-in-python-numpy-scipy
# Python dictionaries> https://realpython.com/iterate-through-dictionary-python/

# class of RMB object
class grbm: # handle class, beware
#         V; #visible units subpopulation struct (S state,B biases, T type ['poiss' or 'bern'])
#         H; #hidden units subpopulation struct (S state,B biases, T type ['poiss' or 'bern'])
#         W; #weight matrixes struct. Element ph,pv is hidden population ph acting on visible population pv and viceversa
#         NV; #vector containing numbers of units for each visible population
#         NH; #vector containing numbers of units for each hidden population
#         Npv; #number of visible subpopulations
#         Nph; #number of hidden subpopulations
#         #state and bias vectors are COLUMN vectors
#         Names;#names of subpopulations
#         TrainParams;#training parameters
#         NeuronInfo;#neuron info
#         WUp;#weights arranged for the 'up' step
#         WDown;#weights arranged for the 'down' step
#         VUp;#all visible neurons
#         VDown;#all hidden neurons
#         persistentChain;

        #constructor
        def __init__(self,nv,nh,tv,th,names,trainparams,neuroninfo): #done
            self.V=[structtype(),structtype(),structtype(),structtype()]
            self.H=[structtype(),structtype(),structtype(),structtype()]
            self.W=[[structtype(),structtype(),structtype(),structtype()]]
            self.NV=nv
            self.NH=nh
            self.Npv=np.size(self.NV)
            self.Nph=np.size(self.NH)
            self.Names=names
            self.TrainParams=trainparams
            self.NeuronInfo=neuroninfo
            self.WUp=[structtype(),structtype(),structtype(),structtype()]
            self.WDown=[structtype(),structtype(),structtype(),structtype()]
            
            for pv in np.arange(self.Npv):
                self.V[pv].S=np.zeros((self.NV[pv],1))
                self.V[pv].B=np.zeros((self.NV[pv],1))
                self.V[pv].T=tv[pv]
            
            for ph in np.arange(self.Nph):
                self.H[ph].S=np.zeros((self.NH[ph],1))
                self.H[ph].B=np.zeros((self.NH[ph],1))
                self.H[ph].T=th[ph]
            
            for pv in np.arange(self.Npv):
                for ph in np.arange(self.Nph):
                    #np.random.seed(255)
                    self.W[ph][pv].W = norm.ppf(np.random.rand(self.NV[pv],self.NH[ph]).T) / np.sum(self.NH) # normalization has to be checked! BEWARE!s
            for ph in np.arange(self.Nph):
                #np.random.seed(255)
                ww=0.01*norm.ppf(np.random.rand(np.sum(self.NV),self.NH[ph]).T)
                vv=np.zeros((np.sum(self.NV),1))
                inds=np.insert(np.cumsum(self.NV),0,0)
                for pv in np.arange(self.Npv):
                    currInds=np.arange(inds[pv],inds[pv+1])
                    ww[:,currInds]=self.W[ph][pv].W #matrix acting on the subpopulation ph
                    vv[currInds]=self.V[pv].S  #global visible activity
                    
                self.WUp[ph].W=ww
                self.VUp=vv
            
            for pv in np.arange(self.Npv):
                #np.random.seed(255)
                ww=0.01*norm.ppf(np.random.rand(self.NV[pv],np.sum(self.NH)).T)
                hh=np.zeros((np.sum(self.NH),1))
                inds=np.insert(np.cumsum(self.NH),0,0)
                for ph in np.arange(self.Nph):
                    currInds=np.arange(inds[ph],inds[ph+1])
                    ww[currInds,:]=self.W[ph][pv].W #matrix acting on the subpopulation pv
                    hh[currInds]=self.H[ph].S    #global hidden activity
        
                self.WDown[pv].W=ww
                self.VDown=hh
                if np.size(self.NeuronInfo[pv].n)==2:
                    self.NeuronInfo[pv].min=self.NeuronInfo[pv].center-self.NeuronInfo[pv].span/2+self.NeuronInfo[pv].sm
                    self.NeuronInfo[pv].max=self.NeuronInfo[pv].center+self.NeuronInfo[pv].span/2-self.NeuronInfo[pv].sm
                    self.NeuronInfo[pv].xAxis=np.linspace(self.NeuronInfo[pv].min[0]-self.NeuronInfo[pv].sm[0],self.NeuronInfo[pv].max[0]+self.NeuronInfo[pv].sm[0],self.NeuronInfo[pv].n[0])
                    self.NeuronInfo[pv].yAxis=np.linspace(self.NeuronInfo[pv].min[1]-self.NeuronInfo[pv].sm[1],self.NeuronInfo[pv].max[1]+self.NeuronInfo[pv].sm[1],self.NeuronInfo[pv].n[1])
    
                self.persistentChain=np.zeros((np.sum(self.NV),self.TrainParams.N_vects))
        
        #compute hidden layer given visible units
        def up(self): #done
            inds=np.insert(np.cumsum(self.NV),0,0)
            for pv in np.arange(self.Npv):
                currInds=np.arange(inds[pv],inds[pv+1])
                self.VUp[currInds]=self.V[pv].S.reshape(-1,1) #global visible activity
            
            for ph in np.arange(self.Nph):
                if np.char.equal(self.H[ph].T,'bern'):
                    gv=self.WUp[ph].W@self.VUp+self.H[ph].B
                    mu=1.0/(1.0+np.exp(-gv))
                    mu=mu.conj().T
                    rr=np.random.rand(1,self.NH[ph])
                    self.H[ph].S=1.0*(rr<mu).conj().T
                    self.H[ph].MU=mu.conj().T
                elif np.char.equal(self.H[ph].T,'poiss'):
                    gv=self.WUp[ph].W@self.VUp+self.H[ph].B
                    mu=np.exp(gv)
                    self.H[ph].S=np.random.poisson(mu)
                    self.H[ph].MU=mu.conj().T
                    
        # compute visible layer given hidden units
        def down(self): #done
            inds=np.insert(np.cumsum(self.NH),0,0)
            for ph in np.arange(self.Nph):
                currInds=np.arange(inds[ph],inds[ph+1])
                self.VDown[currInds]=self.H[ph].S #global hidden activity
            
            for pv in np.arange(self.Npv):
                if np.char.equal(self.V[pv].T,'bern'):
                    gv=self.WDown[pv].W.conj().T@self.VDown+self.V[pv].B
                    mu=1.0/(1.0+np.exp(-gv))
                    mu=mu.conj().T
                    rr=np.random.rand(1,self.NV[pv])
                    self.V[pv].S=1.0*(rr<mu).conj().T
                    self.V[pv].MU=mu.conj().T
                elif np.char.equal(self.V[pv].T,'poiss'):
                    gv=self.WDown[pv].W.conj().T@self.VDown+self.V[pv].B
                    mu=np.exp(gv)
                    self.V[pv].S=np.random.poisson(mu)
                    self.V[pv].MU=mu.conj().T
                    
        #compute hidden layer given visible units via fast matrix multiplication
        def fastUp(self,v): #done 
            w=self.allW()
            bh=self.allBh()
            gv=w@v+bh
            mu=1.0/(1.0+np.exp(-gv))
            mu=mu.conj().T
            rr=np.random.rand(np.shape(mu)[1],np.shape(mu)[0]).T
            h=1.0*(rr<mu).conj().T
            mu=mu.conj().T
            return h,mu
        
        #compute hidden layer given visible units
        def fastDown(self,h): #done 
            w=self.allW()
            bv=self.allBv()
            gv=w.conj().T@h+bv
            mu=np.exp(gv)
            v=np.random.poisson(lam=mu)
            return v,mu
        
        ## Set stuff
    
        #set visible units state from a single activity vector including all visible subpopulations
        def setV(self,ninput): 
            inind=0
            for pv in np.arange(self.Npv):
                self.V[pv].S=ninput[inind:inind+self.NV[pv]]
                inind=inind+self.NV[pv]
        
        #set hidden unit state from a single activity vector including all hidden subpopulations
        def setH(self,ninput):  
            inind=0
            for ph in np.arange(self.Nph):
                self.H[ph].S=ninput[inind:inind+self.NH[ph]]
                inind=inind+self.NH[ph]
           
        #set individual subpopulation weight matrix from a global weight matrix
        def setW(self,ninput): #done
            ini=0
            for ph in np.arange(self.Nph):
                inj=0
                for pv in np.arange(self.Npv):
                    self.W[ph][pv].W=ninput[ini:ini+self.NH[ph],inj:inj+self.NV[pv]]
                    inj=inj+self.NV[pv]
                ini=ini+self.NH[ph]
            
            for ph in np.arange(self.Nph):
                ww=np.zeros((self.NH[ph],np.sum(self.NV)))
                vv=np.zeros((np.sum(self.NV),1))
                inds=np.insert(np.cumsum(self.NV),0,0)
                for pv in np.arange(self.Npv):
                    currInds=np.arange(inds[pv],inds[pv+1])
                    ww[:,currInds]=self.W[ph][pv].W #matrix acting on the subpopulation ph
                    vv[currInds]=self.V[pv].S    #global visible activity
            
                self.WUp[ph].W=ww
                self.VUp=vv
            
            for pv in np.arange(self.Npv):
                ww=np.zeros((np.sum(self.NH),self.NV[pv]))
                hh=np.zeros((np.sum(self.NH),1))
                inds=np.insert(np.cumsum(self.NH),0,0)
                for ph in np.arange(self.Nph):
                    currInds=np.arange(inds[ph],inds[ph+1])
                    ww[currInds,:]=self.W[ph][pv].W #matrix acting on the subpopulation pv
                    hh[currInds]=self.H[ph].S    #global hidden activity
                
                self.WDown[pv].W=ww
                self.VDown=hh
        
        # set hidden units biases from a global vector        
        def setBh(self,ninput): #done
            inind=0
            for ph in np.arange(self.Nph):
                self.H[ph].B=ninput[inind:inind+self.NH[ph]]
                inind=inind+self.NH[ph]
            
        # set visible units biases from a global vector
        def setBv(self,ninput): #done
            inind=0
            for pv in np.arange(self.Npv):
                self.V[pv].B=ninput[inind:inind+self.NV[pv]]
                inind=inind+self.NV[pv]
                
        
        ## Get stuff
        #get global weight matrix from the individual subpopulations weight matrices
        def allW(self): #done
            ww=np.array([])
            for pv in np.arange(self.Npv):
                wh=np.array([])
                for ph in np.arange(self.Nph):
                    wh=np.vstack((wh,self.W[ph][pv].W)) if wh.size else self.W[ph][pv].W     
                ww=np.hstack((ww,wh)) if ww.size else wh
            return ww
        
        #get global visible units bias vector
        def allBv(self): #done
            bv=np.array([])
            for pv in np.arange(self.Npv):
                bv=np.vstack((bv,self.V[pv].B)) if bv.size else self.V[pv].B
            return bv
            
        #get global hidden units bias vector
        def allBh(self): #done
            bh=np.array([])
            for ph in np.arange(self.Nph):
                bh=np.vstack((bh,self.H[ph].B)) if bh.size else self.H[ph].B
            return bh
        
        #get global visible units state vector
        def allV(self): #done
            v=np.array([])
            for pv in np.arange(self.Npv):
                v=np.vstack((v,self.V[pv].S)) if v.size else self.V[pv].S
            return v
        
        #get global hidden units state vector
        def allH(self): #done
            h=np.array([])
            for ph in np.arange(self.Nph):
                h=np.vstack((h,self.H[ph].S)) if h.size else self.H[ph].S
            return h
        
        
        def energy(self): #done
            e=self.allH().conj().T@self.allW()@self.allV()+np.sum(np.multiply(self.allV(),self.allBv()))-np.sum(np.multiply(self.allH(),self.allBh()))
            return -e
        
        def logLikWrong(self): #done
            l=0
            for ph in np.arange(self.Nph):
                gv=self.WUp[ph].W@self.allV()+self.H[ph].B
                l=l+np.sum(np.log(1+np.exp(gv)))
            l=l+np.sum(np.multiply(self.allV(),self.allBv()))
            return l
        
        def logLik(self): #done
            l=0
            for ph in np.arange(self.Nph):
                gv=self.WUp[ph].W@self.allV()+self.H[ph].B
                l=l+np.sum(np.log(1+np.exp(gv)))
            e=0
            v=self.allV()
            print(v.sum())
            for i in np.arange(np.size(v)):
                e=e+np.sum(np.log(np.arange(1,v[i]))) #something is wrong. probably log. 
            l=l+np.sum(np.multiply(self.allV(),self.allBv()))-e
            return l
        
        #show the parameters
        def showPars(self,Plot): #done
            if Plot:
                plt.figure(1)
                count=0
                fig, axs = plt.subplots(1,np.sum((np.size(self.NV),np.size(self.NH))),figsize=(18, 9))
                for ind in np.arange(np.size(self.NV)):
                    count=count+1
                    axs[count-1].plot(self.V[ind].B)
                    axs[count-1].set_title('bias\n'+self.Names.V[ind])
                
                for ind in np.arange(np.size(self.NH)):
                    count=count+1
                    axs[count-1].plot(self.H[ind].B)
                    axs[count-1].set_title('bias\n' + self.Names.H[ind])
                fig.subplots_adjust(wspace=1)
           
            nr, nc=np.shape(self.W)
            count=0
            
            #sort according to peak of receptive field
            wcurr=self.W
            if np.size(wcurr)>2:
                order=np.argsort(np.nanmean(wcurr[0][2].W,1))
                #order=np.argsort(np.nanmean(wcurr[0][2].W,1))
            else:
                order=np.arange(np.shape(wcurr[0][0].W,0))
                #order=np.arange(np.shape(wcurr[0][0].W,0))

            for i in np.arange(nr):
                for j in np.arange(nc):
                    wcurr[i][j].W=wcurr[i][j].W[order,:]
            
            if Plot:
                plt.figure(2)
                fig2, axs2 = plt.subplots(nr,nc,figsize=(18, 9))        
                for i in np.arange(nr):
                    for j in np.arange(nc):
                        count=count+1
                        ww=wcurr[i][j].W
                        im=axs2[count-1].imshow(ww,aspect='auto') 
                        axs2[count-1].set_title(self.Names.V[j]+'\n'+self.Names.H[i])
                        divider = make_axes_locatable(axs2[count-1])
                        cax = divider.append_axes("right", "10%", pad="3%")
                        fig2.colorbar(im, cax=cax)
                fig2.subplots_adjust(wspace=1)
            return wcurr,order
        
        def showState(self): # done
            count=0
            fig, axs = plt.subplots(1,np.sum((np.size(self.NV),np.size(self.NH))))
            for ind in np.arange(np.size(self.NV)):
                count=count+1
                s=self.V[ind].S
                if np.size(self.NeuronInfo[ind].n)>1:
                    s=np.reshape(s,self.NeuronInfo[ind].n)
                im=axs[count-1].imshow(s,aspect="auto")
                axs[count-1].set_title(self.Names.V[ind])
                  
            #order=np.argsort(np.mean(self.W[0][-1].W,1))
            order=np.argsort(np.mean(self.W[0][-1],1))
            for ind in np.arange(np.size(self.NH)):
                count=count+1
                im=axs[count-1].imshow(self.H[ind].S[order],aspect="auto")
                axs[count-1].set_title(self.Names.H[ind])
            fig.subplots_adjust(wspace=1)
            
        def getPosition(self): #done
            pos=[]
            for pv in np.arange(self.Npv):
                if np.size(self.NeuronInfo[pv].n)==2:
                    wx,wy=np.meshgrid(np.arange(self.NeuronInfo[pv].n[0],dtype=float),np.arange(self.NeuronInfo[pv].n[1],dtype=float))
                    w=np.flipud(np.rot90(np.reshape(self.V[pv].S,self.NeuronInfo[pv].n)))
                    barX=np.sum(np.sum(np.multiply(w,wx)))/np.sum(w)
                    barY=np.sum(np.sum(np.multiply(w,wy)))/np.sum(w)
                    pos.append(structtype())
                    pos[pv].posNeurons=[barX,barY]
                    pos[pv].posMeters=indToPos(pos[pv].posNeurons,self.NeuronInfo[pv])
            return pos
        
        def plotPop(self,pv):
            s=np.reshape(self.V[pv].S,self.NeuronInfo[pv].n)
            s=np.flipud(np.rot90(s))
            #plt.imshow(self.NeuronInfo[pv].xAxis,self.NeuronInfo[pv].yAxis,s)
            plt.imshow(s) #extent> https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html
            return s



# compute position in meters given position in neuron units
def indToPos(ind,neuronInfo):
    pos=(neuronInfo.center-neuronInfo.span/2)+np.multiply((np.array(ind)),neuronInfo.span)/(neuronInfo.n-1)
    return pos

# compute position in neuron units given position in meters
def posToInd(pos,neuronInfo):
    ind=np.multiply((neuronInfo.n-1),(pos-neuronInfo.center+neuronInfo.span/2))/neuronInfo.span
    return ind


# generate a population code for tactile, visual and proprioceptive stimuli
def stimgen(pBc,pH,neuronInfo,gains): #done
# INPUT:
# pBc: stimulus position in body centered coordinates
# pH: stimulus position in hand centered coordinates
# neuronInfo: structure containing info about tuning curves, number of
# neurons etc for various populations. It is one of the properties of an RBM
# object
# gains: vector with gains of the different neural populations

# OUTPUT
# Bc: population encoding stimulus in body centered coordinates
# H: population encoding hand position
# T: tactile population

    dCp=.15 #hand radius
    slope=100000 #"sharpness" of tactile RF

    #body centered
    g1=gains[0] # chose gain 
    xg,yg=np.meshgrid(np.arange(neuronInfo[0].n[0],dtype=float),np.arange(neuronInfo[0].n[1],dtype=float))
    pos=posToInd(pBc,neuronInfo[0])
    Bc=g1*np.exp(   (-(pos[0]-xg)**2-(pos[1]-yg)**2 ) / (2*neuronInfo[0].tc**2) ).conj().T   
    Bc=Bc.ravel(order='F')
    
    #hand
    g2=gains[1] # chose gain according to Makin & Sabes 2015
    xg,yg=np.meshgrid(np.arange(neuronInfo[1].n[0],dtype=float),np.arange(neuronInfo[1].n[1],dtype=float))
    pos=posToInd(pH,neuronInfo[1])
    H=g2*np.exp((-(pos[0]-xg)**2-(pos[1]-yg)**2)/(2*neuronInfo[1].tc**2)).conj().T
    H=H.ravel(order='F')
    
    pHc=pBc-pH #hand centered position
    g3=gains[2]
    d=np.linalg.norm(pHc) #,2) #distance from hand
    elambda = 1.0 -np.exp(slope*(d-dCp))/(1.0 +np.exp(slope*(d-dCp))) 
    if np.isnan(elambda): elambda=0
        
    #tactile population
    #T=g3*elambda*np.ones((neuronInfo[2].n,1))
    T=g3*elambda*np.ones(neuronInfo[2].n)

    return Bc, H, T

def stimgenGainsExact_correctDecoupled(pBc,pHp,pHv,neuronInfo,gains):
    #STIMGEN function R = (NV,pos,sigma)
    #generate a population code for a stimulus in head centered coordinates,
    #and corresponding proprioceptive feedback
    #centered in pHc(hand-centered), coded by NV poisson neurons with gaussian 
    #tuning curves of width sigma, and gain randomly picked between 6.4 and 9.6 
    #INPUT:
    #pBc: 
    #Bc: population encoding stimulus in body centered coordinates
    #Hp: population encoding hand position
    dCp=.15 #hand radius
    slope=100000 #"sharpness" of tactile RF

    #body centered
    xg,yg=np.meshgrid(np.arange(neuronInfo[0].n[0],dtype=float),np.arange(neuronInfo[0].n[1],dtype=float))
    pos=posToInd(pBc,neuronInfo[0])
    Bc=gains[0]*np.exp((-(pos[0]-xg)**2-(pos[1]-yg)**2 )/(2*neuronInfo[0].tc**2)).conj().T   
    Bc=Bc.ravel(order='F')
    
    #hand proprioceptive
    g1=6.4+3.2*np.random.rand(1)  # chose gain according to Makin & Sabes 2015
    xg,yg=np.meshgrid(np.arange(neuronInfo[1].n[0],dtype=float),np.arange(neuronInfo[1].n[1],dtype=float))
    pos=posToInd(pHp,neuronInfo[1])
    H=gains[1]*np.exp((-(pos[0]-xg)**2-(pos[1]-yg)**2)/(2*neuronInfo[1].tc**2)).conj().T
    H=H.ravel(order='F')
    
    #hand visual
    xg,yg=np.meshgrid(np.arange(neuronInfo[2].n[0],dtype=float),np.arange(neuronInfo[2].n[1],dtype=float))
    pos=posToInd(pHv,neuronInfo[2])
    V=gains[2]*np.exp((-(pos[0]-xg)**2-(pos[1]-yg)**2)/(2*neuronInfo[2].tc**2)).conj().T
    V=V.ravel(order='F')

    pHc=pBc-pHp #the proprioceptive hand position is the relevant one for touch
    g3=6.4+3.2*np.random.rand(1)
    d=np.linalg.norm(pHc)
    elambda = 1.0 -np.exp(slope*(d-dCp))/(1.0 +np.exp(slope*(d-dCp))) 
    if np.isnan(elambda): elambda=0
        
    #tactile population
    T=gains[3]*elambda*np.ones(neuronInfo[3].n)

    return Bc, H, V, T

def bindata2d(x,y,z,nbins):
    #https://stackoverflow.com/questions/55874812/what-is-a-joint-histogram-and-a-marginal-histogram-in-image-processing
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html#scipy.stats.binned_statistic_2d

    Xedges=np.linspace(np.min(x),np.max(x),nbins+1)
    Yedges=np.linspace(np.min(y),np.max(y),nbins+1)

    xc=(Xedges[:-1]+Xedges[1:])/2
    yc=(Yedges[:-1]+Yedges[1:])/2

    ret = stats.binned_statistic_2d(x,y,None,statistic='count',bins=[Xedges,Yedges],expand_binnumbers=True)
    indX=ret.binnumber[0,:]
    indY=ret.binnumber[1,:]

    grid=np.zeros((nbins,nbins))
    stds=np.zeros((nbins,nbins)) # TODO explore if stds and ns are relevant and behave as expected. 
    ns=np.zeros((nbins,nbins))

    for i in np.arange(nbins):
        for j in np.arange(nbins):
            grid[i,j]=np.nanmean( z [np.where((indX==(j+1))*(indY==(i+1)))] ) 
            stds[i,j]=np.nanstd(z[ np.where((indX==(j+1))*(indY==(i+1))) ])
            ns[i,j]=np.sum(~np.isnan(z[ np.where((indX==(j+1))*(indY==(i+1))) ]))

    return grid,xc,yc,stds,ns


def bindata(x, y, numbins, type='mean', q1=0.02, q2=0.98, bins=None):
    #https://stackoverflow.com/questions/51407329/nargin-functionality-from-matlab-in-python
    #BINDATA: function [mu, bins] = bindata(x, y,numbins, type, q1, q2,bins)
    # bins the data y according to x and returns the bins and the MEAN or
    # MEDIAN value of y for that bin
    #INPUT:
    #x: x data vector
    #y: y data vector
    #numbins: number of bins
    ##type: string#: median, mean or mode, default: mean
    #q1: lower quantile of data to remove, default: 0.02
    #q2: upper quantile of data to remove, default: 0.98
    #bins: bins can be provided, in this case numbins, q1, q2 will be ignored
    #OUTPUT:
    #mu: result by bin
    #centers: centers of bins

#distances_main=distances[(distances<np.percentile(distances,98))&(distances>np.percentile(distances,2))]
#hist, bins = np.histogram(distances_main, bins=20)
#center = (bins[:-1] + bins[1:]) / 2

    medianFlag=0
    modeFlag=0

    if (type=='mean'):
        medianFlag=0
    elif (type=='median'):
        medianFlag=1
    elif (type=='mode'):
        modeFlag=1
    else:
        print('unrecognized statistical function')

    minn=np.quantile(x,q1)
    maxx=np.quantile(x,q2)

    mu = np.zeros(numbins)
    stds = np.zeros(numbins)
    sems = np.zeros(numbins)
    centers = np.zeros(numbins)

    if (bins==None):
        bins = np.linspace(minn, maxx, numbins+1)
        _,_,bin = stats.binned_statistic(x,None,statistic='count',bins=bins) #  [n,bin] = histc(x, bins)
        
        for k in np.arange(numbins):
            ind = np.where(bin==k) #find(bin==k)
            if (~np.all(ind==0)): # ~isempty(ind)
                if medianFlag:
                    mu[k] = np.nanmedian(y[ind])
                elif modeFlag:
                    mu[k] = stats.mode(y[ind])
                else:
                    mu[k] = np.nanmean(y[ind])
                
                stds[k] = np.nanstd(y[ind])
                sems[k] = np.nanstd(y[ind])/np.sqrt(np.sum(~np.isnan(y[ind])))
            else:
                mu[k]= float("NaN")
                stds[k] = float("NaN")
                sems[k] = float("NaN")
    
            centers[k]=bins[k]*0.5+bins[k+1]*0.5
        
    else:
        n,_,bin = stats.binned_statistic(x,None,statistic='count',bins=bins) #[n,bin] = histc(x, bins)

        for k in np.arange(np.size(bins)): # 1:numel(bins)-1
            ind = np.where(bin==k)
            if (~np.all(ind==0)):
                ind
                if medianFlag:
                    mu[k] = np.nanmedian(y[ind])
                elif modeFlag:
                    mu[k] = stats.mode(y[ind])
                else:
                    mu[k] = np.nanmean(y[ind])
                
                stds[k] = np.nanstd(y[ind])
                sems[k] = np.nanstd(y[ind])/np.sqrt(np.sum(~np.isnan(y[ind])))
            else:
                mu[k]= float("NaN")
                stds[k] = float("NaN")
                sems[k] = float("NaN")
    
            centers[k]=bins[k]*0.5+bins[k+1]*0.5
        
    
    return centers, mu, stds, sems