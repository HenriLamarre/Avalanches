from __future__ import print_function
import wrap_adj as wav
import numpy as np
import scipy as sp
from scipy.io import FortranFile
import time
import datetime
from matplotlib import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import copy as cp
from matplotlib.patches import FancyArrowPatch,Circle
from lmfit import minimize, Parameters
from subprocess import call
from load_journal import *
from scipy.linalg import eigh
#from AStools import deriv
#import choose_one_sunpy
#from sys import exit
import sys

class av_model():
    
    def __init__(self,SOC_case,Niter=3*10**6,Nx=48,Ny=48,\
                     Zc=1.0,sigma1=0,sigma2=0,amplitude=5e-5,\
                     D_nc=0.1,lh_soc=0,eps_drive=1e-6,\
                     dt=2.e-2,verbose=1,doplot=1,Ninf=30,fromscratch=True, name='LH'):
        ## SOC_case
        ## 100 is a generic version 
        ## 101 is double precision
        self.Nx = Nx
        self.Ny = Ny
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.amplitude = amplitude
        self.D_nc = D_nc
        self.SOC_case = SOC_case
        self.eps_drive = eps_drive
        self.lh_soc = lh_soc
        self.Zc = Zc
        self.dt = dt
        self.name = name
        if (fromscratch):
            self.gene_soc_init(Ninf=Ninf)
        else:
            self.gene_soc(Niter,verbose,doplot)
        self.time_label = 'iterations'
        self.nu_a = 0
        self.taunu = 0

        self.to_save=["Nx","Ny","sigma1","sigma2","amplitude","D_nc","SOC_case",\
                          "eps_drive","lh_soc","Zc","dt"]
        self.to_save.append("time")
        self.to_save.append("time_label")
        self.to_save.append("taunu")
        self.to_save.append("nu_a")
        self.to_save.append("B")
        self.to_save.append("Z")
        self.to_save.append("lat_e")
        self.to_save.append("rel_e")
        self.to_save.append("B_tmp")
        self.to_save.append("Z_tmp")
        self.to_save.append("lat_e_tmp")
        self.to_save.append("rel_e_tmp")

        self.to_save_h5=["Nx","Ny",\
                             "sigma1","sigma2",\
                             "amplitude","D_nc",\
                             "SOC_case",\
                             "eps_drive","lh_soc",\
                             "Zc","dt","nu_a",\
                             "B","Z"]

        self.state_to_save = cp.copy(self.to_save)
        self.state_to_save.remove("time")
        self.state_to_save.remove("lat_e")
        self.state_to_save.remove("rel_e")
        self.state_to_save.remove("lat_e_tmp")
        self.state_to_save.remove("rel_e_tmp")

        self.to_save_4Dvar = cp.copy(self.state_to_save)
        self.to_save_4Dvar.append("time_tmp")
        self.to_save_4Dvar.append("rel_e_avg")
        self.to_save_4Dvar.append("obs_list")
        self.to_save_4Dvar.append("dns_obs_list")
        self.to_save_4Dvar.append("newB_list")
        self.to_save_4Dvar.append("threshold")
        self.to_save_4Dvar.append("binsize")

    def show_params(self):
        print('****************'                )
        print('Model parameters'                )
        print('****************'                )
        print('SOC case',self.SOC_case          )
        print('Eps_drive ',self.eps_drive       )
        print('D_nc ',self.D_nc                 )
        print('lh_soc ',self.lh_soc             )
        print('Sigma1 ',self.sigma1             )
        print('Sigma2 ',self.sigma2             )
        print('Resolution',self.Nx,'x',self.Ny  )
        print('****************'                )
 
    ## NEW VERSION OF SOC GENERATION
    def gene_soc_init(self,Ninf=30):
        ## B
        self.Bini = np.zeros((self.Nx,self.Ny))
        for i in r_[0:self.Nx]:
            ii = 2.0*float(i)/(self.Nx-1)-1.0
            for j in r_[0:self.Ny]:
                jj = 2.0*float(j)/(self.Ny-1)-1.0
                self.Bini[i,j] = (1.0-(ii**2))/2.0
                for k in r_[1:Ninf:2]:
                    self.Bini[i,j] = self.Bini[i,j] - (16.0/(np.pi**3))* \
                    ( np.sin(k*np.pi*(1.0+ii)/2.0)/(k**3*np.sinh(k*np.pi)) )* \
                    (np.sinh(k*np.pi*(1.0+jj)/2.0)+np.sinh(k*np.pi*(1.0-jj)/2.0))
        self.B=self.Bini

        ## Z
        self.Z = np.zeros((self.Nx,self.Ny))
        for i in r_[1:self.Nx-1]:
            for j in r_[1:self.Ny-1]:
                self.Z[i,j] = self.B[i,j] - (self.B[i-1,j]+self.B[i+1,j]+self.B[i,j-1]+self.B[i,j+1])/4.0
        Zmean=np.mean(self.Z[1:-1,1:-1])
        self.B=self.B*0.59*self.Zc/Zmean
        self.Bini=self.Bini*0.59*self.Zc/Zmean
        self.Z=self.Z*0.59*self.Zc/Zmean

    ## Generates the SOC state
    def gene_soc(self,Niter,verbose,doplot):

        if (verbose):
            print('Generating SOC...')
        if (self.SOC_case == 100):
            idum_ini = idum_bank(0)
            lat_e,rel_e,rel_tmp,B,Z = wav.do_avalanche_generic(Niter,nx=self.Nx,\
                                                                   ny=self.Ny,zc=self.Zc,\
                                                                   d_nc=self.D_nc,\
                                                                   eps_drive=self.eps_drive,\
                                                                   sigma1=self.sigma1,sigma2=self.sigma2,\
                                                                   lh_soc=self.lh_soc,\
                                                                   idum_init=idum_ini,\
                                                                   verbose=verbose)
            self.time = zeros(Niter)
            for it in r_[0:Niter]:
                self.time[it]=it
            self.time_label='Iterations'
        elif (self.SOC_case == 101):
            idum_ini = idum_bank(0)
            lat_e,rel_e,rel_tmp,B,Z = wav.do_avalanche_generic_dp(Niter,nx=self.Nx,\
                                                                      ny=self.Ny,zc=self.Zc,\
                                                                      d_nc=self.D_nc,\
                                                                      eps_drive=self.eps_drive,\
                                                                      sigma1=self.sigma1,sigma2=self.sigma2,\
                                                                      lh_soc=self.lh_soc,\
                                                                      idum_init=idum_ini,\
                                                                      verbose=verbose, name=self.name)
            self.time = zeros(Niter)
            for it in r_[0:Niter]:
                self.time[it]=it
            self.time_label='Iterations'
        elif (self.SOC_case == 102):
            idum_ini = idum_bank(0)
            lat_e,rel_e,rel_tmp,B,Z = wav.do_avalanche_generic_ho(Niter,nx=self.Nx,\
                                                                      ny=self.Ny,zc=self.Zc,\
                                                                      d_nc=self.D_nc,\
                                                                      eps_drive=self.eps_drive,\
                                                                      sigma1=self.sigma1,sigma2=self.sigma2,\
                                                                      lh_soc=self.lh_soc,\
                                                                      idum_init=idum_ini,\
                                                                      verbose=verbose)
            self.time = zeros(Niter)
            for it in r_[0:Niter]:
                self.time[it]=it
            self.time_label='Iterations'
        elif (self.SOC_case == 103):
            idum_ini = idum_bank(0)
            lat_e,rel_e,rel_tmp,B,Z = wav.do_avalanche_generic_ncweird(Niter,nx=self.Nx,\
                                                                           ny=self.Ny,zc=self.Zc,\
                                                                           d_nc=self.D_nc,\
                                                                           eps_drive=self.eps_drive,\
                                                                           sigma1=self.sigma1,sigma2=self.sigma2,\
                                                                           lh_soc=self.lh_soc,\
                                                                           idum_init=idum_ini,\
                                                                           verbose=verbose)
            self.time = zeros(Niter)
            for it in r_[0:Niter]:
                self.time[it]=it
            self.time_label='Iterations'
        else:
            print('SOC case not implemented yet!')
            return
            
        if (doplot == 1):
            figure()
            subplot(121)
            plot(self.time,lat_e/1e6)
            xlabel(self.time_label)
            ylabel('Lattice Energy [10^6]')
            subplot(122)
            plot(self.time,rel_e)
            xlabel(self.time_label)
        self.B = B
        self.Z = Z
        self.lat_e = lat_e
        self.rel_e = rel_e
        self.B_tmp = B
        self.Z_tmp = Z
        self.lat_e_tmp = lat_e
        self.rel_e_tmp = rel_e        

    ## Calculate covariance matrix
    def covar_matrix(self,Niter=10**6,i_idum=0,nsteps=10):
        # Save B state every step
        Binit = cp.copy(self.B)
        self.Bstates = np.zeros((nsteps,self.Nx*self.Ny))
        for ii in xrange(nsteps):
            self.do_soc(Niter=Niter/nsteps,i_idum=i_idum,finish_with_soc=False)
            self.Bstates[ii,:]=cp.copy(self.B).ravel()

        # Compute the co-variance matrix
        Bmean = np.tile(np.mean(self.Bstates,axis=0),(nsteps,1))
        Ystate = (self.Bstates-Bmean) ; nvals=nsteps*self.Nx*self.Ny
        
        #self.Covar = np.zeros((self.Nx*self.Ny,self.Nx*self.Ny))
        self.CoVar = np.dot(Ystate.T,Ystate)/float(nsteps)

        # Find eigenvector and eigenvalues with scipy
        self.eigval,self.eigvec=eigh(self.CoVar,eigvals_only=False)

    ## Do some iterations
    def do_soc(self,Niter=3*10**6,doplot=0,update=1,verbose=0,i_idum=0,psfa='',psfb='',finish_with_soc=True, name='LH'):

        if (self.SOC_case == 100):
            idum_ini = idum_bank(i_idum)
            lat_e,rel_e,rel_tmp,B,Z = wav.do_avalanche_generic(Niter,nx=self.Nx,\
                                                                   ny=self.Ny,zc=self.Zc,\
                                                                   d_nc=self.D_nc,\
                                                                   eps_drive=self.eps_drive,\
                                                                   sigma1=self.sigma1,\
                                                                   sigma2=self.sigma2,\
                                                                   lh_soc=self.lh_soc,\
                                                                   binit=self.B,\
                                                                   idum_init=idum_ini,\
                                                                   verbose=verbose)
            self.rel_e2=rel_tmp
            time = np.linspace(0,Niter-1,num=Niter)
        elif (self.SOC_case == 101):
            if (i_idum == -1):
                if hasattr(self,'last_idum'):
                    idum_ini=self.last_idum
                else:
                    idum_ini = 0
            else:
                idum_ini = idum_bank(i_idum)
            lat_e,rel_e,rel_tmp,B,Z,last_idum = wav.do_avalanche_generic_dp(Niter,nx=self.Nx,\
                                                                                ny=self.Ny,zc=self.Zc,\
                                                                                d_nc=self.D_nc,\
                                                                                eps_drive=self.eps_drive,\
                                                                                sigma1=self.sigma1,\
                                                                                sigma2=self.sigma2,\
                                                                                lh_soc=self.lh_soc,\
                                                                                binit=self.B,\
                                                                                idum_init=idum_ini,\
                                                                                verbose=verbose,
                                                                                name=self.name)
            if (update == 1):
                self.last_idum = last_idum
            self.rel_e2=rel_tmp
            time = np.linspace(0,Niter-1,num=Niter)
        elif (self.SOC_case == 102):
            idum_ini = idum_bank(i_idum)
            lat_e,rel_e,rel_tmp,B,Z = wav.do_avalanche_generic_ho(Niter,nx=self.Nx,\
                                                                      ny=self.Ny,zc=self.Zc,\
                                                                      d_nc=self.D_nc,\
                                                                      eps_drive=self.eps_drive,\
                                                                      sigma1=self.sigma1,\
                                                                      sigma2=self.sigma2,\
                                                                      lh_soc=self.lh_soc,\
                                                                      binit=self.B,\
                                                                      idum_init=idum_ini,\
                                                                      verbose=verbose)
            self.rel_e2=rel_tmp
            time = np.linspace(0,Niter-1,num=Niter)
        elif (self.SOC_case == 103):
            idum_ini = idum_bank(i_idum)
            lat_e,rel_e,rel_tmp,B,Z = wav.do_avalanche_generic_ncweird(Niter,nx=self.Nx,\
                                                                           ny=self.Ny,zc=self.Zc,\
                                                                           d_nc=self.D_nc,\
                                                                           eps_drive=self.eps_drive,\
                                                                           sigma1=self.sigma1,\
                                                                           sigma2=self.sigma2,\
                                                                           lh_soc=self.lh_soc,\
                                                                           binit=self.B,\
                                                                           idum_init=idum_ini,\
                                                                           verbose=verbose)
            self.rel_e2=rel_tmp
            time = np.linspace(0,Niter-1,num=Niter)
        else:
            print("SOC Case not implemented yet")

        # Verify that we always end with a non-avalanching iteration
        if (amax(abs(Z)) > self.Zc)and (rel_e[-1] != 0.0)and(finish_with_soc):
            if (verbose != 0):
                print("Erasing the last avalanche, because it is still going on")
            indx=where(rel_e == 0.0)[0]
            #if (len(indx) < Niter/2):
            #    self.do_soc(Niter=Niter+10,doplot=doplot,update=update,verbose=verbose,i_idum=i_idum)
            #    print 'Trying to finish with soc...'
            #else:
            print('Trying to finish with soc (V2): doing ',indx[-1]+1,'iterations')
            self.do_soc(Niter=indx[-1]+1,doplot=doplot,update=update,verbose=verbose,i_idum=i_idum)

        else:
        ## Update and plot
            if (update == 1):
                self.B = B
                self.Z = Z
                self.lat_e = lat_e
                self.rel_e = rel_e
                self.time  = time
            self.B_tmp=B
            self.Z_tmp=Z
            self.lat_e_tmp=lat_e
            self.rel_e_tmp=rel_e
            self.time_tmp = time
            if (doplot == 1):
                if (psfa == ''):
                    fig=figure(figsize=(10,8))
                else:
                    fig=figure(figsize=(8,5))
                subplot(211)
                plot(time,lat_e)
                ylabel('Lattice energy')
                draw()
                ax=gca()
                #ytlabels = [item.get_text() for item in ax.get_yticklabels()]
                #ytlabels[0] = ''
                #print ytlabels
                #ax.set_yticklabels(ytlabels)
                setp(ax.get_xticklabels(),visible=False)
                ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                annotate('(a)',xy=(0.01,0.925),xycoords='axes fraction')
                subplots_adjust(hspace=0)
                subplot(212)
                plot(time,rel_e)
                xlabel(self.time_label)
                ylabel('Avalanche energy release')
                draw()
                ax=gca()
                #ytlabels = [item.get_text() for item in ax.get_yticklabels()]
                #ytlabels[-1] = ''
                #ax.set_yticklabels(ytlabels)
                ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
                annotate('(b)',xy=(0.01,0.925),xycoords='axes fraction')
                if (psfa != ''):
                    savefig(psfa,transparent=True)
                show()

        if (psfb != ''):
            figure(figsize=(7,5))
            plot(time,rel_e)
            xlabel(self.time_label)
            ylabel('Avalanche energy')
            savefig(psfb,transparent=True)

        if hasattr(self,'lat_e'):
            self.emoy = np.mean(self.lat_e)
            self.sigma = np.std(self.lat_e)
            self.emax = np.max(self.lat_e)
            self.emin = np.min(self.lat_e)
            
    def test_j(self,Niter=1e4,threshold=0,tw=100,binsize=1,i_idum=0,verbose=0):
         # Testing the results from tapenade
        idum_ini = idum_bank(i_idum)
        idum_obs = idum_bank(0)
        lat_e,rel_e,rel_tmp,B,Z,gradJ = wav.test_tapenade(Niter,nx=self.Nx,\
                                                              ny=self.Ny,zc=self.Zc,\
                                                              d_nc=self.D_nc,\
                                                              eps_drive=self.eps_drive,\
                                                              sigma1=self.sigma1,\
                                                              sigma2=self.sigma2,\
                                                              lh_soc=self.lh_soc,\
                                                              binit=self.B,\
                                                              idum_obs=idum_obs,\
                                                              idum_init=idum_ini,\
                                                              verbose=verbose,\
                                                              threshold=threshold,\
                                                              binsize=binsize,\
                                                              tw=tw)
        self.gradJ = gradJ
        self.t_lat_e = lat_e
        self.t_rel_e = rel_e
        self.t_rel_e2= rel_tmp
        self.t_B = B
        self.t_Z = Z
            
    def mkobs(self, binsize=100,threshold=50):
        rel_e_avg=np.zeros_like(self.time_tmp)
    
        index=0
        for i in range(len(self.time_tmp))[::binsize]:
            summ = np.sum(self.rel_e_tmp[index*binsize:(index+1)*binsize])
            rel_e_avg[index*binsize:(index+1)*binsize]=summ/binsize
            index=index+1
    
        for i in range(len(self.time_tmp)):
            if (rel_e_avg[i] < threshold):
                rel_e_avg[i] = 0.0

        return rel_e_avg

    def get_new_init(self,obs,Niter=1e4,threshold=0.0,binsize=1,tw=100,i_idum=0,verbose=0,\
                     algo=0,lambd=1e-2,n_simplex=1,temp=1.,idum_an=1,niter_step=20000,n_steps=25):

        if not hasattr(self,'EVsaved') and algo==4:
            self.EVsaved = ''
            self.ev=np.zeros((self.Nx*self.Ny,2))

        if not hasattr(self,'VersionCost'):
            self.VersionCost=0

        idum_ini = idum_bank(i_idum)
        idum_anneal = idum_bank(idum_an)

        new_B, new_grad, final_cost = wav.find_new_init(int(Niter),nx=self.Nx,\
                                                        ny=self.Ny,\
                                                        binit=self.B,\
                                                        zc=self.Zc,\
                                                        d_nc=self.D_nc,\
                                                        eps_drive=self.eps_drive,\
                                                        sigma1=self.sigma1,\
                                                        sigma2=self.sigma2,\
                                                        lh_soc=self.lh_soc,\
                                                        idum_init=idum_ini,\
                                                        verbose=verbose,\
                                                        binsize=binsize,\
                                                        threshold=threshold,\
                                                        tw=tw,\
                                                        lambd=lambd,\
                                                        n_simplex = n_simplex,\
                                                        temp = temp,\
                                                        idum_anneal=idum_anneal,\
                                                        niter_step=niter_step,\
                                                        n_steps=n_steps,\
                                                        obs_erelavg=obs,\
                                                        versioncost=self.VersionCost,\
                                                        emoy=self.emoy,\
                                                        sigma=self.sigma,\
                                                        emax=self.emax,\
                                                        emin=self.emin,\
                                                        evsaved=self.EVsaved,\
                                                        nev=self.ev.shape[1],\
                                                        algo=algo)

        self.new_B = cp.copy(new_B)
        self.cost  = final_cost
        if verbose:
            print('Final cost (python):',self.cost)

    def find_new_init(self,Niter=1e4,threshold=0.0,binsize=1,tw=100,i_idum=0,verbose=0,
                      algo=0,lambd=1e-2,n_simplex=1,temp=1.,idum_an=1,niter_step=20000,n_steps=25,doplot=False):
        # Create some observations
        self.do_soc(Niter=Niter,update=0,doplot=0,i_idum=0,finish_with_soc=False)
        #self.obs=self.mkobs(binsize=binsize,threshold=threshold)
        self.obs=self.mkobs(binsize=binsize,threshold=0)
        
        # Advance in time to generate a new initial condition
        self.do_soc(Niter=1e5,doplot=0,i_idum=i_idum,finish_with_soc=True)

        # Then, DNS
        self.do_soc(Niter=Niter,update=0,doplot=0,i_idum=i_idum,finish_with_soc=False)
        if verbose:
            IniCost=self.get_cost(self.obs,i_idum=i_idum,binsize=binsize,threshold=threshold,tw=tw)
            print('Initial cost in python',IniCost)

        # Do the minimization to find a new initial condition
        self.get_new_init(self.obs,Niter=Niter,threshold=threshold,binsize=binsize,\
                          tw=tw,i_idum=i_idum,verbose=verbose,\
                          algo=algo,lambd=lambd,n_simplex=n_simplex,\
                          temp=temp,idum_an=idum_an,niter_step=niter_step,n_steps=n_steps)

        # Do a soc case with the new initial condition
        self.B = cp.copy(self.new_B)
        self.do_soc(Niter=Niter,update=0,doplot=0,i_idum=i_idum,finish_with_soc=False)
        #self.obs_4dvar=self.mkobs(binsize=binsize,threshold=threshold)
        self.obs_4dvar=self.mkobs(binsize=binsize,threshold=0)
        #self.obs_4dvar = cp.copy(self.rel_e_avg)
        if verbose:
            self.FinalCost=self.get_cost(self.obs,i_idum=i_idum,binsize=binsize,threshold=threshold,tw=tw)
            print('Final cost in python',self.FinalCost)

        if doplot:
            fig=figure()
            subplot(311)
            plot(self.obs)
            title('OBS')
            subplot(312)
            plot(self.obs_dns)
            title('DNS')
            subplot(313)
            plot(self.obs_4dvar)
            title('4DVAR')

    def get_cost(self,obs,Niter=0,binsize=1,threshold=0.,tw=100,i_idum=0,verbose=0):
        if not hasattr(self,'VersionCost'):
            self.VersionCost=0

        idum_ini = idum_bank(i_idum)
        if (Niter == 0):
            Niter = len(obs)
        Bini = 1.*self.B
        ObsOld=1.*obs
        Cost=wav.get_cost(Niter,nx=self.Nx,\
                                ny=self.Ny,zc=self.Zc,\
                                d_nc=self.D_nc,\
                                eps_drive=self.eps_drive,\
                                sigma1=self.sigma1,\
                                sigma2=self.sigma2,\
                                lh_soc=self.lh_soc,\
                                binit=self.B,\
                                idum_init=idum_ini,\
                                verbose=verbose,\
                                threshold=threshold,\
                                binsize=binsize,\
                                tw=tw,\
                                obs_erelavg=ObsOld,\
                                versioncost=self.VersionCost)
        self.B = 1.*Bini
        return Cost
    def plot_3d_state(self,mycmap='viridis',savefile=''):
        
        fig=figure(figsize=(8,4))
        load_j('')
        ax=fig.add_subplot(1,2,1,projection='3d')
        #Grid
        X,Y=np.meshgrid(np.arange(0.,1.,1./self.Nx),np.arange(0.,1.,1./self.Ny))
        surf=ax.plot_surface(X,Y,self.B,cmap=mycmap,linewidth=0,antialiased=False)
        cset = ax.contourf(X, Y, self.B, zdir='z', offset=1000, cmap=mycmap)
        ax.set_zlim([0.,1000.])
        title('Sandpile')
        ax.set_aspect('equal')
        # Colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5,label='Height')
        
        ax=fig.add_subplot(1,2,2,projection='3d')
        surf=ax.plot_surface(X,Y,self.Z,cmap=mycmap,linewidth=0,antialiased=False)
        cset = ax.contourf(X, Y, self.Z, zdir='z', offset=4., cmap=mycmap)
        ax.set_zlim([0.,4.])
        ax.set_aspect('equal')
        title('Instability criterion')
        # Colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5,label='Curvature')
        fig.tight_layout()

        if savefile != '':
            fig.savefig(savefile,bbox_inches='tight',transparent=True)
        

    def covar_matrix(self,Niter=1e5,nsteps=100):

        Ninf = 30
        self.Bini = np.zeros((self.Nx,self.Ny))
        for i in r_[0:self.Nx]:
            ii = 2.0*float(i)/(self.Nx-1)-1.0
            for j in r_[0:self.Ny]:
                jj = 2.0*float(j)/(self.Ny-1)-1.0
                self.Bini[i,j] = (1.0-(ii**2))/2.0
                for k in r_[1:Ninf:2]:
                    self.Bini[i,j] = self.Bini[i,j] - (16.0/(pi**3))* \
                    ( sin(k*pi*(1.0+ii)/2.0)/(k**3*sinh(k*pi)) )* \
                    (sinh(k*pi*(1.0+jj)/2.0)+sinh(k*pi*(1.0-jj)/2.0))
        
        P = 0
        P_dif = 0
        for i in r_[0:nsteps]:
            self.do_soc(Niter=Niter/nsteps,finish_with_soc=False)
            y = np.ravel(self.B)
            Ptemp = np.dot(y.reshape(len(y),1),y.reshape(1,len(y)))
            P = P + Ptemp
            #y_dif = np.ravel(self.B)/np.max(np.ravel(self.B)) - np.ravel(Bini)/np.max(np.ravel(Bini))
            y_dif = np.ravel(self.B) - np.ravel(self.Bini)
            Ptemp_dif = np.dot(y_dif.reshape(len(y_dif),1),y_dif.reshape(1,len(y_dif)))
            P_dif = P_dif + Ptemp_dif

        P = P/nsteps
        P_dif = P_dif/nsteps
        #P matrix diagonalisation
        self.eigval, self.eigvec = sp.linalg.eigh(P)
        self.eigval_dif, self.eigvec_dif = sp.linalg.eigh(P_dif)

    def OrganizeEigenFunc(self,Niter=10**7,Ntest=10**4,nsteps=100,binsize=1,threshold=0,tw=200,i_idum=0,recompute=False,verbose=False):

        if not hasattr(self,'ID'):
            self.CreateID()
        
        if not hasattr(self,'eigval'):
            if not hasattr(self,'dir_saved'):
                self.dir_saved='/Users/astrugar/WORK/Astro/ModelesReduits/Avalanches/AVSF/Cases/'
            saved_file=self.dir_saved+'EigData_'+self.ID+'_{}_{}'.format(Niter,nsteps)
            if not os.path.exists(saved_file+".bin"):
                print("Recomputing...")
                self.covar_matrix(Niter=Niter,nsteps=nsteps)
                save_elements([self.eigval,self.eigvec],saved_file)
            else:
                print("Reading {}.bin".format(saved_file))
                [self.eigval,self.eigvec]=read_elements(2,saved_file)

        # perturbation
        pert=1.e-2

        # Store initial condition
        Bini = 1.*self.B

        # Filter zero eigenvalues
        indx=np.where(self.eigval != 0.)[0]
        eigvec=self.eigvec[:,indx]
        eigval=self.eigval[indx]
        neig=len(eigval)
        #print("I filtered {}/{} eigenvectors ".format(neig,len(self.eigval)))
        #list_vec = [eigvec[:,ii] for ii in range(len(eigval))]
        
        # Project the initial condition on the EigenVectors
        eei = 1.*eigvec
        #eei = np.concatenate([v.reshape(1,self.Nx*self.Nx) for v in eigvec.T]).T
        lambdas = np.dot(Bini.reshape(1,self.Nx*self.Ny),eei).squeeze()

        # Generate synthetic obs
        self.do_soc(Ntest,update=0)
        RefObs=self.mkobs(binsize=binsize,threshold=threshold)
        MaxEvent=RefObs.max()
        MyThreshold=MaxEvent/2.
        RefObs2=self.mkobs(binsize=binsize,threshold=MyThreshold)
        if verbose:
            print("Obs done")
        
        # Save information
        meanB=np.mean(Bini) # just for identification
        bsaved = self.dir_saved+'EigData_'+self.ID+'_{}_{}_{}_{}'.format(Ntest,nsteps,meanB,MyThreshold)

        # Loop over random number sequences and comput costs
        if verbose:
            print("Calculating costs")
        cost=np.zeros(len(lambdas))
        bsaved_idum=bsaved+"_{}".format(i_idum)
        if not os.path.exists(bsaved_idum+".bin") or recompute:
            for il,lamb in enumerate(lambdas):
                print("Lambda {}/{}\r".format(il,len(lambdas)),end='\r')
                sys.stdout.flush()
                lambdas_pert = 1.*lambdas
                lambdas_pert[il] = (1.+pert)*lamb
                newB = np.sum(np.tile(lambdas_pert,(self.Nx*self.Ny,1))*eei,axis=1).reshape(self.Nx,self.Ny)
                self.B = newB
                cost[il]=self.get_cost(RefObs,Niter=Ntest,binsize=binsize,threshold=MyThreshold,tw=tw,i_idum=i_idum)
            if verbose:
                print("Saving "+bsaved_idum)
            save_elements([cost.ravel()],bsaved_idum)
        else:
            if verbose:
                print("Reading "+bsaved_idum)
            cost = read_elements(1,bsaved_idum)[0]

        # Reorganize eigenvalues
        self.ev = eei[:,cost.argsort()[::-1]]
        if verbose:
            print("Shape of eigvectors array is {}".format(self.ev.shape))
        
        # Save ev into file to be read in fortran (too large for some machine to communicate through f2py)
        EVsaved = self.dir_saved+'OragnizedEigData_'+self.ID+'_{}_{}_{}_{}'.format(Ntest,nsteps,meanB,MyThreshold)
        fd=FortranFile(EVsaved,'w')
        fd.write_record(self.ev.T)
        fd.close()
        self.EVsaved=EVsaved
        
        # Restore initial state
        self.B = 1.*Bini
        
    def TestOrganizeEigenFunc(self,Niter=10**6,nsteps=100,binsize=1,threshold=0,tw=200,doplot=True,nidum=1,recompute=False):

        if not hasattr(self,'ID'):
            self.CreateID()
        
        if not hasattr(self,'eigval'):
            if not hasattr(self,'dir_saved'):
                self.dir_saved='/Users/astrugar/WORK/Astro/ModelesReduits/Avalanches/AVSF/Cases/'
            saved_file=self.dir_saved+'EigData_'+self.ID+'_{}_{}'.format(Niter,nsteps)
            if not os.path.exists(saved_file+".bin"):
                print("Recomputing...")
                self.covar_matrix(Niter=Niter,nsteps=nsteps)
                save_elements([self.eigval,self.eigvec],saved_file)
            else:
                print("Reading {}.bin".format(saved_file))
                [self.eigval,self.eigvec]=read_elements(2,saved_file)

        # perturbation
        pert=1.e-2

        # Store initial condition
        Bini = 1.*self.B

        # Project the initial condition on the EigenVectors
        eei = np.concatenate([v.reshape(1,self.Nx*self.Nx) for v in self.eigvec]).T
        lambdas = np.dot(Bini.reshape(1,self.Nx*self.Ny),eei).squeeze()

        # Generate synthetic obs
        print("Generating synthetic data")
        self.do_soc(Niter,update=0)
        RefObs=self.mkobs(binsize=binsize,threshold=threshold)
        #RefObs=1.*self.rel_e_avg
        MaxEvent=self.rel_e_avg.max()
        MyThreshold=MaxEvent/2.
        RefObs2=self.mkobs(binsize=binsize,threshold=MyThreshold)

        # Save information
        meanB=np.mean(Bini) # just for identification
        bsaved = self.dir_saved+'EigData_'+self.ID+'_{}_{}_{}_{}'.format(Niter,nsteps,meanB,MyThreshold)

        # Loop over random number sequences and comput costs
        print("Generate cost")
        cost=np.zeros((nidum,len(lambdas)))
        for idum in range(nidum):
            # Perturb and calculate cost
            bsaved_idum=bsaved+"_{}".format(idum)
            if not os.path.exists(bsaved_idum+".bin") or recompute:
                print("Saving cost to "+bsaved_idum)
                for il,lamb in enumerate(lambdas):
                    lambdas_pert = 1.*lambdas
                    lambdas_pert[il] = (1.+pert)*lamb
                    newB = np.sum(np.tile(lambdas_pert,(self.Nx*self.Ny,1))*eei,axis=1).reshape(self.Nx,self.Ny)
                    self.B = newB
                    #cost[idum,il]=self.get_cost(RefObs,Niter=Niter,binsize=binsize,threshold=MyThreshold,tw=tw,i_idum=idum)
                #save_elements([cost[idum,:].ravel()],bsaved_idum)
            else:
                print("Reading cost in "+bsaved_idum)
                cost[idum,:] = read_elements(1,bsaved_idum)[0]

        print("Plot")
        if doplot:
            # Sort and Plot
            fig=figure(figsize=(8,3))
            subplot(121)
            ind_sorted = cost[-1,:].argsort()[::-1]
            for idum in range(nidum):
                #ind_sorted=cost[idum,:].argsort()[::-1]
                # Plot
                plot(range(len(lambdas)),cost[idum,ind_sorted],marker='o',ls='',label='idum {}'.format(idum))
            yscale('log')
            subplot(122)
            for idum in range(nidum):
                # Plot
                plot(range(len(lambdas)),cost[idum,:],marker='o',ls='',label='idum {}'.format(idum))
            yscale('log')            
            
        self.cost_testEig = cost

        # Restore initial state
        self.B = 1.*Bini

    def TestEigenFunc(self,Niter=10**6,nsteps=100,recompute=False,RecoCoVar=False):

        if not hasattr(self,'ID'):
            self.CreateID()
            
        if not hasattr(self,'dir_saved'):
            self.dir_saved='/Users/astrugar/WORK/Astro/ModelesReduits/Avalanches/AVSF/Cases/'
        saved_file=self.dir_saved+'EigData_'+self.ID+'_{}_{}'.format(Niter,nsteps)
        if hasattr(self,'EigFile'):
            if saved_file != self.EigFile:
                recompute=True
        if not hasattr(self,'eigval'):
            recompute=True

        if recompute:
            if not os.path.exists(saved_file+".bin") or RecoCoVar:
                print("Recomputing...")
                self.covar_matrix(Niter=Niter,nsteps=nsteps)
                save_elements([self.eigval,self.eigvec],saved_file)
                self.EigFile=saved_file
            else:
                print("Reading {}.bin".format(saved_file))
                [self.eigval,self.eigvec]=read_elements(2,saved_file)
                self.EigFile=saved_file                

        # Filter only positive eigenvalues
        indx=np.where(self.eigval != 0.)[0]
        eigvec=self.eigvec[:,indx]
        eigval=self.eigval[indx]
        list_vec = [eigvec[:,ii] for ii in range(len(eigval))]
        list_allvec = [self.eigvec[:,ii] for ii in range(len(self.eigval))]
        
        nbad=0 ; bad_eigvals=[]
        for ivec,vec in enumerate(list_allvec):
            if vec.max() == 1.:
                nbad+=1
                bad_eigvals.append(self.eigval[ivec])
        if nbad != 0:
            print("I found {}/{} bad eigenvectors".format(nbad,len(self.eigval)))
            print("Bad eigenvalues are {}".format(bad_eigvals))

        print("I filtered {}/{} eigenvectors ".format(len(self.eigval)-len(eigval),len(self.eigval)))

        nbad=0
        for ivec,vec in enumerate(list_vec):
            if vec.max() == 1.:
                nbad+=1
        if nbad != 0:
            print("I found {}/{} bad eigenvectors in the filtered list".format(nbad,len(self.eigval)))
        else:
            print("No bad eigenvectors left")

        #for ivec,vec in enumerate(self.eigvec):
        #    if vec.max() == 1.:
        #        fig=figure(figsize=(3,3))
        #        pcolormesh(np.arange(self.Nx+1),np.arange(self.Ny+1),vec.reshape((self.Nx,self.Ny)),\
        #                   cmap='bwr',rasterized=True)
        #        axis('equal') ; axis('tight')
        #        ax=gca()
        #        setp(ax.get_xticklabels(),visible=False)
        #        setp(ax.get_yticklabels(),visible=False)
        #        title("EigenVector {}".format(ivec))
        
    def CreateID(self):
        """ Create an ID for the model when saving various data """
        self.ID = "{}_{}_{}_{}_{}_{}_{}_{}".format(self.SOC_case,self.eps_drive,self.D_nc,self.lh_soc,self.sigma1,self.sigma2,self.Nx,self.Ny)
        
#################################################################
# End of SOC class

def get_only_cost(e_relavg,obs,Niter=0,threshold=0.,tw=100):
    if (Niter == 0):
        Niter = len(obs)
    return wav.get_only_cost(Niter,threshold=threshold,tw=tw,\
                                 obs_erelavg=obs,\
                                 e_relavg=e_relavg)

#################################################################
# routines rox

              
def cum_wt(B,Niter=1e7, NE=100, recompute=False):
    
    tini = time.time()
    Nname = '%3.0e' % Niter
    save_name_DNS = '/Users/rbarnabe/These/Avalanches/avsf/Resultats_avsf/Epic'+Nname
    if os.path.exists(save_name_DNS+'.bin') and not recompute:
        print('reading - released energy')
        Epic, Tpic  = read_elements(2,save_name_DNS)
    else :
        print('calculating - released energy')
        B.do_soc(1e6)
        B.do_soc(Niter)
        Epic, Tpic = condense_time_serie(B.rel_e,B.time,threshold=0)
        save_elements([Epic,Tpic],save_name_DNS)

    print('analysing')
    Emax = max(Epic)
    Emin = 0
    E_th = np.arange(Emin, Emax, round(Emax/NE))[0:NE]

    mean_WT = np.zeros(NE)        #initialisation mean wt
    for ie, E in enumerate(E_th):
        indx = np.where(Epic > E)[0]      #moments ou il y a avalanche > seuil
        WT = np.zeros(len(indx)-1)        #initialisation waiting time
        for i in r_[0:len(indx)-1]:
            i1 = indx[i]
            i2 = indx[i+1]
            WT[i] = Tpic[i2]-Tpic[i1]     #calcul waiting time

        mean_WT[ie] = mean(WT)

    fig = figure()
    plot(E_th,mean_WT, 'o')
    xlabel('Released energy')
    ylabel('Mean waiting time')
    figdir = '/Users/rbarnabe/These/Avalanches/avsf/Resultats_avsf/results_DNS/cumul_WT/cumul_WT_N'+Nname+'_NE'+str(NE)+'.pdf'
    savefig(figdir)
    print(time.time() - tini, 'sec')

    


#################################################################
## Analysis routines

def condense_time_serie(E_released,time,threshold=0.0,timers=[1000],\
                        datetype=False,ini_avalanche=False,end_avalanche=False,\
                        verbose=False,InfoRedux=None,peakmode=False):

    dt      = time[1]-time[0]
    E_av        = 0.0
    time_ini_av = 0.0
    if (datetype):
        delay   = datetime.timedelta(minutes=0)
        timers  = [time[-1]]
    else:
        delay   = 0
    new_time = []
    new_E    = []
    for it in r_[0:len(time)]:
        if (E_released[it] != 0.0):
            if (E_av != 0.0):
                # Middle of an avalanche
                if peakmode:
                    E_av = max([E_av, E_released[it]])
                else:
                    E_av = E_av + E_released[it]
                delay = delay + dt
            else:
                # Beginning of an avalanche
                E_av = E_released[it]
                time_ini_av = time[it]
        else:
            if (E_av != 0.0): 
                # End of an avalanche
                new_E.append(E_av)
                delay = delay + dt
                new_time.append(time[it]-delay)
                E_av = 0.0
            else:
                # between avalanches
                new_E.append(0.0)
                new_time.append(time[it]-delay)
            if (len(new_time)>3):
                if (new_time[-2]<=timers[0])&(new_time[-1]>=timers[0]):
                    timers.append(it)
    if verbose:
        print('Delayed time from avalanches was',delay)

    if not InfoRedux is None:
        InfoRedux["delay"]=delay

    new_E=array(new_E)
    new_time=array(new_time)

    # Get rid of avalanches at start up or end of run
    if (ini_avalanche):
        new_E    = new_E[2:]
        new_time = new_time[2:]
    
    if (end_avalanche):
        new_E    = new_E[0:-1]
        new_time = new_time[0:-1]

    # Add threshold
    indx=where(new_E < threshold)
    new_E[indx] = 0.0

    return new_E,new_time

## Study power laws
def gene_av(Bstate,Niter=1e7,update=0):

    print('Generating avalanche data over ',Niter,' iterations')
    Bstate.do_soc(Niter=Niter,doplot=0,update=update)
    E_av = []
    T_av = []
    P_av = []
    WT_av= []
    start_av = 0
    iav = 0
    iw  = -1
    for i in r_[0:Niter]:
        if (Bstate.rel_e_tmp[i] > 0):
            if (start_av == 0):
                iw = iw + 1
                start_av = 1
                E_av.append(Bstate.rel_e_tmp[i])
                T_av.append(1)
                P_av.append(Bstate.rel_e_tmp[i])
            else:
                E_av[iav] = E_av[iav] + Bstate.rel_e_tmp[i]
                T_av[iav] = T_av[iav]+1
                P_av[iav] = max(P_av[iav],Bstate.rel_e_tmp[i])
        else:
            if (start_av == 1):
                start_av = 0
                iav=iav+1
                if (iw != -1):
                    WT_av.append(1)
            else:
                if (iw != -1):
                    WT_av[iw] = WT_av[iw]+ 1

    print('I counted',size(E_av),'avalanches for',Niter,'iterations')

    Bstate.E_av = array(E_av)
    Bstate.P_av = array(P_av)
    Bstate.T_av = array(T_av)
    Bstate.WT_av = array(WT_av)

def compact_av(Bstate,verbose=True,threshold=1e4):

    Niter=len(Bstate.rel_e_tmp)
    mytype = Bstate.rel_e_tmp.dtype
    if (verbose):
        print('Generating avalanche data over ',Niter,' iterations')
    E_av = []
    T_av = []
    P_av = []
    WT_av= []
    WT_large_av= []
    I_av = []
    start_av = 0
    iav = 0
    iw   = -1
    iwl  = -1
    for i in r_[0:Niter]:
        if (Bstate.rel_e_tmp[i] > 0):
            if (start_av == 0):
                start_av = 1
                E_av.append(Bstate.rel_e_tmp[i])
                I_av.append(i)
                T_av.append(1)
                P_av.append(Bstate.rel_e_tmp[i])
            else:
                E_av[iav] = E_av[iav] + Bstate.rel_e_tmp[i]
                T_av[iav] = T_av[iav]+1
                P_av[iav] = max(P_av[iav],Bstate.rel_e_tmp[i])
        else:
            if (start_av == 1):
                start_av = 0
                iav=iav+1
                iw = iw + 1
                WT_av.append(1)
                if (E_av[iav-1]>=threshold):
                    iwl=iwl+1
                    WT_large_av.append(1)
                elif (iwl != -1):
                    WT_large_av[iwl] = WT_large_av[iwl]+T_av[iav-1]
            else:
                if (iw != -1):
                    WT_av[iw] = WT_av[iw]+ 1
                if (iwl != -1):
                    WT_large_av[iwl] = WT_large_av[iwl]+1

    if (verbose):
        print('I counted',size(E_av),'avalanches for',Niter,'iterations')

    Bstate.E_av = array(E_av,dtype=mytype)
    Bstate.P_av = array(P_av,dtype=mytype)
    Bstate.T_av = array(T_av,dtype=mytype)
    Bstate.WT_av = array(WT_av,dtype=mytype)
    Bstate.WT_large_av = array(WT_large_av,dtype=mytype)
    Bstate.I_av = array(I_av,dtype=mytype)

def my_log_pdf(T_av,sub_bin=5,lw=1,color='k',normed=False):
    min_bin = floor(amin(log10(T_av)))
    max_bin = ceil(amax(log10(T_av)))
    print(max(T_av))
    mybins_T = exp(linspace(min_bin,max_bin,num=(max_bin-min_bin+1)*sub_bin)*log(10))
    pdf_T,mybins_T,patches=hist(T_av,bins=mybins_T,\
                                    histtype='step',lw=lw,color=color)
    if (normed):
        pdf_T=pdf_T/(size(T_av)*diff(mybins_T))

    return pdf_T,mybins_T

def study_dist(T_av,wt=False,sub_bin=5,slope_thresh=-0.8):

    min_bin = int(amin(log10(T_av)))-1
    max_bin = int(amax(log10(T_av)))+1
    mybins_T = exp(linspace(min_bin,max_bin,num=(max_bin-min_bin+1)*sub_bin)*log(10))
    pdf_T,mybins_T,patches=hist(T_av,bins=mybins_T,histtype='step')
    pdf_T=pdf_T/(size(T_av)*diff(mybins_T))

    if (wt): 
        cbins_T    = (mybins_T[1:]+mybins_T[0:-1])/2.0
        mypdf_T    = cp.copy(pdf_T)
        alpha_T    = 0.
        dist_fit_T = cp.copy(mypdf_T)
    else:
        cbins_T,mypdf_T=clean_bins(mybins_T,pdf_T)
        if (len(cbins_T) >= 2):
            alpha_T,dist_fit_T=new_optimize(cbins_T,mypdf_T,slope_thresh=slope_thresh)
        else:
            alpha_T =[1.0,amax(mybins_T)/2.]
            dist_fit_T=mypdf_T

    return pdf_T,mybins_T,alpha_T,cbins_T,mypdf_T,dist_fit_T

def plot_study_stat(pdf_E,mybins_E,alpha_E,cbins_E,mypdf_E,dist_fit_E,\
                        verbose=False,printalpha=True,fullplot=False,oplot=False):
    if (fullplot):
        step(mybins_E[0:-1],pdf_E,where='post')
    else:
        step(cbins_E,mypdf_E,where='mid')
    xscale('log')
    yscale('log')
    if (verbose):
        print('Alpha for E/e0 :',alpha_E[0])
    if not (oplot):
        plot(cbins_E,dist_fit_E)
        axvline(alpha_E[1],ls='--',color='k')
        title('f(E/e0), alpha= '+str(alpha_E[0]))
    if (printalpha):
        annotate("alpha ="+str(alpha_E[0])[0:4],xy=(0.1,0.03),xycoords='axes fraction') 

def clean_bins(mybins,pdf):
    nb = size(mybins)
    dbins=mybins[1]-mybins[0]
    cbins = (mybins[1:nb]+mybins[0:nb-1])/2.0

    cbins=cp.copy(cbins)
    mypdf=cp.copy(pdf)

    # Get rid of zero bins
    indx=where(mypdf != 0.0)[0]
    indexes = diff(indx)
    ind_indexes = where(indexes != 1)[0]
    if (len(ind_indexes) > 0):
        if (ind_indexes[0] != 0):
            ind_indexes=np.append([0],ind_indexes)
        if (ind_indexes[-1] != len(indx)-1):
            ind_indexes=np.append(ind_indexes,[len(indx)-1])
        sizes = diff(ind_indexes)
        ind_zero = where(sizes == amax(sizes))[0][0]
        indx = indx[ind_indexes[ind_zero]+1:ind_indexes[ind_zero+1]+1]

    cbins=cbins[indx]
    mypdf=mypdf[indx]

    if (len(cbins) < 2):
        print('HOY',indx)

    return cbins,mypdf

def find_populations(cbins,mypdf,slope_thresh=-0.8):
    #dpdf = deriv(append(diff(log10(cbins)),diff(log10(cbins))[-1]),log10(mypdf))
    dpdf = np.gradient(log10(mypdf),diff(log10(cbins))[0])
    indx2=where(dpdf <= slope_thresh)[0]

    indexes = diff(indx2)
    ind_indexes = where(indexes != 1)[0]
    if (len(ind_indexes) > 0):
        if (ind_indexes[0] != 0):
            ind_indexes=np.append([0],ind_indexes)
        if (ind_indexes[-1] != len(indx2)-1):
            ind_indexes=np.append(ind_indexes,[len(indx2)-1])
        sizes = diff(ind_indexes)
        ind_zero = where(sizes == amax(sizes))[0][0]
        indx_new = indx2[ind_indexes[ind_zero]+1:ind_indexes[ind_zero+1]+1]
    else:
        indx_new = indx2
        
    ind_end=indx_new

    #indx=where(dpdf > -0.6)[0]
    #istart = 1
    #iend = len(mypdf)
    #if (len(indx) == 0):
    #    ind_end = r_[0:len(mypdf)]
    #else:
    #    if (indx[0] == 0):
    #        if (len(indx) > 1):
    #            while ((istart+1 != len(indx)) and (indx[istart] == istart)):
    #                istart = istart +1
    #            ind_end = r_[indx[istart-1]:indx[istart]]
    #        else:
    #            ind_end = r_[0:len(mypdf)]
    #    else:
    #        ind_end = r_[0:indx[0]]

    return ind_end

def new_residuals(params, x, data):

    alpha = params['alpha'].value
    xcut  = params['xcuts'].value
    f0    = params['f0'].value
    model = myfunc([alpha,xcut],x,f0)
    return log10(data)-log10(model)

def new_optimize(cbins,mypdf,slope_thresh=-0.8):

    ind_end = find_populations(cbins,mypdf,slope_thresh=slope_thresh)

    if (len(ind_end) <= 2):
        new_pdf = copy(mypdf)
        new_cbins=copy(cbins)
    else:
        new_pdf   = copy(mypdf[ind_end])
        new_cbins = copy(cbins[ind_end])

    xc_max=amax(new_cbins)
    params = Parameters()
    params.add('alpha', value=2.0,min=1.001,max=1.9999)
    params.add('xcuts', value=amax(new_cbins)/2.,\
                   min=amax(new_cbins)/20.,max=xc_max)
    params.add('f0', value=new_pdf[0],vary=False)

    out = minimize(new_residuals, params, args=(new_cbins, new_pdf))
    alpha = [out.params['alpha'].value,out.params['xcuts'].value]
    dist_fit=myfunc(alpha,new_cbins,out.params['f0'].value)
    if (len(new_cbins) != len(cbins)):
        new_distfit = np.zeros(len(mypdf))
        new_distfit[ind_end] = dist_fit
    else:
        new_distfit=dist_fit
    return alpha,new_distfit

def myfunc(alpha,x,f0):
    return f0*((x/x[0])**(-alpha[0]))*exp(-x/alpha[1])
def linfunc(gamma,A,B):
    return log(B) - gamma*log(A)

def optimize_leastsq(cbins,mypdf,alpha_guess):
    alpha,cov,infodcit,mesg,ier = optimize.leastsq(\
        myresiduals, alpha_guess,args=(cbins,mypdf,mypdf[0]),full_output=True)
    dist_fit=myfunc(alpha,cbins,mypdf[0])
    return alpha,dist_fit

def myresiduals(alpha,x,f,f0):
    return log(f)-log(myfunc(alpha,x,f0))

# Gaussian fit
def residuals_gauss(params, x, data):
    x0     = params['x0'].value
    delta  = params['delta'].value
    A      = params['A'].value
    model  = func_gauss([x0,delta,A],x)
    return data-model
    #return log(data)-log(model)
def func_gauss(parameters,x):
    return parameters[2]*exp(-((x-parameters[0])/parameters[1])**2)
def fit_gaussian(cbins,mypdf):
    non_zero = where(mypdf != 0)[0]
    pdf = mypdf[non_zero]
    bins= cbins[non_zero]
    x0 = cbins[pdf.argmax()]
    params = Parameters()
    params.add('x0', value=x0)
    params.add('delta',value=x0)
    params.add('A', value=pdf.max())
    out = minimize(residuals_gauss, params, args=(bins, pdf))    
    dist_fit = func_gauss([params['x0'].value,\
                               params['delta'].value,\
                               params['A'].value],cbins)
    width_gauss = params['delta']*sqrt(2.0)
    maxi = params['x0'].value
    return dist_fit,[width_gauss,maxi,out.redchi]

# Weibull fit
def residuals_db_gauss(params, x, data):
    x0     = params['x0'].value
    delta  = params['delta'].value
    A      = params['A'].value
    model  = func_db_gauss([x0,delta,A],x)
    return data-model
def func_db_gauss(parameters,x):
    xx   = abs(x/parameters[1])
    powa = parameters[0]
    return parameters[2]*(xx)**(powa-1.0)*exp(-(xx**powa))
def fit_db_gaussian(cbins,mypdf):
    non_zero = where(mypdf != 0)[0]
    pdf = mypdf[non_zero]
    bins= cbins[non_zero]
    x0 = cbins[pdf.argmax()]
    params = Parameters()
    params.add('x0', value=2,min=1,max=10.0)
    params.add('delta',value=2.5,min=1.0)
    params.add('A', value=pdf.max(),min=0.,max=2000.)
    out = minimize(residuals_db_gauss, params, args=(bins, pdf))    
    dist_fit = func_db_gauss([params['x0'].value,\
                                  params['delta'].value,\
                                  params['A'].value],cbins)
    k = params['x0'].value
    lambd = params['delta'].value
    maxi  = lambd*((k-1.0)/k)**(1.0/k)
    width = params['delta'].value
    return dist_fit,[width,maxi,out.redchi]

# Lorentz fit
def residuals_lorentz(params, x, data):
    A      = params['A'].value
    x0     = params['x0'].value
    delta  = params['delta'].value
    model  = func_lorentz([x0,delta,A],x)
    return data-model
def func_lorentz(parameters,x):
    lor = parameters[2]/ \
        (1.0 + ((x-parameters[0])/parameters[1])**2)
    return lor
def fit_lorentz(cbins,mypdf):
    non_zero = where(mypdf != 0)[0]
    pdf = mypdf[non_zero]
    bins= cbins[non_zero]
    x0 = cbins[pdf.argmax()]
    params = Parameters()
    params.add('x0', value=x0)
    params.add('delta',value=x0)
    params.add('A',value=pdf.max())
    out = minimize(residuals_lorentz, params, args=(bins, pdf))    
    dist_fit = func_lorentz([params['x0'].value,\
                                 params['delta'].value,\
                                 params['A'].value],cbins)
    width_lor = params['delta'].value
    return dist_fit,width_lor

# Log-normal fit
def residuals_lognormal(params, x, data):
    A      = params['A'].value
    sigma  = params['sigma'].value
    mu     = params['mu'].value
    model  = func_lognormal([sigma,mu,A],x)
    return data-model
def func_lognormal(parameters,x):
    lognormal = (parameters[2]/(x/parameters[0]))* \
        exp(-(log(x/parameters[0])-parameters[1])**2)
    return lognormal
def fit_lognormal(cbins,mypdf):
    non_zeros = where(mypdf != 0.0)[0]
    pdf = mypdf[non_zeros]
    bins= cbins[non_zeros]
    params = Parameters()
    params.add('A',value = bins.max())
    params.add('sigma', value=1.)
    params.add('mu',log(bins[pdf.argmax()]))
    out = minimize(residuals_lognormal, params, args=(bins, pdf))    
    dist_fit = func_lognormal([params['sigma'].value,\
                                   params['mu'].value,\
                                   params['A'].value],cbins)
    width = params['sigma'].value
    return dist_fit,width

def residuals_square(params, x, data):
    x0     = params['x0'].value
    D0     = params['D0'].value
    model  = func_square([x0,D0],x)
    return log10(data)-log10(model)
def func_square(parameters,x):
    return parameters[0] + parameters[1]*x**2
    #return parameters[0]*exp(x/parameters[1])
def fit_square(cbins,mypdf):
    x0 = cbins[mypdf[0]]
    params = Parameters()
    params.add('x0', value=x0)
    params.add('D0',value=1.0)
    out = minimize(residuals_square, params,args=(cbins, mypdf))
    fit_pars = [params['x0'].value,params['D0'].value]
    dist_fit = func_square(fit_pars,cbins)
    return dist_fit,fit_pars

def residuals_pl(params, x, data):
    x0     = params['x0'].value
    D0     = params['D0'].value
    expo   = params['expo'].value
    model  = func_pl([x0,D0,expo],x)
    return log10(data)-log10(model)
def func_pl(parameters,x):
    #return parameters[0] + (x)**parameters[1]
    return parameters[0] + (x/parameters[1])**parameters[2]
    #return parameters[0]*exp(x/parameters[1])
def fit_pl(cbins,mypdf):
    x0 = cbins[mypdf[0]]
    params = Parameters()
    params.add('x0', value=x0)
    params.add('D0',value=1.0)
    params.add('expo',value=2.0)
    out = minimize(residuals_pl, params,args=(cbins, mypdf))
    #fit_pars = [params['x0'].value,\
    fit_pars = [params['x0'].value,params['D0'].value,\
                    params['expo'].value]
    dist_fit = func_pl(fit_pars,cbins)
    return dist_fit,fit_pars
                
## Visualy show avalanches
def see_av(Bstate,nstep=10*4):

    fig=figure()
    Ztmp=0*Bstate.Z
    Ztmp[where(abs(Bstate.Z)>Bstate.Zc)] = 1
    p=pcolormesh(Ztmp)
    axis('tight')
    p.set_clim([0,1])
    iav=0
    stop = 0
    while (stop != 1):
        if (stop != 1):
            Bstate.do_soc(Niter=1,doplot=0,update=1,i_idum=-1,finish_with_soc=False)
            Ztmp[where(abs(Bstate.Z_tmp)>Bstate.Zc)] = 1
            if (sum(abs(Bstate.Z_tmp)>Bstate.Zc) >= 1):
                if (iav == 0):
                    print('Beginning of avalanche!')
                p.set_array(Ztmp.ravel())
                p.set_clim([0,1])
                if (iav == 0):
                    colorbar()
                draw()
                time.sleep(0.01)
                iav = iav+1
            else:
                if (iav != 0):
                    print('End of avalanche!')
                    stop = 1

def draw_patch(ax,patch):
    patch.set_clip_box(ax.bbox)
    ax.add_patch(patch)

def see_model(B,nstep=100,show_Z=False,other_idum=False,simple_ini=False):

    bdir = '/Users/rbarnabe/These/Avalanches/avsf/Resultats_avsf/'
    figdir = bdir+'Movie/'

    # Get suitable initial conditions
    if (not simple_ini):
        binits = read_elements(2,bdir+"Cases/LH_inits")
        B.last_idum = -binits[0]#-2147483647
        B.B = cp.copy(binits[1][0])
        B.do_soc(Niter=3250,doplot=0,update=0,i_idum=-1,finish_with_soc=False)
        compact_av(B,threshold=0.,verbose=False)
        time_happ = int(B.I_av[B.E_av.argmax()])
        l_happ = int(B.T_av[B.E_av.argmax()])
        print('It happens at',time_happ,'and dured',l_happ)
        B.do_soc(Niter=time_happ-120,doplot=0,update=1,i_idum=-1,finish_with_soc=False)
    else:
        B.do_soc(Niter=1e5,doplot=0,update=0,i_idum=0,finish_with_soc=False)
        compact_av(B,threshold=0.,verbose=False)
        time_happ = int(B.I_av[B.E_av.argmax()])
        l_happ = int(B.T_av[B.E_av.argmax()])
        print('It happens at',time_happ,'and dured',l_happ,'with E=',B.E_av.max())
        B.do_soc(Niter=time_happ-10,doplot=0,update=1,i_idum=0,finish_with_soc=False)

    cmap = matplotlib.colors.ListedColormap(['w','k','r','b','m','c','g','y'])

    load_j('solar physics')
    Ztmp=np.zeros(np.shape(B.Z))
    Ztmp[where(abs(B.Z)>B.Zc)] = 1
    fig=figure(figsize=(8,3))
    subplot(121)
    if (show_Z):
        p1=pcolormesh(np.arange(B.Nx+1),np.arange(B.Ny+1),np.abs(B.Z)/B.Zc,vmin=0,vmax=1.0)
    else:
        p1=pcolormesh(np.arange(B.Nx+1),np.arange(B.Ny+1),B.B)
    ax1=gca()
    axis('equal') ; axis('tight')
    subplot(122)
    p2=pcolormesh(np.arange(B.Nx+1),np.arange(B.Ny+1),Ztmp,cmap=cmap,vmin=0,vmax=7)
    ax2=gca()
    axis('equal'); axis('tight')
    Bold = cp.copy(B.B)
    
    if other_idum:
        figdir=figdir+'Oidum/'
        B.last_idum = B.last_idum-2
    if (show_Z):
        figdir=figdir+'Z3/'
    call('mkdir -p '+figdir,shell=True)
    bname = 'movie_' 
    for i in range(l_happ+10):
        B.do_soc(Niter=1,doplot=0,update=1,i_idum=-1,finish_with_soc=False)
        Ztmp[where(abs(B.Z)>B.Zc)] += 1
        clf()
        subplot(121)
        if (show_Z):
            p1=pcolormesh(np.arange(B.Nx+1),np.arange(B.Ny+1),np.abs(B.Z)/B.Zc,vmin=0,vmax=1.0)
        else:
            p1=pcolormesh(np.arange(B.Nx+1),np.arange(B.Ny+1),B.B)
        ax1=gca()
        axis('equal') ; axis('tight')
        subplot(122)
        p2=pcolormesh(np.arange(B.Nx+1),np.arange(B.Ny+1),Ztmp,cmap=cmap,vmin=0,vmax=7)
        ax2=gca()
        axis('equal'); axis('tight')
        if (sum(abs(B.Z_tmp)>B.Zc) >= 1):
            ax1.set_title('')
            ax2.set_title('Avalanching...')
            Bold = cp.copy(B.B)
        else:
            if (B.eps_drive <= 0):
                [indx],[indy]=np.where(B.B-Bold != 0.)
                cc = Circle((indy,indx),1.0,ec='w',fc='None',lw=3)
                draw_patch(ax1,cc)
                ax1.set_title('Forcing randomly...')
                Bold = cp.copy(B.B)
            else:
                ax1.set_title('Forcing...')
            ax2.set_title('')
            Ztmp[:,:] = 0.
        draw()
        fname = bname+"%05i" % (i)
        savefig(figdir+fname+'.pdf')

def see_model_v2(B,simple_ini=False,output_dir='Movie',i_idum=0):

    bdir = '/Users/astrugar/WORK/Astro/ModelesReduits/Avalanches/AVSF/'
    figdir = bdir+'figs/'+output_dir+'/'

    # Get suitable initial conditions
    if (not simple_ini):
        binits = read_elements(2,bdir+"Cases/LH_inits")
        B.last_idum = -binits[0]#-2147483647
        B.B = cp.copy(binits[1][0])
        B.do_soc(Niter=3250,doplot=0,update=0,i_idum=-1,finish_with_soc=False)
        compact_av(B,threshold=0.,verbose=False)
        time_happ = int(B.I_av[B.E_av.argmax()])
        l_happ = int(B.T_av[B.E_av.argmax()])
        print('It happens at',time_happ,'and dured',l_happ)
        B.do_soc(Niter=time_happ-120,doplot=0,update=1,i_idum=-1,finish_with_soc=False)
    else:
        B.do_soc(Niter=1e3,doplot=0,update=0,i_idum=0,finish_with_soc=False)
        compact_av(B,threshold=0.,verbose=False)
        time_happ = int(B.I_av[B.E_av.argmax()])
        l_happ = int(B.T_av[B.E_av.argmax()])
        print('It happens at',time_happ,'and dured',l_happ,'with E=',B.E_av.max())
        B.do_soc(Niter=time_happ-50,doplot=0,update=1,i_idum=0,finish_with_soc=False)

    cmap = matplotlib.colors.ListedColormap(['w','k','r','b','m','c','g','y'])
    zmap = 2.
    
    load_j('')
    fig=figure(figsize=(8,3))
    X,Y=np.meshgrid(np.arange(0.,1.,1./B.Nx),np.arange(0.,1.,1./B.Ny))
    ax1=fig.add_subplot(1,2,1,projection='3d')
    surf=ax1.plot_surface(X,Y,B.Z/B.Zc,cmap='viridis',linewidth=0,antialiased=False,vmin=0.,vmax=1.)
    #cset = ax1.contourf(X, Y, B.Z/B.Zc, zdir='z', offset=zmap, cmap='viridis',vmin=0.,vmax=1.,antialiased=False)
    ax1.set_zlim([0.,zmap+0.1])
    ax1.set_aspect('equal')
    # Colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5,label='Curvature')

    subplot(122)
    Ztmp=np.zeros(np.shape(B.Z))
    Ztmp[where(abs(B.Z)>B.Zc)] = 1
    ax2=gca()
    p2=ax2.pcolorfast(np.arange(B.Nx+1)/float(B.Nx),np.arange(B.Ny+1)/float(B.Ny),Ztmp,cmap=cmap,vmin=0,vmax=7)
    ax2.set_aspect('equal')
    Bold = cp.copy(B.B)
    
    call('mkdir -p '+figdir,shell=True)

    # Now do the movie
    bname = 'movie_' 
    for i in range(l_happ+10):
        if i == 0 and i_idum != 0:
            B.do_soc(Niter=1,doplot=0,update=1,i_idum=i_idum,finish_with_soc=False)
        else:
            B.do_soc(Niter=1,doplot=0,update=1,i_idum=-1,finish_with_soc=False)
        Ztmp[where(abs(B.Z)>B.Zc)] += 1
        fig.clf()
        ax1=fig.add_subplot(1,2,1,projection='3d')
        surf=ax1.plot_surface(X,Y,B.Z/B.Zc,cmap='viridis',linewidth=0,antialiased=False,vmin=0.,vmax=1.)
        #cset = ax1.contourf(X, Y, B.Z/B.Zc, zdir='z', offset=zmap, cmap='viridis',vmin=0.,vmax=1.)
        ax1.set_zlim([0.,zmap+0.1])
        ax1.set_aspect('equal')
        # Colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5,label='Curvature')

        subplot(122)
        ax2=gca()
        p2=ax2.pcolorfast(np.arange(B.Nx+1)/float(B.Nx),np.arange(B.Ny+1)/float(B.Ny),Ztmp,cmap=cmap,vmin=0,vmax=7)
        ax2.set_aspect('equal')
        
        if (sum(abs(B.Z_tmp)>B.Zc) >= 1):
            ax1.set_title('')
            ax2.set_title('Avalanching...')
            Bold = cp.copy(B.B)
        else:
            if (B.eps_drive <= 0):
                [indx],[indy]=np.where(B.B-Bold != 0.)
                ax1.quiver([X[indx,indy]],[Y[indx,indy]],[zmap],[0],[0],[-1],color='r',length=0.5,arrow_length_ratio=0.2)
                ax1.set_title('Forcing randomly...')
                Bold = cp.copy(B.B)
            else:
                ax1.set_title('Forcing...')
            ax2.set_title('')
            Ztmp[:,:] = 0.
        draw()
        fname = bname+"%04i" % (i)
        #fig.tight_layout()
        fig.savefig(figdir+fname+'.png')
        
    #iav=0
    #stop = 0
    #while (stop != 1):
    #    if (stop != 1):
    #        Bstate.do_soc(Niter=1,doplot=0,update=1,i_idum=-1,finish_with_soc=False)
    #        Ztmp[where(abs(Bstate.Z_tmp)>Bstate.Zc)] = 1
    #        if (sum(abs(Bstate.Z_tmp)>Bstate.Zc) >= 1):
    #            if (iav == 0):
    #                print 'Beginning of avalanche!'
    #            p.set_array(Ztmp.ravel())
    #            p.set_clim([0,1])
    #            if (iav == 0):
    #                colorbar()
    #            draw()
    #            time.sleep(0.01)
    #            iav = iav+1
    #        else:
    #            if (iav != 0):
    #                print 'End of avalanche!'
    #                stop = 1



def study_statistics(Bcase,Niter=1e6,update=0,psf='test2.pdf',doplot=True,fullplot=False,finish_with_soc=True,\
                         v2=False,sub_bin=5):

    if Niter != -1:
        Bcase.do_soc(Niter=Niter,update=update,doplot=0,finish_with_soc=finish_with_soc)
    else:
        print("Using previously computed model...")
    if (v2):
        Bcase.rel_e_tmp = Bcase.rel_e2
    compact_av(Bcase,threshold=0.0,verbose=False)       

    if (doplot):

        fig = figure(figsize=(15,4))        
        fig.add_subplot(1,4,1)
        pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.E_av,sub_bin=sub_bin)                     
        Bcase.alphaE=alphaE[0]
        plot_study_stat(pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE,fullplot=fullplot)              
        title("Energy released")                     
        annotate("Nb av ="+str(size(Bcase.E_av)),xy=(0.1,0.1),xycoords='axes fraction')   
        subplots_adjust(left=0.05,right=0.95)                                            
        
        fig.add_subplot(1,4,2)
        pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.T_av,sub_bin=sub_bin) 
        Bcase.alphaT=alphaE[0]
        plot_study_stat(pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE,fullplot=fullplot)
        title("Avalanche time")                     
        
        fig.add_subplot(1,4,3)
        pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.P_av,sub_bin=sub_bin)                     
        Bcase.alphaP=alphaE[0]
        plot_study_stat(pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE,fullplot=fullplot)
        title("Avalanche peak")                     

        #fig.add_subplot(1,4,4)        
        #hold(False)                                                                       
        #pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.WT_av,wt=True)    
        #alphaE = [0,0]
        #plot_study_stat(pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE,fullplot=fullplot)
        #title("Waiting time")                     

        #!rox : study_dist(wt=True) -> Ne fonctionne pas
        fig.add_subplot(1,4,4)
        pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.WT_av,sub_bin=sub_bin)    
        Bcase.alphaE = alphaE[0]
        plot_study_stat(pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE,fullplot=fullplot)
        title("Waiting time") 
        
        #fig.add_subplot(2,3,5)
        #scatter(Bcase.P_av,Bcase.E_av,facecolors='none')
        #xscale('log')
        #yscale('log')
        #xlabel('P')
        #xlim([amin(Bcase.P_av),amax(Bcase.P_av)])
        #ylabel('E')
        #ylim([amin(Bcase.E_av),amax(Bcase.E_av)])
        #hold(True)
        #plot(Bcase.P_av,Bcase.P_av**gamma_PE[0][0],color='r',lw=3)
        #plot([1.,100.],[100.,1.],color='k',ls='--')
        #hold(False)
        #title('Gamma PE='+str(gamma_PE[0][0])[0:4])
        #
        #fig.add_subplot(2,3,6)
        #scatter(Bcase.T_av,Bcase.P_av,facecolors='none')
        #xscale('log')
        #yscale('log')
        #xlabel('T')
        #xlim([amin(Bcase.T_av),amax(Bcase.T_av)])
        #ylabel('P')
        #ylim([amin(Bcase.P_av),amax(Bcase.P_av)])
        #hold(True)
        #plot(Bcase.T_av,Bcase.T_av**gamma_TP[0][0],color='r',lw=3)
        #plot([1.,100.],[100.,1.],color='k',ls='--')
        #hold(False)
        #title('Gamma TP='+str(gamma_TP[0][0])[0:4])

        savefile = '/Users/rbarnabe/These/Avalanches/avsf/Resultats_avsf/PDF/'+psf
        if (psf != 'None'):
            savefig(savefile,transparent=True)

    else:
        pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.E_av) 
        Bcase.alphaE=alphaE[0]
        pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.T_av)   
        Bcase.alphaT=alphaE[0]
        pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.P_av)       
        Bcase.alphaP=alphaE[0]
        pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.WT_av)           

def plot_Estats(Bcase,Niter=int(1e6),update=0,psf='Estats.pdf',finish_with_soc=False,sub_bin=5):
    Bcase.do_soc(Niter=Niter,update=update,doplot=0,finish_with_soc=finish_with_soc)
    compact_av(Bcase,threshold=0.0,verbose=False)       

    load_j('solar physics')
    fig = figure(figsize=(4,4))
    pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(Bcase.E_av,sub_bin=sub_bin)                     
    Bcase.alphaE=alphaE[0]
    plot_study_stat(pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE,fullplot=True)              
    title("Energy released")                     
    annotate("Nb av ="+str(size(Bcase.E_av)),xy=(0.1,0.1),xycoords='axes fraction')   
    #subplots_adjust(left=0.05,right=0.95)
    show()
    savefig(psf,transparent=True)

def study_time_distribution(B48,psfile=''):
    # for the moment use the previous run
    # and assume the study_statistics has been run before
    if not hasattr(B48,'E_av'):
        print("Run study_statistics first please")
        return
    
    obs_pic,time_pic = condense_time_serie(B48.rel_e_tmp,B48.time_tmp,threshold=0.0)
    pdfE,binsE,alphaE,cbinsE,mypdfE,distfitE = study_dist(B48.E_av)                     
    B48.cbins = cp.copy(cbinsE)
    B48.deltat = 0.0*B48.cbins
    for ii,energy in enumerate(B48.cbins):
        obs = cp.copy(obs_pic)
        time= cp.copy(time_pic)
        obs[where(obs_pic < energy)] = 0.0
        nb_pic,E_pic,t_pic = nb_peaks(obs,time)
        B48.deltat[ii] = mean(diff(t_pic))

    fig=figure()
    loglog(B48.cbins,B48.deltat)
    title('Typical waiting time between avalanches larger than x')
    ylabel('Iterations')
    xlabel('Avalanche energy')
    grid(True)
    if (psfile != ''):
        savefig(psfile,transparent=True)

## Tests...
def copy_to_disk(H,name='data',method='Pickle',state_only=0):
    if (state_only == 1):
        mylist = cp.copy(H.state_to_save)
        filename=name+'_state'
    else:
        mylist = cp.copy(H.to_save)
        filename=name
    if (method == 'Pickle'):
        f=file(filename+'.bin','w')
        for str_var in mylist:
            if (hasattr(H,str_var)):
                #print 'saving',str_var
                exec("cPickle.dump(H.%s,f,1)" %(str_var))

    elif(method == 'h5'):
        print('Waiting the install')
        #f = h5py.File(filename+'.hdf5',"w")
        #for str_var in H.to_save_h5:
        #    if (hasattr(H,str_var)):
        #        exec "dset = f.create_dataset('%s',shape(H.%s))" %(str_var,str_var)
        #        exec "dset[...] = H.%s" %(str_var)
    else:
        print("Method not known")

    f.close()

def read_from_disk(name='data',method='Pickle',state_only=0,verbose=False):
    H=av_model(100,Niter=1,doplot=0,verbose=verbose,fromscratch=False)
    if (state_only == 1):
        mylist = cp.copy(H.state_to_save)
        filename=name+'_state'
    else:
        mylist = cp.copy(H.to_save)
        filename=name
    if (method == 'Pickle'):
        f=file(filename+'.bin','r')
        for str_var in mylist:
            tmp = cPickle.load(f)
            exec("H.%s = tmp" %(str_var))
        f.close()
    elif(method == 'h5'):
        print('Waiting the install')
        #Htmp = loadHDF5(name+'.hdf5')
        #for str_var in Htmp.keys:
        #    exec "H.%s = Htmp.%s" %(str_var,str_var)
    else:
        print("Method not known")

    return H

class null_class():
    def __init__(self):
        # nothing
        a=[]

#--------------------------------------------------
# Load an HDF5 file
#--------------------------------------------------
#class loadHDF5():
#    def __init__(self,filename):
#        fh5       = h5py.File(filename,'r')
#        var_in_h5 = fh5.keys()
#        self.keys = var_in_h5
#        for str_var in var_in_h5:
#            exec "%s = fh5['%s']" %(str_var,str_var)
#            exec "sh = shape(%s)" %(str_var)
#            if (sh != ()):
#                exec "self.%s = %s[:].reshape(%s)" %(str_var,str_var,sh)
#            else:
#                exec "self.%s = %s[()]" %(str_var,str_var)
#        fh5.close()
        
def save_elements(list_e,casename):
    f=open(casename+'.bin','wb')
    for el in list_e:
        cPickle.dump(el,f)
    f.close()

def read_elements(nb_el,casename):
    f=open(casename+'.bin', mode='rb')
    list_el = []
    for el in range(nb_el):
        list_el.append(cPickle.load(f))
    f.close()
    return list_el

def save_state(BB,casename):
    save_elements([BB.SOC_case,BB.eps_drive,BB.lh_soc,\
                   BB.D_nc,BB.sigma1,BB.sigma2,BB.B,\
                   BB.Nx,BB.Ny,BB.Zc, BB.name],\
                  casename+'_state')
def read_state(casename,verbose=False):
    BB=av_model(100,Niter=1,doplot=0,verbose=verbose,fromscratch=True)
    if os.path.exists(casename+'_state.bin'):
        [BB.SOC_case,BB.eps_drive,BB.lh_soc,\
         BB.D_nc,BB.sigma1,BB.sigma2,BB.B,\
         BB.Nx,BB.Ny,BB.Zc, BB.name]=read_elements(11,casename+'_state')
    else:
        print("No state file found")
    try:
        BB.dir_saved='/'.join(casename.split('/')[:-1])+'/'
    except:
        BB.dir_saved='./'
    return BB

def study_predictability(cases_dir,casename,figfilename,nthresh=0,mytit='',tw=1e3,e0=1e4,secondary_peaks=True,plot_labels=True):
    
    # read the case
    loc_dir = cases_dir
    savename=loc_dir+casename+'_all'
    [list_fullT,list_fullE,\
         list_thresh,nb_runs,ncases,e_large]=read_elements(6,savename)

    if (nthresh == 0):
        nthresh = len(list_thresh)

    proba_time   = np.zeros((nthresh,3,nb_runs))
    pred_time    = np.zeros((nthresh,4,nb_runs))
    proba_energy = np.zeros((nthresh,4,nb_runs))
    pred_energy  = np.zeros((nthresh,4,nb_runs))
    err_energy   = np.zeros((nthresh,4,nb_runs))
    err_time     = np.zeros((nthresh,nb_runs))

    # Do the analysis and plot it
    for inb,list_alltpics in enumerate(list_fullT):

        list_alltpics = list_alltpics[0:nthresh]
        list_allepics = list_fullE[inb][0:nthresh]

        #  Remove empty lists
        if ([] in list_alltpics):
            list_alltpics.remove([])
            list_allepics.remove([])

        figfile=figfilename+"_"+str(inb)
        nxp=nthresh
        fig=figure(figsize=(6,3*nthresh))
        
        max_t=amax(array(list_alltpics[0]))
        for ii,list_tpic in enumerate(list_alltpics):
            if not (list_tpic == []):
                max_t=max(max_t,amax(array(list_tpic)))
        max_e=amax(array(list_allepics[0]))/e0
        min_e=amin(array(list_allepics[0]))/e0
        for ii,list_epic in enumerate(list_allepics):
            if not (list_epic == []):
                max_e=max(max_e,amax(array(list_epic))/e0)
                min_e=min(min_e,amin(array(list_epic))/e0)

        # Do the plots
        ip = 1
        for ii,list_tpic in enumerate(list_alltpics):

            ax=subplot(nxp,2,ip)
            nbins = 20
            mybins = linspace(0.0,1.0,nbins+1)
            pdf,bins,patches=hist(array(list_tpic),histtype='step',\
                                      bins=mybins,lw=3,color='k')
            imaxt1 = (pdf == amax(pdf))
            cbins = (bins[0:-1]+bins[1:])/2. ; #dist_fit,g_width=fit_gaussian(cbins,pdf)
            #pred_time[ii,0,inb] = cbins[dist_fit.argmax()]
            pred_time[ii,1,inb] = cbins[pdf.argmax()]

            proba_time[ii,0,inb] = float(amax(pdf))/ncases
            maxpdf = pdf.max()
            dist = 0.
            for mypdf in pdf:
                if (mypdf != maxpdf):
                    dist = dist + abs(maxpdf-mypdf)/(len(pdf)-1.0)
            proba_time[ii,0,inb] = dist/maxpdf

            #err_time[ii,inb]  = g_width[1]
            while (size(pdf[imaxt1]) > 1):
                for i,im in enumerate(imaxt1):
                    if (im):
                        imaxt1[i]=False
                        pdf[i]=0.
                        break
            pdf[imaxt1]=0.
            imaxt2 = (pdf == amax(pdf))
            proba_time[ii,1,inb] = float(amax(pdf))/ncases
            pred_time[ii,2,inb] = cbins[pdf.argmax()]
            while (size(pdf[imaxt2]) > 1):
                for i,im in enumerate(imaxt2):
                    if (im):
                        imaxt2[i]=False
                        pdf[i]=0.0
                        break
            pdf[imaxt2]=0.
            imaxt3 = (pdf == amax(pdf))
            proba_time[ii,2,inb] = float(amax(pdf))/ncases
            pred_time[ii,3,inb] = cbins[pdf.argmax()]
            while (size(pdf[imaxt3]) > 1):
                for i,im in enumerate(imaxt3):
                    if (im):
                        imaxt3[i]=False
                        pdf[i]=0.
                        break
            dbins=bins[1]-bins[0]
            indx1 = (array(list_tpic) <= bins[imaxt1]+dbins)&(array(list_tpic) >= bins[imaxt1])
            indx2 = (array(list_tpic) <= bins[imaxt2]+dbins)&(array(list_tpic) >= bins[imaxt2])
            indx3 = (array(list_tpic) <= bins[imaxt3]+dbins)&(array(list_tpic) >= bins[imaxt3])
            axvline(bins[imaxt1]+dbins/2,ls='--',color='r')
            if (secondary_peaks):
                axvline(bins[imaxt2]+dbins/2,ls='--',color='b')
                axvline(bins[imaxt3]+dbins/2,ls='--',color='g')
            #hold(True);plot(cbins,dist_fit,ls='--',color='k',lw=2);hold(False)
            #xlim([0.,max_t])
            xlim([0.,1.0])
            if (ii == nthresh-1):
                if (plot_labels):
                    xlabel("Iteration/time window")
                ylabel("PDF")
                #ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
            else:
                setp(ax.get_xticklabels(),visible=False)

            # Now the energies
            frac_e = 0.5
            list_epic=list_allepics[ii]

            nbins=20
            ax=subplot(nxp,2,ip+1)
            pdf,bins=my_log_pdf(array(list_epic)/e0,\
                                    sub_bin=nbins,lw=1,color='k')
            cbins = (bins[0:-1]+bins[1:])/2. 
            step(cbins,pdf,where='mid',lw=3,\
                     color='k',zorder=2)

            # Try to match a gaussian
            dist_fit_gauss,res_gauss=fit_gaussian(cbins,pdf)
            dist_fit_db,res_weibull=fit_db_gaussian(cbins,pdf)
            if (res_gauss[2] > res_weibull[2]):
                dist_fit    = dist_fit_db 
                g_width     = res_weibull[0]
                predicted_e = res_weibull[1]
                col_fit     = 'r'
                dist_fit_2  = dist_fit_gauss
                col_fit_2   = 'b'
                leglab = ['Weibull','Gaussian']
            else:
                dist_fit    = dist_fit_gauss 
                g_width     = res_gauss[0]
                predicted_e = res_gauss[1]
                col_fit     = 'b'
                dist_fit_2  = dist_fit_db
                col_fit_2   = 'r'
                leglab = ['Gaussian','Weibull']
            hold(True)
            ll1,=plot(cbins,dist_fit,color=col_fit,lw=2,\
                     marker='o',mec=col_fit,mfc='w',\
                     markevery=10,zorder=2)
            
            ll2,=plot(cbins,dist_fit_2,color=col_fit_2,lw=1,\
                     ls='--',zorder=2)
            hold(False)
            dist_fit=dist_fit_db
            pred_energy[ii,0,inb] = predicted_e
            err_energy[ii,0,inb]  = g_width
            if (cbins[dist_fit.argmax()] >= e_large*frac_e/e0):
                proba_energy[ii,0,inb] = sum(dist_fit)/ncases
            else:
                proba_energy[ii,0,inb] = 0.0
            # plot the rest
            if (secondary_peaks):
                hold(True)
                pdf,bins=my_log_pdf(array(list_epic)[indx1]/e0,sub_bin=nbins,lw=1,color='r')
                cbins = (bins[0:-1]+bins[1:])/2. ; dist_fit,g_width=fit_gaussian(cbins,pdf)
                indx_tmp = where(cbins >= list_thresh[ii]/e0) ; cbins = cbins[indx_tmp] 
                dist_fit=dist_fit[indx_tmp]
                pred_energy[ii,1,inb] = cbins[dist_fit.argmax()]
                err_energy[ii,1,inb]  = g_width[1]
                plot(cbins,dist_fit,ls='--',color='r',lw=2)
                if (cbins[dist_fit.argmax()] >= e_large*frac_e/e0):
                    proba_energy[ii,1,inb] = sum(dist_fit)/ncases
                else:
                    proba_energy[ii,1,inb] = 0.0
                if (len(array(list_epic)[indx2]) != 0):
                    pdf,bins=my_log_pdf(array(list_epic)[indx2]/e0,sub_bin=nbins,lw=1,color='b')
                    cbins = (bins[0:-1]+bins[1:])/2. ; dist_fit,g_width=fit_gaussian(cbins,pdf)
                    indx_tmp = where(cbins >= list_thresh[ii]/e0) ; cbins = cbins[indx_tmp] 
                    dist_fit=dist_fit[indx_tmp]
                    plot(cbins,dist_fit,ls='--',color='b',lw=2)
                    pred_energy[ii,2,inb] = cbins[dist_fit.argmax()]
                    err_energy[ii,1,inb]  = g_width[1]
                    if (cbins[dist_fit.argmax()] >= e_large*frac_e/e0):
                        proba_energy[ii,2,inb] = sum(dist_fit)/ncases
                    else:
                        proba_energy[ii,2,inb] = 0.0
                if (len(array(list_epic)[indx3]) != 0):
                    pdf,bins=my_log_pdf(array(list_epic)[indx3]/e0,sub_bin=nbins,lw=1,color='g')
                    cbins = (bins[0:-1]+bins[1:])/2. ; dist_fit,g_width=fit_gaussian(cbins,pdf)
                    indx_tmp = where(cbins >= list_thresh[ii]/e0) ; cbins = cbins[indx_tmp] 
                    dist_fit=dist_fit[indx_tmp]
                    plot(cbins,dist_fit,ls='--',color='g',lw=2)
                    pred_energy[ii,3,inb] = cbins[dist_fit.argmax()]
                    err_energy[ii,3,inb]  = g_width[1]
                    if (cbins[dist_fit.argmax()] >= e_large*frac_e/e0):
                        proba_energy[ii,3,inb] = sum(dist_fit)/ncases
                    else:
                        proba_energy[ii,3,inb] = 0.0
                hold(False)
            if (ii == nthresh-1):
                if (plot_labels):
                    xlabel("Released energy")
            else:
                setp(ax.get_xticklabels(),visible=False)
            #axvline(list_thresh[ii],ls='--',color='k')
            #annotate("Threshold = %d" % (list_thresh[ii]),xy=(0.5,0.8),\
            #             xycoords='axes fraction')
            axvline(e_large/e0,ls='--',lw=2,color='k',\
                        alpha=0.5,zorder=1)
            axvline(1.0,lw=2,color='k',\
                        alpha=0.5,zorder=1)
            xlim([8e-2,2e1])
            #xlim([min_e,max_e])
            #ax.set_xscale('log')
            ip = ip + 2
        
            if (amax(list_tpic) == 0.0):
                # not good, seems we are in the middle of an avalanche or sthg
                proba_time[ii,:,inb]   = 0.0
                proba_energy[ii,:,inb] = 0.0
                pred_energy[ii,:,inb]  = 0.0
        annotate(mytit,xy=(0.85,0.9),\
                     xycoords='axes fraction',color='k')
        legend([ll1,ll2],leglab,\
                   loc='center right',\
                   prop={'size':11}).draw_frame(0)
        subplots_adjust(bottom=0.2)
        savefig(figfile+".pdf",transparent=True,bbox_inches='tight')
    
    # Save the study
    #call("mkdir -p "+cases_dir+'Preds/',shell=True)
    savename=loc_dir+casename
    save_elements([proba_time,pred_time,err_time,\
                       proba_energy,pred_energy,err_energy,\
                       list_thresh,nb_runs,ncases],savename)
    #plot_predictability(savename,psfile=figdir+'stats2.png')

def plot_predictability(name,psfile=''):

    proba_time,pred_time,err_time,proba_energy,pred_energy,err_energy,\
        list_thresh,nb_runs,ncases = read_elements(9,name)

    fig=figure(figsize=(10,10))
    ax=subplot(2,2,1)
    scatter(proba_time[:,0,:],proba_energy[:,0,:],facecolors='none')
    xlim([0,1]);ylim([0,1])
    title('full E')
    ylabel('Energy')
    grid(True)
    ax=subplot(2,2,2)
    scatter(proba_time[:,0,:],proba_energy[:,1,:],facecolors='none')
    xlim([0,1]);ylim([0,1])
    title('Most probable')
    grid(True)
    ax=subplot(2,2,3)
    scatter(proba_time[:,1,:],proba_energy[:,2,:],facecolors='none')
    xlim([0,1]);ylim([0,1])
    title('Second probable')
    xlabel('Time');ylabel('Energy')
    grid(True)
    ax=subplot(2,2,4)
    scatter(proba_time[:,2,:],proba_energy[:,3,:],facecolors='none')
    xlim([0,1]);ylim([0,1])
    title('Third probable')
    xlabel('Time')
    grid(True)

    if (psfile != ''):
        savefig(psfile,transparent=True)


def mkobs2(realdata,binsize=100):

    obs=zeros(len(realdata))
    index=0
    for i in r_[0:len(realdata)]/binsize:
        summ = sum(realdata[index*binsize:(index+1)*binsize])
        obs[index*binsize:(index+1)*binsize]=summ/binsize
        index=index+1

    return obs

def nb_peaks(E_seq,time_seq):
    indx=where(E_seq > 0.0)
    E_pics = cp.copy(E_seq[indx])
    T_pics = cp.copy(time_seq[indx])
    nb_pics = size(E_pics)
    return nb_pics,E_pics,T_pics

def stat_peaks(Bcase):
    # Perform statistics over the number of "flares" obtained for a certain 
    # time window of assimilation, after binning and time compression 
    min_bin = 2.5
    max_bin = 3.5
    sub_bin = 5
    Niters  = exp(linspace(min_bin,max_bin,num=(max_bin-min_bin+1)*sub_bin)*log(10))
    n_niters=len(Niters)
    n_idum  = 50
    nb_pic  = zeros((n_idum,n_niters))
    time_sim= zeros((n_idum,n_niters))
    nb_iter = zeros((n_idum,n_niters))
    for i_idum in r_[0:n_idum]:
        for j_Niter in r_[0:n_niters]:
            Bcase.mkobs(Niter=int(Niters[j_Niter]),doplot=0,update=0,i_idum=i_idum,\
                            binsize=Bcase.binsize,threshold=Bcase.threshold)
            tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_avg,Bcase.time_tmp)
            nb_pics,E_pics,T_pics = nb_peaks(tmp_E,tmp_t)
            nb_pic[i_idum,j_Niter]   = nb_pics
            if (nb_pics != 0):
                time_sim[i_idum,j_Niter] = (tmp_t[-1]-tmp_t[0])*1e7
            else:
                time_sim[i_idum,j_Niter] = 0.0
            nb_iter[i_idum,j_Niter]  = int(Niters[j_Niter])

    figure()
    scatter(time_sim,nb_pic,c=nb_iter,s=50)
    
def study_quality(Bcase,nbcases=10,Niter=3e3,compute=1,binsize=100,threshold=50,plot_all=0,time_window=0.5e-7):

    if (compute == 1):
        compute_4dvar(Bcase,nbcases=nbcases,Niter=Niter,binsize=binsize,threshold=threshold)
    else:
        Bcase.mkobs(Niter=Niter,doplot=0,update=0,binsize=Bcase.binsize,threshold=Bcase.threshold)

    # Get the condensed time series of reference observation
    obs_pic, time_pic = condense_time_serie(Bcase.rel_e_avg,Bcase.time_tmp)

    DNS_obs_list = []
    DNS_time_list = []
    FDvar_obs_list = []
    FDvar_time_list = []
    quality_DNS   = zeros(nbcases)
    quality_FDvar = zeros(nbcases)
    for icase in r_[0:nbcases]:
        # Get the condensed time series of DNS runs
        tmp_E,tmp_t = condense_time_serie(Bcase.dns_obs_list[icase],Bcase.time_tmp)
        DNS_obs_list.append(tmp_E)
        DNS_time_list.append(tmp_t)
        # Calculate the quality
        print("--> CASE "+str(icase+1)+"<--")
        quality_DNS[icase] = compute_quality(DNS_obs_list[icase],DNS_time_list[icase],obs_pic, time_pic,time_window=time_window,verbose=0)
        # Get the condensed time series of the 4dvar runs
        tmp_E,tmp_t = condense_time_serie(Bcase.obs_list[icase],Bcase.time_tmp)
        FDvar_obs_list.append(tmp_E)
        FDvar_time_list.append(tmp_t)
        # Calculate the quality
        quality_FDvar[icase] = compute_quality(FDvar_obs_list[icase],FDvar_time_list[icase],obs_pic, time_pic,time_window=time_window)
    
    # Do the plots
    figure()
    subplot(121)
    plot(quality_DNS,ls='.',marker='s',mfc='None',mec='k')
    title("Quality of runs")
    hold(True)
    plot(quality_FDvar,ls='.',marker='s',mfc='r')
    hold(False)
    subplot(122)
    pdf,bins,patches=hist(quality_FDvar[where(quality_FDvar != 0.0)],histtype='step',color='r',bins=20)
    hold(True)
    pdf,bins,patches=hist(quality_DNS[where(quality_DNS != 0.0)],histtype='step',color='k',bins=20)
    hold(False)

    if (plot_all == 1):
        nb_pic,E_pic,t_pic = nb_peaks(array(obs_pic),array(time_pic))
        nxp = int(sqrt(nbcases))+1
        figure()
        for icase in r_[0:nbcases]:
            subplot(nxp,nxp,icase+1)
            plot(array(time_pic)/1e-7,obs_pic,lw=3,color='k',label='OBS')
            hold(True)
            grid(True)
            ax=gca()
            for ipic in r_[0:nb_pic]:
                fill_between([(t_pic[ipic]-time_window)/1e-7,(t_pic[ipic]+time_window)/1e-7],[E_pic[ipic],E_pic[ipic]],y2=0.0,alpha=0.1,color='k')
                
            plot(array(DNS_time_list[icase])/1e-7,DNS_obs_list[icase],lw=1,color='b',label='DNS '+str(quality_DNS[icase]))
            plot(array(FDvar_time_list[icase])/1e-7,FDvar_obs_list[icase],lw=2,color='r',label='4Dvar '+str(quality_FDvar[icase]))
            axvline(FDvar_time_list[icase][-1],color='k')
            title("Case = "+str(icase+1))
            leg=legend(loc='upper left')
            leg.draw_frame(0)
            ltext=leg.get_texts()
            setp(ltext,fontsize=8)

    # Select only best cases (Q > 0.6) for the prediction
    Bcase.best_dns = zeros(nbcases)
    Bcase.best_var = zeros(nbcases)
    min_qual = 0.45
    for icase in r_[0:nbcases]:
        if (quality_FDvar[icase] > min_qual):
            Bcase.best_var[icase] = 1.0
        if (quality_DNS[icase] > min_qual):
            Bcase.best_dns[icase] = 1.0

    print(sum(Bcase.best_var)," 4Dvar cases are selected")
    print(sum(Bcase.best_dns)," DNS   cases are selected")

def predict_first_occ(Bcase,nbcases=10,Niter=3e3,compute=1,time_window=0.066e-7,only_sel=0):

    Bcase.mkobs(Niter=Niter,doplot=0,update=0,binsize=Bcase.binsize,threshold=Bcase.threshold)
    obs_pic,time_pic = condense_time_serie(Bcase.rel_e_avg,Bcase.time_tmp)
    nb_pic,E_pic,t_pic = nb_peaks(obs_pic,time_pic)

    Bcase.mkobs(Niter=2*Niter,doplot=0,update=0,binsize=Bcase.binsize,threshold=Bcase.threshold)
    obs_pic,time_pic = condense_time_serie(Bcase.rel_e_avg,Bcase.time_tmp)
    pred_nb_pic,pred_E_pic,pred_t_pic = nb_peaks(obs_pic,time_pic)

    if (nb_pic == pred_nb_pic):
        print('No new pic, please use a wider window')
        return

    if (only_sel == 1):
        ncases_var=int(sum(Bcase.best_var))
        ncases_dns=int(sum(Bcase.best_dns))
    else:
        ncases_var=nbcases
        ncases_dns=nbcases

    if (compute == 1):
        Bcase.OCC_dns_obs_c = []
        Bcase.OCC_var_obs_c = []
        Bcase.OCC_dns_tim_c = []
        Bcase.OCC_var_tim_c = []
        Bcase.OCC_dns_obs   = []
        Bcase.OCC_var_obs   = []
        for icase in r_[0:nbcases]:
            if ((only_sel == 0)|(Bcase.best_dns[icase] != 0.0)):
                Btmp = cp.copy(Bcase)
                Btmp.mkobs(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1,binsize=Bcase.binsize,threshold=Bcase.threshold)
                Bcase.OCC_dns_obs.append(Btmp.rel_e_avg)
                tmp_E,tmp_t = condense_time_serie(Btmp.rel_e_avg,Btmp.time_tmp)
                Bcase.OCC_dns_obs_c.append(tmp_E)
                Bcase.OCC_dns_tim_c.append(tmp_t)
        for icase in r_[0:nbcases]:
            if ((only_sel == 0)|(Bcase.best_var[icase] != 0.0)):
                Btmp = cp.copy(Bcase)
                Btmp.B = cp.copy(Bcase.newB_list[icase])
                Btmp.mkobs(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1,binsize=Bcase.binsize,threshold=Bcase.threshold)
                Bcase.OCC_var_obs.append(Btmp.rel_e_avg)
                tmp_E,tmp_t = condense_time_serie(Btmp.rel_e_avg,Btmp.time_tmp)
                Bcase.OCC_var_obs_c.append(tmp_E)
                Bcase.OCC_var_tim_c.append(tmp_t)

    # Look for first flare occurence
    OCC_success_var = zeros(ncases_var)
    OCC_success_dns = zeros(ncases_dns)
    for icase in r_[0:ncases_dns]:
        tmp_nb_pic,tmp_E_pic,tmp_t_pic = nb_peaks(Bcase.OCC_dns_obs_c[icase],Bcase.OCC_dns_tim_c[icase])
        matched=False
        for i in r_[0:tmp_nb_pic]:
            # Filter first av
            if (tmp_t_pic[i] > 1e-11):
                if ((abs(tmp_t_pic[i]-pred_t_pic[nb_pic]) <= time_window)&(matched == False)):
                    # match
                    matched = True
        OCC_success_dns[icase] = matched
    for icase in r_[0:ncases_var]:
        tmp_nb_pic,tmp_E_pic,tmp_t_pic = nb_peaks(Bcase.OCC_var_obs_c[icase],Bcase.OCC_var_tim_c[icase])
        matched=False
        for i in r_[0:tmp_nb_pic]:
            # Filter first av
            if (tmp_t_pic[i] > 1e-11):
                if ((abs(tmp_t_pic[i]-pred_t_pic[nb_pic]) <= time_window)&(matched == False)):
                    # match
                    matched = True
        OCC_success_var[icase] = matched

    print(sum(OCC_success_var)," / ",ncases_var," 4Dvar succeeded to forecast the first occuring flare")
    print(sum(OCC_success_dns)," / ",ncases_dns," DNS  succeeded to forecast the first occuring flare")

    quality_DNS   = zeros(ncases_dns)
    quality_FDvar = zeros(ncases_var)
    for icase in r_[0:ncases_dns]:
        # Calculate the quality
        quality_DNS[icase]   = compute_quality(Bcase.OCC_dns_obs_c[icase],Bcase.OCC_dns_tim_c[icase], \
                                                   obs_pic, time_pic, time_window=time_window)
    for icase in r_[0:ncases_var]:
        quality_FDvar[icase] = compute_quality(Bcase.OCC_var_obs_c[icase],Bcase.OCC_var_tim_c[icase], \
                                                   obs_pic, time_pic, time_window=time_window)
    
    # Do the plots
    figure()
    subplot(121)
    plot(quality_DNS,ls='.',marker='s',mfc='None',mec='k')
    title("Quality of runs")
    hold(True)
    plot(quality_FDvar,ls='.',marker='s',mfc='r')
    hold(False)
    subplot(122)
    pdf,bins,patches=hist(quality_FDvar,histtype='step',color='r',bins=20)
    #hist(quality_FDvar[where(quality_FDvar != 0.0)],histtype='step',color='r',bins=20)
    hold(True)
    pdf,bins,patches=hist(quality_DNS,histtype='step',color='k',bins=20)
    #hist(quality_DNS[where(quality_DNS != 0.0)],histtype='step',color='k',bins=20)
    hold(False)

    print("Mean quality  (DNS,4Dvar):",mean(quality_DNS),mean(quality_FDvar))
    print("Std Deviation (DNS,4Dvar):",std(quality_DNS),std(quality_FDvar))


def compute_dns(Bcase,nbcases=10,Niter=1e4,verbose=0):
    # Produces 'nbcases' dns runs (with nbcases different sets of random numbers, up to 10000).
    # Ouputs: - obs_list    : list of observations with 4dvar initial condition
    # The outputs are saved directly in Bcase
    
    Bcase.dns_obs_list = []
    Bcase.Bend_list    = []
    for icase in r_[0:nbcases]:
        if (verbose == 1):
            print("---> CASE ",icase+1,"<---")
        # DNS RUN
        Bcase.do_soc(Niter=Niter,doplot=0,update=0,i_idum=icase+1)
        Bcase.dns_obs_list.append(Bcase.rel_e_tmp)
        Bcase.Bend_list.append(Bcase.B_tmp)

def study_quality_dns(Bcase,nbcases=10,Niter=3e3,\
                          compute=1,update=0,\
                          threshold=0.0,plot_all=0,\
                          time_window=0.5e-7,verbose=0):

    # Get the condensed time series of reference observation
    Bcase.do_soc(Niter=Niter,doplot=0,update=update)
    obs_pic, time_pic = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
                                                threshold=threshold)

    if (compute == 1):
        # compute dns
        compute_dns(Bcase,nbcases=nbcases,Niter=Niter,verbose=verbose)

    DNS_obs_list = []
    DNS_time_list = []
    Bcase.quality_DNS     = zeros(nbcases)
    Bcase.quality_DNS_HTW = zeros(nbcases)
    Bcase.quality_DNS_TTW = zeros(nbcases)
    for icase in r_[0:nbcases]:
        # Get the condensed time series of DNS runs
        tmp_E,tmp_t = condense_time_serie(Bcase.dns_obs_list[icase],Bcase.time_tmp,\
                                              threshold=threshold)
        DNS_obs_list.append(tmp_E)
        DNS_time_list.append(tmp_t)
        # Calculate the quality
        Bcase.quality_DNS[icase] = compute_quality(DNS_obs_list[icase],\
                                                       DNS_time_list[icase],\
                                                       obs_pic, time_pic,\
                                                       time_window=time_window,\
                                                       verbose=0)
        Bcase.quality_DNS_HTW[icase] = compute_quality(DNS_obs_list[icase],\
                                                           DNS_time_list[icase],\
                                                           obs_pic, time_pic,\
                                                           time_window=time_window/2.0,\
                                                       verbose=0)
        Bcase.quality_DNS_TTW[icase] = compute_quality(DNS_obs_list[icase],\
                                                           DNS_time_list[icase],\
                                                           obs_pic, time_pic,\
                                                           time_window=time_window*2.0,\
                                                       verbose=0)

    # Select only best cases (Q > 0.7) for the prediction
    Bcase.best_dns = zeros(nbcases)
    min_qual = 0.7
    for icase in r_[0:nbcases]:
        if (Bcase.quality_DNS[icase] > min_qual):
            Bcase.best_dns[icase] = 1.0

    print(sum(Bcase.best_dns)," DNS cases are selected")
    if (int(sum(Bcase.best_dns)) == 0):
        return

    # Do the plots
    figure()
    pdf,bins,patches=hist(Bcase.quality_DNS[where(Bcase.quality_DNS != 0.0)],\
                              histtype='step',color='k',bins=20,lw=2)
    hold(True)
    pdf,bins,patches=hist(Bcase.quality_DNS_HTW[where(Bcase.quality_DNS != 0.0)],\
                              histtype='step',color='r',bins=20,lw=1)
    pdf,bins,patches=hist(Bcase.quality_DNS_TTW[where(Bcase.quality_DNS != 0.0)],\
                              histtype='step',color='b',bins=20,lw=1)
    hold(False)

    if (plot_all == 1):
        if (Bcase.SOC_case == 4):
            norm_time = 1.0e-7
        else:
            norm_time = 1.0
        nb_pic,E_pic,t_pic = nb_peaks(array(obs_pic),array(time_pic))
        nxp = int(sqrt(nbcases))+1
        figure()
        for icase in r_[0:nbcases]:
            subplot(nxp,nxp,icase+1)
            plot(array(time_pic)/norm_time,obs_pic,lw=3,color='k',label='OBS')
            hold(True)
            grid(True)
            ax=gca()
            for ipic in r_[0:nb_pic]:
                fill_between([(t_pic[ipic]-time_window)/norm_time,\
                                  (t_pic[ipic]+time_window)/norm_time],\
                                 [E_pic[ipic],E_pic[ipic]],\
                                 y2=0.0,alpha=0.1,color='k')
            plot(array(DNS_time_list[icase])/norm_time,\
                     DNS_obs_list[icase],\
                     lw=1,color='b',\
                     label='DNS '+str(Bcase.quality_DNS[icase]))
            title("Case = "+str(icase+1))
            leg=legend(loc='upper left')
            leg.draw_frame(0)
            ltext=leg.get_texts()
            setp(ltext,fontsize=8)


def study_quality_dns_pred(Bcase,nbcases=10,Niter=3e3,\
                               update=0,\
                               threshold=0.0,plot_all=0,\
                               time_window=0.5e-7,verbose=0):

    # Get the condensed time series of reference observation
    # Full set
    Bcase.do_soc(Niter=2*Niter,doplot=0,update=0)
    ttimers=[Niter]
    obs_pic_full, time_pic_full = condense_time_serie(Bcase.rel_e_tmp,\
                                                          Bcase.time_tmp,\
                                                          threshold=threshold,\
                                                          timers=ttimers)
    Bcase.do_soc(Niter=ttimers[1],doplot=0,update=update)
    obs_pic, time_pic = condense_time_serie(Bcase.rel_e_tmp,\
                                                Bcase.time_tmp,\
                                                threshold=threshold)

    n_tmp = len(time_pic)
    time_pic_next=time_pic_full[n_tmp::]-time_pic_full[n_tmp]
    obs_pic_next = obs_pic_full[n_tmp::]- obs_pic_full[n_tmp]
    # compute dns
    Bcase.dns_obs_list = []
    Bcase.Bend_list    = []
    DNS_obs_list = []
    DNS_time_list = []
    Bcase.quality_DNS     = zeros(nbcases)
    Bcase.quality_DNS_HTW = zeros(nbcases)
    Bcase.quality_DNS_TTW = zeros(nbcases)
    # DNS RUN
    for icase in r_[0:nbcases]:
        if (verbose == 1):
            print("---> CASE ",icase+1,"<---")
        Bcase.do_soc(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1)
        ttimers=[Niter]
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,\
                                              Bcase.time_tmp,\
                                              threshold=threshold,timers=ttimers)
        Bcase.do_soc(Niter=ttimers[1],doplot=0,update=0,i_idum=icase+1)
        Bcase.dns_obs_list.append(Bcase.rel_e_tmp)
        Bcase.Bend_list.append(Bcase.B_tmp)
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,\
                                              Bcase.time_tmp,\
                                              threshold=threshold)
        DNS_obs_list.append(tmp_E)
        DNS_time_list.append(tmp_t)
    
    for icase in r_[0:nbcases]:
        # Calculate the quality
        Bcase.quality_DNS[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window,\
                                verbose=0)
        Bcase.quality_DNS_HTW[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window/2.0,\
                                verbose=0)
        Bcase.quality_DNS_TTW[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window*2.0,\
                                verbose=0)

    # Select only best cases (Q > 0.7) for the prediction
    Bcase.best_dns = zeros(nbcases)
    min_qual = 0.7
    for icase in r_[0:nbcases]:
        if (Bcase.quality_DNS[icase] > min_qual):
            Bcase.best_dns[icase] = 1.0

    nb_selected=int(sum(Bcase.best_dns))
    print(nb_selected,"cases selected")

    if (nb_selected == 0) :
        return

    nb_dums = int(nbcases/nb_selected)

    # Do the prediction
    PRED_obs_list=[]
    PRED_time_list=[]
    DNS_PRED_obs_list=[]
    DNS_PRED_time_list=[]
    Btmp = cp.copy(Bcase.B)
    for icase in r_[0:nbcases]:
        # DNS RUN
        Bcase.B = cp.copy(Bcase.Bend_list[icase])
        Bcase.do_soc(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1)
        ttimers=[time_pic_full[-1]-Niter]
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
                                              threshold=threshold,\
                                              timers=ttimers)
        Bcase.do_soc(Niter=ttimers[1],doplot=0,update=0,i_idum=icase+1)
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
                                              threshold=threshold)
        DNS_PRED_obs_list.append(tmp_E)
        DNS_PRED_time_list.append(tmp_t)
        #if (Bcase.best_dns[icase] == 1.0):
        #    if (verbose == 1):
        #        print "---> PREDICTION, CASE ",icase+1,"<---"
        #    # PRED RUN
        #    Bcase.do_soc(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1)
        #    ttimers=[time_pic_full[-1]-Niter]
        #    tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
        #                                          threshold=threshold,\
        #                                          timers=ttimers)
        #    Bcase.do_soc(Niter=ttimers[1],doplot=0,update=0,i_idum=icase+1)
        #    tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
        #                                      threshold=threshold)
        #    PRED_obs_list.append(tmp_E)
        #    PRED_time_list.append(tmp_t)

    Bcase.B = cp.copy(Btmp)

    #####################################
    # Compute the new quality of the runs
    quality_pred         = []
    quality_pred_DNS     = []
    quality_pred_HTW     = []
    quality_pred_DNS_HTW = []
    quality_pred_TTW     = []
    quality_pred_DNS_TTW = []
    for icase in r_[0:nbcases]:
        # Calculate the quality
        qual_tmp = \
            compute_quality(DNS_PRED_obs_list[icase],\
                                DNS_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window,\
                                verbose=0)
        quality_pred_DNS.append(qual_tmp)
        if (Bcase.best_dns[icase] == 1.0):
            quality_pred.append(qual_tmp)
        qual_tmp = \
            compute_quality(DNS_PRED_obs_list[icase],\
                                DNS_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window/2.0,\
                                verbose=0)
        quality_pred_DNS_HTW.append(qual_tmp)
        if (Bcase.best_dns[icase] == 1.0):
            quality_pred_HTW.append(qual_tmp)
        qual_tmp = \
            compute_quality(DNS_PRED_obs_list[icase],\
                                DNS_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window*2.0,\
                                verbose=0)
        quality_pred_DNS_TTW.append(qual_tmp)
        if (Bcase.best_dns[icase] == 1.0):
            quality_pred_TTW.append(qual_tmp)

    quality_pred         = array(quality_pred        )
    quality_pred_DNS     = array(quality_pred_DNS    )
    quality_pred_HTW     = array(quality_pred_HTW    )
    quality_pred_DNS_HTW = array(quality_pred_DNS_HTW)
    quality_pred_TTW     = array(quality_pred_TTW    )
    quality_pred_DNS_TTW = array(quality_pred_DNS_TTW)

    # Save fields 
    Bcase.quality_pred         = quality_pred        
    Bcase.quality_pred_DNS     = quality_pred_DNS    
    Bcase.quality_pred_HTW     = quality_pred_HTW    
    Bcase.quality_pred_DNS_HTW = quality_pred_DNS_HTW
    Bcase.quality_pred_TTW     = quality_pred_TTW    
    Bcase.quality_pred_DNS_TTW = quality_pred_DNS_TTW
    Bcase.DNS_PRED_time_list   = DNS_PRED_time_list
    Bcase.DNS_PRED_obs_list    = DNS_PRED_obs_list
    Bcase.DNS_time_list        = DNS_time_list
    Bcase.DNS_obs_list         = DNS_obs_list
    Bcase.obs_pic_full         = obs_pic_full
    Bcase.time_pic_full        = time_pic_full
    Bcase.obs_pic              = obs_pic
    Bcase.time_pic             = time_pic
    Bcase.nbcases              = nbcases
    Bcase.nb_selected          = nb_selected

    print("Fields saved")

    ##############
    # Do the plots
    do_plot_pred(Bcase,plot_all=plot_all,time_window=time_window)

def study_quality_dns_pred2(Bcase,nbcases=10,Niter=3e3,\
                                update=0,\
                                threshold=0.0,plot_all=0,\
                                time_window=0.5e-7,verbose=0):

    # Get the condensed time series of reference observation
    # Full set
    Bcase.do_soc(Niter=2*Niter,doplot=0,update=0)
    ttimers=[Niter]
    obs_pic_full, time_pic_full = condense_time_serie(Bcase.rel_e_tmp,\
                                                          Bcase.time_tmp,\
                                                          threshold=threshold,\
                                                          timers=ttimers)
    Bcase.do_soc(Niter=ttimers[1],doplot=0,update=update)
    obs_pic, time_pic = condense_time_serie(Bcase.rel_e_tmp,\
                                                Bcase.time_tmp,\
                                                threshold=threshold)

    n_tmp = len(time_pic)
    print(n_tmp)
    time_pic_next=time_pic_full[n_tmp::]-time_pic_full[n_tmp]
    obs_pic_next = obs_pic_full[n_tmp::]#- obs_pic_full[n_tmp]
    print(time_pic_next)
    # compute dns
    Bcase.dns_obs_list = []
    Bcase.Bend_list    = []
    DNS_obs_list = []
    DNS_time_list = []
    Bcase.quality_DNS     = zeros(nbcases)
    Bcase.quality_DNS_HTW = zeros(nbcases)
    Bcase.quality_DNS_TTW = zeros(nbcases)
    # DNS RUN
    for icase in r_[0:nbcases]:
        if (verbose == 1):
            print("---> CASE ",icase+1,"<---")
        Bcase.do_soc(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1)
        ttimers=[Niter]
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,\
                                              Bcase.time_tmp,\
                                              threshold=threshold,timers=ttimers)
        Bcase.do_soc(Niter=ttimers[1],doplot=0,update=0,i_idum=icase+1)
        Bcase.dns_obs_list.append(Bcase.rel_e_tmp)
        Bcase.Bend_list.append(Bcase.B_tmp)
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,\
                                              Bcase.time_tmp,\
                                              threshold=threshold)
        DNS_obs_list.append(tmp_E)
        DNS_time_list.append(tmp_t)
    
    for icase in r_[0:nbcases]:
        # Calculate the quality
        Bcase.quality_DNS[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window,\
                                verbose=0)
        Bcase.quality_DNS_HTW[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window/2.0,\
                                verbose=0)
        Bcase.quality_DNS_TTW[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window*2.0,\
                                verbose=0)

    # Select only best 10 cases for the prediction
    Bcase.best_dns = zeros(nbcases)
    nb_selected = min(10,nbcases)
    min_qual = get_maxim_n(Bcase.quality_DNS,nb_selected)
    for icase in r_[0:nbcases]:
        if (Bcase.quality_DNS[icase] >= min_qual):
            Bcase.best_dns[icase] = 1.0

    nb_selected=int(sum(Bcase.best_dns))
    print(nb_selected,"cases selected, with q > ",min_qual)

    nb_dums = int(nbcases/nb_selected)

    # Do the prediction
    PRED_obs_list=[]
    PRED_time_list=[]
    DNS_PRED_obs_list=[]
    DNS_PRED_time_list=[]
    Btmp = cp.copy(Bcase.B)
    for icase in r_[0:nbcases]:
        # DNS RUN
        Bcase.B = cp.copy(Bcase.Bend_list[icase])
        Bcase.do_soc(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1)
        ttimers=[time_pic_full[-1]-Niter]
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
                                              threshold=threshold,\
                                              timers=ttimers)
        Bcase.do_soc(Niter=ttimers[1],doplot=0,update=0,i_idum=icase+1)
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
                                              threshold=threshold)
        DNS_PRED_obs_list.append(tmp_E)
        DNS_PRED_time_list.append(tmp_t)
    icase_sel = 0
    SEL_PRED_obs_list=[]
    SEL_PRED_time_list=[]
    for icase in r_[0:nbcases]:
        if (Bcase.best_dns[icase] == 1.0):
            # SELECTED RUNS
            for ii in r_[0:nb_dums]:
                Bcase.B = cp.copy(Bcase.Bend_list[icase])
                ii_idum = icase_sel*nb_dums+ii
                Bcase.do_soc(Niter=2*Niter,doplot=0,update=0,i_idum=ii_idum)
                ttimers=[time_pic_full[-1]-Niter]
                tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
                                                      threshold=threshold,\
                                                      timers=ttimers)
                Bcase.do_soc(Niter=ttimers[1],doplot=0,update=0,i_idum=ii_idum)
                tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,Bcase.time_tmp,\
                                                      threshold=threshold)
                SEL_PRED_obs_list.append(tmp_E)
                SEL_PRED_time_list.append(tmp_t)

    Bcase.B = cp.copy(Btmp)

    #####################################
    # Compute the new quality of the runs
    quality_pred         = []
    quality_pred_DNS     = []
    quality_pred_HTW     = []
    quality_pred_DNS_HTW = []
    quality_pred_TTW     = []
    quality_pred_DNS_TTW = []
    for icase in r_[0:nbcases]:
        # Calculate the quality
        qual_tmp = \
            compute_quality(DNS_PRED_obs_list[icase],\
                                DNS_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window,\
                                verbose=0)
        quality_pred_DNS.append(qual_tmp)
        qual_tmp = \
            compute_quality(DNS_PRED_obs_list[icase],\
                                DNS_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window/2.0,\
                                verbose=0)
        quality_pred_DNS_HTW.append(qual_tmp)
        qual_tmp = \
            compute_quality(DNS_PRED_obs_list[icase],\
                                DNS_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window*2.0,\
                                verbose=0)
        quality_pred_DNS_TTW.append(qual_tmp)
    # SELECTED CASES QUALITY
    for icase in r_[0:nb_selected*nb_dums]:
        qual_tmp = \
            compute_quality(SEL_PRED_obs_list[icase],\
                                SEL_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window,\
                                verbose=1)
        quality_pred.append(qual_tmp)
        qual_tmp = \
            compute_quality(SEL_PRED_obs_list[icase],\
                                SEL_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window/2.0,\
                                verbose=0)
        quality_pred_HTW.append(qual_tmp)
        qual_tmp = \
            compute_quality(SEL_PRED_obs_list[icase],\
                                SEL_PRED_time_list[icase],\
                                obs_pic_next, time_pic_next,\
                                time_window=time_window*2.0,\
                                verbose=0)
        quality_pred_TTW.append(qual_tmp)

    quality_pred         = array(quality_pred        )
    quality_pred_DNS     = array(quality_pred_DNS    )
    quality_pred_HTW     = array(quality_pred_HTW    )
    quality_pred_DNS_HTW = array(quality_pred_DNS_HTW)
    quality_pred_TTW     = array(quality_pred_TTW    )
    quality_pred_DNS_TTW = array(quality_pred_DNS_TTW)

    # Save fields 
    Bcase.quality_pred         = quality_pred        
    Bcase.quality_pred_DNS     = quality_pred_DNS    
    Bcase.quality_pred_HTW     = quality_pred_HTW    
    Bcase.quality_pred_DNS_HTW = quality_pred_DNS_HTW
    Bcase.quality_pred_TTW     = quality_pred_TTW    
    Bcase.quality_pred_DNS_TTW = quality_pred_DNS_TTW
    Bcase.DNS_PRED_time_list   = DNS_PRED_time_list
    Bcase.DNS_PRED_obs_list    = DNS_PRED_obs_list
    Bcase.DNS_time_list        = DNS_time_list
    Bcase.DNS_obs_list         = DNS_obs_list
    Bcase.obs_pic_full         = obs_pic_full
    Bcase.time_pic_full        = time_pic_full
    Bcase.obs_pic              = obs_pic
    Bcase.time_pic             = time_pic
    Bcase.nbcases              = nbcases
    Bcase.nb_selected          = nb_selected
    Bcase.nb_dums              = nb_dums

    ##############
    # Do the plots
    do_plot_pred(Bcase,plot_all=plot_all,time_window=time_window)

def test_compactness(Bcase,nbcases=10,Niter=1.5e4,\
                         plot_all=0,time_window=2000,\
                         threshold=400,verbose=1,update=0):
    
    # Generate observations
    Bcase.do_soc(Niter=Niter,doplot=0,update=update)
    obs_pic, time_pic = condense_time_serie(Bcase.rel_e_tmp,\
                                                Bcase.time_tmp,\
                                                threshold=threshold)
    # Do DNS from the same initial condition
    DNS_obs_list = []
    DNS_time_list = []
    Bcase.quality_DNS     = zeros(nbcases)
    Bcase.quality_DNS_HTW = zeros(nbcases)
    Bcase.quality_DNS_TTW = zeros(nbcases)
    for icase in r_[0:nbcases]:
        if (verbose == 1):
            print("---> CASE ",icase+1,"<---")
        Bcase.do_soc(Niter=Niter,doplot=0,update=0,i_idum=icase+1)
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_tmp,\
                                              Bcase.time_tmp,\
                                              threshold=threshold)
        DNS_obs_list.append(tmp_E)
        DNS_time_list.append(tmp_t)
        Bcase.quality_DNS[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window,\
                                verbose=0)
        Bcase.quality_DNS_HTW[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window/2.0,\
                                verbose=0)
        Bcase.quality_DNS_TTW[icase] = \
            compute_quality(DNS_obs_list[icase],\
                                DNS_time_list[icase],\
                                obs_pic, time_pic,\
                                time_window=time_window*2.0,\
                                verbose=0)

    Bcase.quality_pred      = Bcase.quality_DNS
    Bcase.quality_pred_HTW  = Bcase.quality_DNS_HTW
    Bcase.quality_pred_TTW  = Bcase.quality_DNS_TTW
    Bcase.quality_pred_DNS      = Bcase.quality_DNS
    Bcase.quality_pred_DNS_HTW  = Bcase.quality_DNS_HTW
    Bcase.quality_pred_DNS_TTW  = Bcase.quality_DNS_TTW

    Bcase.best_dns = np.ones(nbcases)
    
    Bcase.DNS_PRED_time_list   = DNS_time_list
    Bcase.DNS_PRED_obs_list    = DNS_obs_list
    Bcase.DNS_time_list        = DNS_time_list
    Bcase.DNS_obs_list         = DNS_obs_list
    Bcase.obs_pic_full         = obs_pic
    Bcase.time_pic_full        = time_pic
    Bcase.obs_pic              = obs_pic
    Bcase.time_pic             = time_pic
    Bcase.nbcases              = nbcases
    Bcase.nb_selected          = nbcases
    Bcase.nb_dums              = 1

    # Do the plots
    do_plot_pred(Bcase,plot_all=plot_all,time_window=time_window,filter_0=False,plot_pred=False)

def do_plot_pred(Bcase,plot_all=0,time_window=2000,filter_0=True,plot_pred=True):

    if (Bcase.SOC_case == 4):
        norm_time = 1.0e-7
    else:
        norm_time = 1.0

    # Try to predict the position of the pic
    # with an ensemble mean
    tpic_ens = []
    Epic_ens = []
    for iicase in r_[0:Bcase.nbcases]:
        nb_pic,E_pic,t_pic = nb_peaks(Bcase.DNS_obs_list[iicase],\
                                          Bcase.DNS_time_list[iicase])
        if (nb_pic != 0):
            tpic_ens.append(t_pic[0])
            Epic_ens.append(E_pic[0])

    print(100.0-len(array(Epic_ens))*100.0/Bcase.nbcases,\
        '% of the cases have no pics at all')
    figure()
    subplot(121)
    pdf,bins,patches=hist(array(tpic_ens),normed=True,\
                              histtype='step',color='k',bins=30,lw=1)
    title("Time distribution")

    imaxt1 = where(pdf == amax(pdf))[0]
    print('Pic expected at ',bins[imaxt1:imaxt1+2])
    pdf[imaxt1]=0.
    imaxt2 = where(pdf == amax(pdf))[0]
    print(pdf[imaxt2],bins[imaxt2],bins[imaxt2+1])
    pdf[imaxt2]=0.
    imaxt3 = where(pdf == amax(pdf))[0]
    print(pdf[imaxt3],bins[imaxt3],bins[imaxt3+1])
    
    ttpic=array(tpic_ens)
    eepic=array(Epic_ens)
    indx1 = array(where((ttpic <= bins[imaxt1+1])&(ttpic >= bins[imaxt1]))[0])
    #indx2 = array(where((ttpic <= bins[imaxt2+1])&(ttpic >= bins[imaxt2]))[0])
    #indx3 = array(where((ttpic <= bins[imaxt3+1])&(ttpic >= bins[imaxt3]))[0])

    print('indx1',indx1)
    #print 'indx2',indx2
    #print 'indx3',indx3

    subplot(122)
    pdf,bins,patches=hist(eepic,\
                              histtype='step',color='k',bins=30,lw=3)
    hold(True)
    pdf,bins,patches=hist(eepic[indx1],\
                              histtype='step',color='r',bins=30,lw=1)
    #pdf,bins,patches=hist(eepic[indx2],\
    #                          histtype='step',color='b',bins=30,lw=1)
    #pdf,bins,patches=hist(eepic[indx3],\
    #                          histtype='step',color='g',bins=30,lw=1)
    hold(False)
    title("E distribution")
    
    #return


    if (plot_all == 1):
        ##############
        # Time plots
        nb_pic,E_pic,t_pic = nb_peaks(array(Bcase.obs_pic_full),\
                                          array(Bcase.time_pic_full))
        nxp = int(sqrt(Bcase.nb_selected))+1
        if (nxp*(nxp-1) < Bcase.nb_selected):
            nyp = nxp
        else:
            nyp = nxp-1
        figure()
        iicase=0
        for icase in r_[0:Bcase.nbcases]:
            if (Bcase.best_dns[icase] == 1.0):
                subplot(nxp,nyp,iicase+1)
                plot_bar_ts(Bcase,icase,norm_time,time_window,\
                                plot_pred=plot_pred)
                #plot_bar_ts(Bcase,icase,norm_time,time_window,\
                #                plot_pred=plot_pred,pred_tpic=pred_tpic,\
                #                pred_Epic=pred_Epic)
                iicase = iicase+1
    #######
    # Pdfs
    print(size(where(Bcase.quality_pred_DNS == 0.0))*100.0/Bcase.nbcases,\
        '% DNS are 0.0')
    print(size(where(Bcase.quality_pred == 0.0))*100.0/(Bcase.nb_selected*\
                                                            Bcase.nb_dums),\
        '% selected are 0.0')
    if (size(where(Bcase.quality_pred == 0.0)) == Bcase.nb_selected*Bcase.nb_dums):
        return

    indx=where(Bcase.quality_pred_DNS == 0.0)[0]
    bad_nbpic = []
    indx=array(indx)
    for iicase in indx:
        nb_pic,E_pic,t_pic = nb_peaks(Bcase.DNS_obs_list[iicase],\
                                          Bcase.DNS_time_list[iicase])
        bad_nbpic.append(nb_pic)
    
    print(size(where(array(bad_nbpic)!=0))*100.0/size(indx),\
        ' % of bad cases have pic(s)')
    figure()
    pdf,bins,patches=hist(array(bad_nbpic),histtype='step',color='k',bins=3,lw=1)
    title('Nb of pics for bad cases')

    if (filter_0):
        q_p_DNS     = Bcase.quality_pred_DNS[where((Bcase.quality_pred_DNS != 0.0))]
        q_p_DNS_TTW = Bcase.quality_pred_DNS_TTW[where((Bcase.quality_pred_DNS_TTW != 0.0))]
        q_p_DNS_HTW = Bcase.quality_pred_DNS_HTW[where((Bcase.quality_pred_DNS_HTW != 0.0))]
        q_p     = Bcase.quality_pred[where((Bcase.quality_pred != 0.0))]
        q_p_TTW = Bcase.quality_pred_TTW[where((Bcase.quality_pred_TTW != 0.0))]
        q_p_HTW = Bcase.quality_pred_HTW[where((Bcase.quality_pred_HTW != 0.0))]
    else:
        q_p_DNS     = Bcase.quality_pred_DNS
        q_p_DNS_TTW = Bcase.quality_pred_DNS_TTW
        q_p_DNS_HTW = Bcase.quality_pred_DNS_HTW
        q_p     = Bcase.quality_pred
        q_p_TTW = Bcase.quality_pred_TTW
        q_p_HTW = Bcase.quality_pred_HTW

    figure()
    pdf,bins,patches=hist(q_p_DNS,histtype='step',color='k',bins=20,lw=1)
    hold(True)
    pdf,bins,patches=hist(q_p_DNS_TTW,histtype='step',color='b',bins=20,lw=1)
    pdf,bins,patches=hist(q_p_DNS_HTW,histtype='step',color='r',bins=20,lw=1)
    pdf,bins,patches=hist(q_p,histtype='step',color='k',bins=20,lw=3)
    pdf,bins,patches=hist(q_p_TTW,histtype='step',color='b',bins=20,lw=3)
    pdf,bins,patches=hist(q_p_HTW,histtype='step',color='r',bins=20,lw=3)
    hold(False)
    legend(('Normal','2x tw','tw/2'),loc='upper right')


def plot_bar(H,time_window,norm_time=1.0,label='OBS'):

    line,=plot(H.time_pic/norm_time,H.obs_pic,lw=3,label=label)
    for ipic in r_[0:H.nb_pic]:
        fill_between([(H.t_pic[ipic]-time_window)/norm_time,\
                          (H.t_pic[ipic]+time_window)/norm_time],\
                         [H.E_pic[ipic],H.E_pic[ipic]],\
                         y2=0.0,alpha=0.1,color=line.get_color())
    axvline(H.time_pic[-1]/norm_time,ls='--',color=line.get_color())

def plot_bar_ts(Bcase,icase,norm_time,time_window,plot_pred=True,\
                    pred_Epic=0.0,pred_tpic=0.0):

    nb_pic,E_pic,t_pic = nb_peaks(array(Bcase.obs_pic_full),\
                                      array(Bcase.time_pic_full))
    plot(Bcase.time_pic_full/norm_time,\
             Bcase.obs_pic_full,\
             lw=3,color='k',label='OBS')
    hold(True)
    grid(True)
    ax=gca()
    for ipic in r_[0:nb_pic]:
        fill_between([(t_pic[ipic]-time_window)/norm_time,\
                          (t_pic[ipic]+time_window)/norm_time],\
                         [E_pic[ipic],E_pic[ipic]],\
                         y2=0.0,alpha=0.1,color='k')
    plot(array(Bcase.DNS_time_list[icase])/norm_time,\
             Bcase.DNS_obs_list[icase],\
             lw=1,color='b',\
             label='DNS '+str(Bcase.quality_DNS[icase]))
    if (plot_pred):
        plot((Bcase.DNS_PRED_time_list[icase]+\
                  Bcase.DNS_time_list[icase][-1])/norm_time,\
                 Bcase.DNS_PRED_obs_list[icase],\
                 lw=1,color='r',\
                 label='PREDICTION '+str(Bcase.quality_pred_DNS[icase]))
    if (pred_Epic != 0.0):
        axvline(pred_tpic/norm_time,ymax=pred_Epic,ls='--',color='r')
    axvline(Bcase.time_pic[-1]/norm_time,ls='--',color='k')
    axvline(Bcase.DNS_time_list[icase][-1]/norm_time,ls='--',color='b')
    title("Case = "+str(icase+1))
    leg=legend(loc='upper left')
    leg.draw_frame(0)
    ltext=leg.get_texts()
    setp(ltext,fontsize=8)

def get_maxim_n(a,nth_max):
    
    b=cp.copy(a)
    for i in r_[0:nth_max]:
        c=amax(b)
        b[where(b==c)] = 0.0

    return c

def get_WT(Bcase):

    NN = int(1e6)
    Fclasses = [0.0,5.0,30.0,150.0]
    #Bsizes   = [20,50,100,500,1000]
    Bsizes   = [1,2,5,10,20]
    Nclasses = len(Fclasses)
    Nbinsize = len(Bsizes)
    WT = zeros((Nclasses,Nbinsize))
    
    Bcase.do_soc(Niter=NN,update=0,doplot=0)

    #for i in r_[1:Nclasses]:
    for j in r_[0:Nbinsize]:
        Bcase.mkobs(Niter=NN,doplot=0,update=0,compute=0,binsize=Bsizes[j],threshold=0.0)#Fclasses[i])
        tmp_E,tmp_t = condense_time_serie(Bcase.rel_e_avg,Bcase.time_tmp)
        tmptmp_E = array(tmp_E)
        indx_E = where(tmptmp_E != 0.0)
        for i in r_[0:Nclasses]:
            print(amax(tmptmp_E[indx_E])-amin(tmptmp_E[indx_E]),amax(tmptmp_E[indx_E]),amin(tmptmp_E[indx_E]))
            locthresh=exp(i*log(amax(tmptmp_E[indx_E])-amin(tmptmp_E[indx_E]))/4)
            tmptmp_E[where(tmptmp_E < locthresh)] = 0.0
            tmptmp_t = array(tmp_t)
            tmp_WT = tmptmp_t[where(tmptmp_E != 0.0)]
            tmp_WT = tmp_WT[where(tmp_WT > 1e-11)] # get rid of eventual first flare
            WT[i,j] = mean(diff(tmp_WT))
    
    figure()
    plot(Fclasses,WT[:,0],ls='.',marker='s')
    hold(True)
    for i in r_[1:Nbinsize]:
        plot(Fclasses,WT[:,i],ls='.',marker='s')
    hold(False)
    

def multiple_4dvar(Bcase,nbcases=10,Niter=1e4,compute=1,prediction=0):

    if (compute == 1):
        compute_4dvar(Bcase,nbcases=nbcases,Niter=Niter)
    else:
        Bcase.mkobs(Niter=Niter,doplot=0,update=0,binsize=Bcase.binsize,threshold=Bcase.threshold)

    #figure()
    #plot(Bcase.time_tmp,Bcase.rel_e_avg,lw=3)
    #hold(True)
    #for obs_case in Bcase.obs_list:
    #    plot(Bcase.time_tmp,obs_case,lw=1)

    new_obs_list = []
    new_time_list = []
    for icase in r_[0:nbcases]:
        tmp_E,tmp_t = condense_time_serie(Bcase.obs_list[icase],Bcase.time_tmp)
        new_obs_list.append(tmp_E)
        new_time_list.append(tmp_t)
    obs_pic, time_pic = condense_time_serie(Bcase.rel_e_avg,Bcase.time_tmp)

    # Calculate a quality factor
    quality_cases = zeros(nbcases)
    for icase in r_[0:nbcases]:
        quality_cases[icase] = compute_quality(new_obs_list[icase],new_time_list[icase],obs_pic, time_pic)

    first_av = True
    second_av= False
    third_av = False
    for i in r_[1:len(obs_pic)]:
        if ((obs_pic[i] > 0)&(third_av)):
            time_third_av = time_pic[i]
            third_av=False
        if ((obs_pic[i] > 0)&(second_av)):
            time_second_av = time_pic[i]
            second_av=False
            third_av = True
        if ((obs_pic[i] > 0)&(first_av)):
            time_first_av = time_pic[i]
            first_av=False
            second_av=True
    time_win = 0.07e-7
    first_success=[]
    second_success=[]
    third_success =[]
    for icase in r_[0:nbcases]:
        first_av = True
        second_av= False
        third_av = False
        for i in r_[1:len(new_obs_list[icase])]:
            if ((new_obs_list[icase][i] > 0)&(third_av)):
                third_av = False
                if (abs(new_time_list[icase][i]-time_third_av) <= time_win):
                    third_success.append(True)
                else:
                    third_success.append(False)
            if ((new_obs_list[icase][i] > 0)&(second_av)):
                second_av= False
                third_av=True
                if (abs(new_time_list[icase][i]-time_second_av) <= time_win):
                    second_success.append(True)
                else:
                    second_success.append(False)
            if ((new_obs_list[icase][i] > 0)&(first_av)):
                first_av = False
                second_av=True
                if (abs(new_time_list[icase][i]-time_first_av) <= time_win):
                    first_success.append(True)
                else:
                    first_success.append(False)
        if (len(first_success) != icase+1):
            first_success.append(False)
        if (len(second_success) != icase+1):
            second_success.append(False)
        if (len(third_success) != icase+1):
            third_success.append(False)
            
    print(sum(first_success),'over',nbcases,'models reproduced the first avalanche')
    print(sum(second_success),'over',nbcases,'models reproduced the second avalanche')
    print(sum(third_success),'over',nbcases,'models reproduced the third avalanche')

    figure()
    plot(quality_cases,ls='.',marker='s')
    title("Quality of runs")
    hold(True)
    qual_qualif = zeros(nbcases)
    for icase in r_[0:nbcases]:
        if (first_success[icase]):
            plot([icase],[quality_cases[icase]],ls='.',marker='o',mfc='None',mec='r',mew=2)
        if (quality_cases[icase] > 0.66666666):
            qual_qualif[icase]=1
    hold(False)

    nxp = int(sqrt(sum(first_success)))+1
    figure()
    i_plot=1
    for icase in r_[0:nbcases]:
        if (first_success[icase]):
            subplot(nxp,nxp,i_plot)
            plot(time_pic,obs_pic,lw=3)
            hold(True)
            if (second_success[icase] == True):
                if (third_success[icase] == True):
                    mycol = 'r'
                else:
                    mycol = 'm'
            else:
                if (third_success[icase] == True):
                    mycol = 'k'
                else:
                    mycol = 'g'
            plot(new_time_list[icase],new_obs_list[icase],lw=1,color=mycol)
            axvline(new_time_list[icase][-1],color='k')
            title("Idum = "+str(idum_bank(icase+1))+" (icase="+str(icase+1)+")")
            i_plot=i_plot+1

    q_nxp = int(sqrt(sum(qual_qualif)))+1
    figure()
    i_plot=1
    for icase in r_[0:nbcases]:
        if (qual_qualif[icase]):
            print("--------------------------------------------------")
            print("Case ",icase+1," (qual =",quality_cases[icase],")")
            tmp_qual = compute_quality(new_obs_list[icase],new_time_list[icase],obs_pic, time_pic)
            subplot(q_nxp,q_nxp,i_plot)
            plot(time_pic,obs_pic,lw=3)
            hold(True)
            plot(new_time_list[icase],new_obs_list[icase],lw=1,color='r')
            axvline(new_time_list[icase][-1],color='k')
            title("Qual = "+str(quality_cases[icase])+" (icase="+str(icase+1)+")")
            i_plot=i_plot+1

    if (prediction == 1):
        # Do the evolution of the model over twice as many iterations
        Bcase.mkobs(Niter=2*Niter,doplot=0,update=0,binsize=Bcase.binsize,threshold=Bcase.threshold)
        pred_obs_pic, pred_time_pic = condense_time_serie(Bcase.rel_e_avg,Bcase.time_tmp)
        # Do the evolution for the best models
        pred_time_list =[]
        pred_obs_list =[]
        dns_time_list =[]
        dns_obs_list =[]
        pred_quality_cases = cp.copy(quality_cases)
        dns_quality_cases  = cp.copy(quality_cases)
        dns_old_quality_cases  = cp.copy(quality_cases)
        for icase in r_[0:nbcases]:
            obs_pic_dns, time_pic_dns = condense_time_serie(Bcase.dns_obs_list[icase],Bcase.time_tmp)
            dns_old_quality_cases[icase] = compute_quality(obs_pic_dns,time_pic_dns,obs_pic,time_pic)
            if (first_success[icase]):
                Btmp = cp.copy(Bcase)
                Btmp.mkobs(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1,binsize=Bcase.binsize,threshold=Bcase.threshold)
                obs_pic_dns, time_pic_dns = condense_time_serie(Btmp.rel_e_avg,Btmp.time_tmp)
                dns_time_list.append(time_pic_dns)
                dns_obs_list.append(obs_pic_dns)
                dns_quality_cases[icase] = compute_quality(obs_pic_dns,time_pic_dns,pred_obs_pic,pred_time_pic)
                Btmp.B = cp.copy(Bcase.newB_list[icase])
                Btmp.mkobs(Niter=2*Niter,doplot=0,update=0,i_idum=icase+1,binsize=Bcase.binsize,threshold=Bcase.threshold)
                obs_pic_tmp, time_pic_tmp = condense_time_serie(Btmp.rel_e_avg,Btmp.time_tmp)
                pred_time_list.append(time_pic_tmp)
                pred_obs_list.append(obs_pic_tmp)
                pred_quality_cases[icase] = compute_quality(obs_pic_tmp,time_pic_tmp,pred_obs_pic,pred_time_pic)
                
        figure()
        i_plot=1
        for icase in r_[0:nbcases]:
            if (first_success[icase]):
                subplot(nxp,nxp,i_plot)
                plot(pred_time_pic,pred_obs_pic,lw=3)
                hold(True)
                if (second_success[icase] == True):
                    if (third_success[icase] == True):
                        mycol = 'r'
                    else:
                        mycol = 'm'
                else:
                    if (third_success[icase] == True):
                        mycol = 'k'
                    else:
                        mycol = 'g'
                plot(pred_time_list[i_plot-1],pred_obs_list[i_plot-1],lw=1,color=mycol)
                plot(dns_time_list[i_plot-1],dns_obs_list[i_plot-1],lw=1,color='k')
                axvline(new_time_list[icase][-1],ls='--',color='k')
                axvline(pred_time_list[i_plot-1][-1],color='k')
                i_plot=i_plot+1

        figure()
        plot(quality_cases,ls='.',marker='s')
        title("Quality of runs")
        hold(True)
        for icase in r_[0:nbcases]:
            plot(dns_old_quality_cases,ls='.',marker='s',mfc='None',mec='k')
            if (first_success[icase]):
                plot([icase],[quality_cases[icase]],ls='.',marker='o',mfc='None',mec='r',mew=2)
            if (pred_quality_cases[icase] != quality_cases[icase]):
                plot([icase],[pred_quality_cases[icase]],ls='.',marker='>',mfc='c')
            if (dns_quality_cases[icase] != quality_cases[icase]):
                plot([icase],[dns_quality_cases[icase]],ls='.',marker='<',mec='k',mfc='None')
        hold(False)

        figure()
        plot((quality_cases-dns_old_quality_cases)/dns_old_quality_cases)
        title("Percentage of improvment by 4Dvar...")

def compute_quality(new_obs,new_time,obs,time,time_window=0.066e-7,verbose=0):

    alpha = 4.0
    beta  = 2.0
    gamma = 1.0

    nb_pics,E_pics,T_pics = nb_peaks(obs,time)
    new_nb_pics,new_E_pics,new_T_pics = nb_peaks(new_obs,new_time)

    if (verbose == 1):
        print("Nb peaks (obs, sim): ",new_nb_pics,nb_pics)

    match = 0.0
    match_tmp = 0.0
    miss  = 0.0
    fals_a= 0.0
    Etot = sum(obs)
    nmatch = 0
    nmatch_tmp = 0
    nmiss  = 0
    nfals  = 0
    for j in r_[0:nb_pics]:
        matched=False
        match_tmp  = 0.0
        nmatch_tmp = 0
        for i in r_[0:new_nb_pics]:
            if ((abs(new_T_pics[i]-T_pics[j]) <= time_window)&(new_T_pics[i]>1e-11)):
                # match
                match_tmp = match_tmp + abs(1-(E_pics[j]-new_E_pics[i])**2/(E_pics[j]+new_E_pics[i])**2)
                nmatch_tmp = nmatch_tmp+1
                matched=True
        if (verbose == 1):
            print("Pic "+str(j)+" (E= "+str(E_pics[j])+") matched "+str(nmatch)+" times")
        if (matched == False): 
            # miss
            miss = miss + E_pics[j]/Etot
            nmiss = nmiss + 1
        else:
            # eventually several matches
            nmatch = nmatch+1
            match = match + match_tmp/nmatch_tmp
    for i in r_[0:new_nb_pics]:
        # Filter first av
        matched=False
        for j in r_[0:nb_pics]:
            if ((abs(new_T_pics[i]-T_pics[j]) <= time_window)&(matched == False)&(new_T_pics[i]>1e-11)):
                # match
                matched = True
        if (matched == False):
            # false alarm
            fals_a = fals_a + new_E_pics[i]/Etot
            nfals = nfals + 1

    if (nb_pics != 0):
        qual = (alpha*match - beta*miss - gamma*fals_a)/(alpha*nb_pics)
    else:
        if (new_nb_pics == 0):
            qual=1.0
        else:
            qual=0.0
    if (verbose == 1):
        print("Matches :",nmatch)
        print("Misses  :",nmiss)
        print("False Al:",nfals)

    if (qual < 0.0):
        qual = 0.0

    return qual

def gene_list_4dvar(H):
    H.to_save_4Dvar = cp.copy(H.state_to_save)
    H.to_save_4Dvar.append("time_tmp")
    H.to_save_4Dvar.append("rel_e_avg")
    H.to_save_4Dvar.append("obs_list")
    H.to_save_4Dvar.append("dns_obs_list")
    H.to_save_4Dvar.append("newB_list")
    H.to_save_4Dvar.append("threshold")
    H.to_save_4Dvar.append("binsize")

def my_pdf(E_av,alpha_guess=[1.4,10**4],doplot=0,sub_bin=5):
    min_bin = int(amin(log10(E_av[where(E_av != 0)])))-1
    max_bin = int(amax(log10(E_av[where(E_av != 0)])))+1
    mybins_E = exp(linspace(min_bin,max_bin,num=(max_bin-min_bin+1)*sub_bin)*log(10))
    pdf_E,mybins_E,patches=hist(E_av,bins=mybins_E,histtype='step')
    pdf_E=pdf_E/(size(E_av)*diff(mybins_E))
    cbins_E,mypdf_E=clean_bins(mybins_E,pdf_E)
    alpha_E,dist_fit_E=optimize_leastsq(cbins_E,mypdf_E,alpha_guess)
    print('Alpha:',alpha_E[0])
        
    if (doplot == 1):
        figure()
        step(mybins_E[:-1],pdf_E,where='post')
        xscale('log')
        yscale('log')
        hold(True)
        plot(cbins_E,dist_fit_E)
        hold(False)
        axvline(alpha_E[1],ls='--',color='k')

    return pdf_E,alpha_E,dist_fit_E

def idum_bank(int_idum):
    # Bank of initial integer
    # to generate sequences of random numbers.
    # Taken from all the files of E. Belanger
    my_bank=[\
        -67209495,     -827774,       -9871, -2003241131,      -22631, -1101417806, -1806651135, -1774802181,      -28320,  -114496446,\
             -1161878826, -1554484342,  -424437830,     -138418,      -25730, -1479333046, -1050400305,  -417416817, -1632597228,       -3659,\
             -779, -1492381609,    -9741900, -1507062478, -1515512030, -1800305219,   -43945530, -1612412386, -1936349465, -1393521792,\
             -412651983,  -821700980,  -708985071,       -4303,  -526153415,  -861669009,       -4938,   -68015377,  -186675295,   -66494675,\
             -4619298,      -16981,    -1947198,      -68938, -1048053180, -1666625447, -1359403137,  -136785884, -1437338506, -2107971135,\
             -8800,  -639273420,       -6516,        -911, -1120558198,      -80034,    -3239082,  -302160331, -1876626559, -1308096776,\
             -1563467997,      -58664, -1041631411,  -118760136, -1751659967,  -681553239, -2054879674,      -93452,  -289840009,       -4862,\
             -2294030,  -190752618,     -997774,  -820400162,   -29615303,   -61826658,      -36087,       -7258,      -12461,   -96494480,\
             -1579592562,    -2843826,      -67254,      -96798,   -95437789,  -814722916,       -1789,  -629503946, -1275619672,  -798092267,\
             -336713003,       -4530,     -305938,   -72669894,    -5444868,       -7035,   -26673828, -1181693510,  -682757121,      -53853,\
             -946597, -1902883305,    -2533950,     -445311,  -485915587,  -503137969, -2051051543,  -280400040,        -117,     -979843,\
             -404261360,       -2768,    -2780710,  -680799696, -1493273522,  -524667043,  -995351494,  -276288732,       -6012,  -218602956,\
             -70028119, -1098157704, -1028142944,   -17956729,  -955727863, -1739803929,  -160547000,  -170345078,  -602592290,  -298065675,\
             -965207908,    -5437698,      -95479,  -657644103,  -579594414, -1416107324,       -4834,  -264106378, -1747980631,  -890564206,\
             -754336280,  -124757891,  -403128914,       -2572, -1846719126,       -8380, -1636879871, -2038324594, -1565801616, -1589833939,\
             -54854928,     -412392,        -347,      -54427,  -740595439,     -308142, -1417086683,     -357182, -2054262476, -1292746198,\
             -156969736, -1109369026,    -5303507,     -279606,    -5230480, -1556161445,  -178558738,  -491294083,   -92185025,        -151,\
             -256539102, -2122563737,  -905661892,      -86869,   -68126305,   -10134546,      -69351,   -49530609,  -121701366, -1672879798,\
             -31868842,  -294054153,      -44461,  -751795324,      -25639,     -343680,      -47633,  -936334000, -1364283070, -1312063259,\
             -1975466888, -1758416345,        -321,     -529683, -1621293156,  -592617721, -1414525842, -1054803025,   -51716648,  -809947008,\
             -54231,     -930105, -1913785678,  -157789067, -1914921164,  -351940762, -1389196396,  -787117736,  -101570882,   -40783047,\
             -1002924109,     -208223,     -922561, -1583247886,  -270678998,  -725864499, -1768889589,  -858772667,       -4655,  -220649151,\
             -1036381768, -1414443149,    -6702742, -2110197606, -1715234096,      -76947,  -106721345,  -134838858,  -702786068,  -201652962,\
             -999202677,      -29859, -1392122382,  -495575174, -1859678295,  -341351380,       -1485,    -1371885,    -3820433,   -16126473,\
             -8280953, -1347513041,     -927915, -1253585849,     -556246, -1941965196,     -917657,      -95706, -1553152977,  -717477164,\
             -34483711,  -475447600, -1976550765,    -3682193,       -5936,     -313854, -2133510727, -1020228476, -1189020113,      -67095,\
             -1382778602, -1866632682, -1340567479,       -5310, -1924657707,  -669243692,  -119121280,   -95621807,  -928983094,     -433009,\
             -903720614,    -2316170,    -6391789, -1627538539,    -3651228,  -743036685, -2119040446,      -65635,      -43125, -1289019222,\
             -3760292,   -11455303,  -423186446,       -4427, -1034748494,    -7062403,       -8886,  -314431669,  -608807184,  -926855596,\
             -834182996,   -27592843,   -97089961, -1610386327,  -433000835,    -4908114,      -42216,      -21965, -2130389580, -1202960443,\
             -1591755040,      -61498,  -792448123,    -8492859, -1262032516, -1043820587,        -998,     -736177,      -22314, -2103911757,\
             -1326471416,      -85345,  -163937324,    -6473663, -1628181068,  -117486701, -1935878830, -1382814165, -1342834382,  -824670268,\
             -682606066,   -50373458,      -43479,  -516406232,  -864784659,     -663151,  -501841315, -2077847023,  -449188870,  -693428335,\
             -1327867901,  -775029938,    -6949273,   -12755963, -1922896806,     -988649, -1628158171,  -162229224, -2095145734,     -438345,\
             -1085214057,    -8890533,      -82673, -1396794972,       -5336,   -51288397,       -5952,    -1927787, -1588625059, -1265747686,\
             -536356667, -1130597678,       -7234,  -750718504,   -17870126, -1351168900, -1923183253,       -3748,  -340734593,     -902021,\
             -1738350721,      -11778,  -220326379, -1593723177,  -464204707,       -9810,       -5290,  -226051762, -1360604137,    -1577548,\
             -1797881,   -22261350,    -8502815,     -421655,  -322796122,     -767008,       -8429, -1994552434,      -55217,  -728681314,\
             -689897686,  -166628601,  -380657482,  -430471024,  -442455323,  -843319743, -1570804360, -2056711681,       -4766, -1303214043,\
             -7675, -1857832734,      -65053,     -123992,  -100868734, -1163294343,  -954269444,  -703248416, -1050688887,  -788831164,\
             -443045,  -485130571,  -450712491, -1534437071, -1348988495,   -36233454,   -24982344,   -81274264,   -81937706, -1648394983,\
             -459482248, -1871059852,     -575717,  -980273648,      -85098,   -24330621,       -1135, -2043651809, -1755158440, -1698090650,\
             -4412636,  -199280370,     -166654, -1792858707,  -668759450,  -996870452, -1930470652,  -919903698,  -174541873,     -349583,\
             -605985620,      -50354, -1890133899,  -956401733,      -32811,      -35124,  -536802269, -1911710228,  -597131010,  -695553199,\
             -50673315,       -5921,  -664874727,       -3234,       -5587, -1761271840, -1104415128, -2011497549,   -59904126,  -722890650,\
             -589166,      -78673, -1742294154, -1581901755,       -3222,    -5203540,      -51364,      -81337,      -23413, -1130353620,\
             -985788227,  -541444932, -1316704555,     -628598,     -683392, -1095485447,    -2102035,  -628845457,  -986878341,  -830828147,\
             -1904132143, -2005340894,   -71923826,       -8401,    -7604136,    -7612932,      -89243,   -50626976, -1796087660,  -575602535,\
             -1676852383,      -77264, -1772227972,     -121831, -1199698718, -1520295538,  -472939417,  -348176342,  -253770225,  -625972398,\
             -990295559,       -5156, -1033174040,  -190270854,     -746107,  -212589817, -1600760131,  -609772435,  -896588738,   -45570352,\
             -808060, -1900172960, -1551552367,  -473364487,  -416149889,    -4356335, -1683475133,  -621757768, -1159492504,  -408950572,\
             -2035671228,  -566733927,   -58975308, -1236617348, -2113626682,     -573989,     -653108,  -108776996,  -992967852,    -1664198,\
             -66498,   -93293345,     -790841,      -27372,       -4593, -1579098525,      -87117,  -423504778,       -9113, -1460223717,\
             -6205,   -17263301,  -649045454,  -169222124, -1122455370,   -35023008,  -456318297,  -231013678, -1182198304, -1848301101,\
             -581021735,      -31806, -1947875087,      -48811,   -82730364,  -170631961,    -9363247,        -390,  -349086961,     -826603,\
             -403925611,    -3201004,       -7817,      -32297,  -208524803,     -974683,       -4960,   -11044164,       -2499,  -254582312,\
             -382531,  -355979441,   -92892301, -1274627384,  -874914119,     -823172,  -897759252,      -89888,  -753090611, -1621828897,\
             -1134743766, -1677176703,  -405884657, -1431055323,  -546542161, -1006510752,  -165086235,       -7265,  -372287713,  -902380287,\
             -344737623,  -506507813,  -695520216,  -216689038,  -821969242,       -9423,  -690427017,  -839792839,       -1235,      -24462,\
             -1748457450,   -25585057,  -469318892,  -387473434,     -545039, -1448763161, -1996011402,   -71948160, -1680458928,     -298272,\
             -306958541,   -22248076,  -274332828,   -96674514,    -3862551,      -53823,  -755719626,  -287326671, -1663068818, -1332011179,\
             -6723,     -400474,  -771315899,  -472239641,  -401674600,    -2016348,    -1739564, -1650593950, -1444643887, -1858265428,\
             -987553,   -71484679,     -817388, -1107431121, -1717375799,   -72360468,  -985202147,  -174639287,   -25449397,  -170356032,\
             -1052,  -110956771,  -888072143, -1647780815,   -13359130,  -788295212,  -849982348, -1446149673,      -78110,  -612806201,\
             -266114131,      -39081,  -932070534,   -26452060,   -52978120,    -3680549, -2015906236,  -298973606,  -555602632,   -23115162,\
             -1168338031,        -836,     -927053,  -547830363, -1624118417,    -7926639,  -776855879, -1315767748,  -727040520,     -691665,\
             -8098,    -6246860, -1274360137, -1774702835, -1643581953,    -4205556, -1005775442, -1186864905,  -658395637, -1044879521,\
             -73739674, -1427945575,  -833033561,  -246955860,      -39443, -1337943752, -1255310678,  -257300116,      -28273,  -593244110,\
             -1676581654,  -115544599,  -720117545,  -450094058,   -87363071, -1563715542,    -6945087, -1571378596,  -251114093,    -7512557,\
             -560,    -3431325,    -2549414, -1162676063,    -2235432,      -23115,  -356296356,  -244435157,     -844980,     -621524,\
             -194332820,  -644567971, -1382838908, -1333657869,      -54657,   -96537762,      -97544,     -808949,     -367563,    -2313786,\
             -354437732, -1454412954, -1573162440, -1451062143,       -6491,       -9774, -1200552429,     -792554, -1427244440,      -11900,\
             -372494037, -1854176982,       -4808, -1859622427,    -3949754,     -728829, -1759619686,      -11327,     -210699,    -2592352,\
             -630913986,      -18157,  -610521899,     -442123, -1102013075,  -790558531, -1170751541,    -8799949, -1083619674,      -54022,\
             -745555265,  -749613631,  -112730766, -1060944741,   -69431063,  -267189516,   -87516439,       -7428, -1839027231,  -950346115,\
             -351214832,     -111795,    -2852457,   -23607873, -1337073145,    -3652374,  -951109258,     -200750,     -773204,  -204188998,\
             -549928662,    -6649615,     -609650, -1715497512,     -974521,  -841227083, -1321289085,      -80319,   -73390602,     -230472,\
             -5910275,      -98827,   -44988155,  -227377449,  -692135836,  -522751466,  -627476962,  -236418027,  -865620180,    -8897194,\
             -4187,  -626247326,   -50132521, -1905947383,     -947182,   -95436672,   -31953662, -1588836935,     -277365, -1777917676,\
             -8693962,  -911768399,   -85316189, -2042139115, -1533536463,     -641371,   -96293648,    -8313147,    -5836025,  -852326563,\
             -2132615087, -1533678872,  -576475359,   -88571578,       -7034,  -122058290,  -278875886, -1376015825,  -913073459,  -125537024,\
             -52465, -2138016383,       -8396, -1239772051,     -551699, -1383497308,    -7474271, -1328632984,  -338682429, -1201496139,\
             -31453797,  -389527353, -1239890557, -1965920572, -1487764588, -1926307349, -1956644693, -1473967872,  -575514780,   -95324183,\
             -6101,       -8455,      -37057, -1112508469, -1547414925,  -504705244,  -820039086,   -65608616, -1989310494,  -916723597,\
             -653405457,  -223550692,       -9738, -1706722481, -2077674541,   -79238019,   -75685310,  -477638202,  -392078770, -1155587050,\
             -1323, -2089511588,    -5304923, -1373029780,     -890535,    -1466888, -1712681801,  -147837407,  -882781239, -1772505634,\
             -118349341,  -631731191, -2043687939,  -686017242,   -83644266,  -625594081,  -215885593,   -77404034, -1853570388,  -212160411,\
             -1361762460,    -2933789,       -2670, -1340938956, -1875806490,    -6818209, -1966882339,    -9151197,     -441746, -1880659089,\
             -68352,    -4176642, -1752821099, -1344135713, -1201807191,  -515364500,      -66189,    -9214782,  -745511155,    -4171367,\
             -1855700764, -1097940858,   -36889881,    -8493776,   -97598361,  -929089847, -1952023659, -1363561555,       -6889,   -13494545,\
             -992645651,       -8039, -1336896506,  -941674758,  -751292988,  -229607029, -1203263326, -2096856577,  -390357707, -1008819228,\
             -43398845,    -9917963, -1679005933,  -515198790,     -315977,  -469468427, -1841511274,  -338300277,  -428760817,  -527319918,\
             -5387, -1015208729,     -355300, -1413953532,     -894355,     -212336,  -845748030, -1358243731,       -2985, -1168397869,\
             -15024536,  -172506612, -1889725475,     -746529,  -881367824,  -482486888,       -9540,  -753056214,  -178026510, -1354443190,\
             -1578726745,  -206576328,       -3989,  -519906881,   -23322249, -1312004871,   -46040975,  -356096437, -1090333084,   -97746848,\
             -7118039,      -11185,  -570053952, -1991975108, -1837816475, -1736083060,      -92506,   -90423804, -1433573694, -1328178301,\
             -4388856,       -6527, -1963187916, -1722128055,  -725528894, -1043664331, -1106473474,    -2407452,     -990969, -1512968284,\
             -1783772285,     -383230, -2008689284,      -73762,  -754681461,   -40653811,  -677865804,  -808109612, -1180182518,      -36045,\
             -1115194009,     -185981, -1711945816,  -577480902, -1531812947, -1178972882,    -1455161,      -76772,   -85004009,    -6955377,\
             -1340507124,  -974056365,     -648669,      -23448, -1870768476,     -448430,     -290188,  -683133773,      -96088, -1733012303,\
             -46180,       -3682,   -20016271, -1800504717,     -785560, -1544810870,  -593203737, -1084806626,  -529768092,   -96037169,\
             -2488035,  -188435920, -1244566759, -2108494139,    -8873066,  -962347399,     -269251,  -695711227, -1326551178, -1644170944,\
             -81228670,  -279802594, -1277249271,    -8886489,      -36598,     -590789, -1030705270,    -1740963,     -294256, -1325581583,\
             -762364860, -1180143821,   -99517101, -2067909634,     -127355,  -927154634,       -4283,      -99981,  -892963753, -1419604403,\
             -693515948,  -991925473,    -3221126,      -82380,     -733832,   -91440983,     -229863,  -161596600, -1512468672,  -899802957,\
             -31839599,   -65934695,   -83006490,  -945044454,  -686349370, -1423828644,  -558848275,   -99166761,  -497033374,  -862755844,\
             -167643094,  -451009269,       -7533, -1693360475, -2043588120,  -597203242, -1753437611, -1461411380,  -599072578,  -641268558,\
             -7344,  -285046643,  -338311914, -1808068702, -2079857480, -1125747745,  -844850890,       -1316,  -931720018, -1558194091,\
             -2132353907,  -339816867,      -44352,    -4012337,      -22757, -1110116368,  -722278480,    -3580521,  -713965465, -1622020218,\
             -1559222697,  -511283183, -1493273172,       -4250,      -36943, -1290493362, -1640912318,  -512493634,      -31395, -1244799059,\
             -59730425, -1560937856,     -104998,     -788498,   -60797886,  -638081062,    -1101247, -1211894368, -1630269018,  -161042741,\
             -1861835,  -381975847, -1833863571, -1228369259,     -765387,  -647748524,  -359446443,    -9905622,     -554408,  -274584714,\
             -1511671636,     -671348,  -798166627,  -449515347,   -98813785,      -84794,     -333539, -2075518714, -2106937822, -2116372902,\
             -949046,       -4380,   -47096721,   -94162186, -2085998189, -1491484331, -1528907711,  -187141790, -1410747163,     -927477,\
             -49370984, -1290755727,    -8401516,      -91258,     -896263, -1093412590, -1185283368,      -84144, -1149948287,     -989400,\
             -880,  -950065167,  -473092129, -1565149296,    -2754230,     -611218,  -869181851,  -925583485,  -456831175,  -390286649,\
             -994363556,  -158320450,  -645931438,     -871889,  -196759813, -1791561611,      -67636,  -410024515,    -7572703, -1157530977,\
             -1242338873,   -27445593, -1077965499, -2059698592,    -5778971,     -204075, -1611512886,     -534865, -1876193657, -1886000657,\
             -1621171160, -1888852116,   -89380967, -1252109619,  -690546245,       -3494,  -990884916,   -81270231,      -67926,      -27694,\
             -1273166181,    -2231752,    -7642913,      -91596,  -968532942,       -4917,  -126331629,  -856865955,  -238355164, -1486909998,\
             -359452999,     -509940, -1295549024, -1132988513,      -47241,  -131393371,   -52136384, -1213725540,   -78112496, -1332116901,\
             -1273263733,  -573086320,  -796620557, -1508113032, -1803905019, -1218868426,  -622899828,   -39405961,      -82298, -1912638487,\
             -2107608696,     -160833,   -65224276,      -65219,  -506581737,       -9654,       -8449,   -96746267, -1143730208,  -820294493,\
             -1394989505, -1629800168,  -980774818,      -67257,      -21756,     -159162,  -214905348,  -619430564,  -717921397, -1010373359,\
             -1768490515,       -5599, -1178791786,    -3486375, -1985964521, -1779527248,  -736308980,  -858814328, -1873107927,  -213681266,\
             -1266744813, -1547710705, -1068401266, -2009104322,      -21883, -2056206773, -1855214094,  -102860856,  -684765544, -1143267555,\
             -2003819048,  -941561808,      -15197,  -298733345, -1830480197,   -41849177, -1990438322,  -466061901,    -3619933, -1402490068,\
             -1902052488,       -9214,    -4680923,  -753915702,     -325674,  -911053839,  -493594546, -1131476361,     -349183,  -972955804,\
             -1708844883,    -8959398,    -2036756,  -706267742, -1739023988, -1022912637,   -26220270,      -95981,  -175127513, -1792938496,\
             -614066668,     -938184,  -270297004,  -270339889,  -592637154, -1084493636,  -858195742, -1507770959,     -114974, -1111032639,\
             -49385170,  -781321663, -1388004988,       -9126, -1788662563,  -135357992,       -7168,  -655709952, -1785451728, -1965705334,\
             -70904433, -1547768545, -1594080401, -1951405694, -2058578547,  -239865081,      -67998, -1755164939,  -291910095,   -22455620,\
             -821866, -1091016326,  -524586176,   -77735342,      -35967, -1818469154,    -9636275,  -884956103, -1836113427, -1429258349,\
             -764809313,  -248546951,      -78691,  -733722996, -1261259402,  -932626997,       -6353,     -839926, -1586200731,  -110662733,\
             -3769,     -534752,   -27648852,     -273977,     -437166, -1558636876,    -3468970,     -723514, -1948447427,     -366442,\
             -5357619,  -194834876,  -613519167, -1985251600,  -188065909, -1847402315, -2041314532,      -28349,    -8447385, -1959229820,\
             -4608582, -1173963631, -1781237304,  -384709100, -1901923671, -2075711152, -1066437313,     -544333, -1408551123,  -812907934,\
             -22382395, -1754034715,    -9284299,    -8672259,       -2935, -1842396436,     -167748, -1508896233,   -55746042,       -3300,\
             -436541,   -50248384, -1349331605, -1285738499, -2034723777,  -274374451, -1752001813,   -19357500, -1727819007, -1856663237,\
             -433540372,  -916349168,      -67791,  -362098348, -1637839994,  -934812235,  -160826739,   -83887644,  -995430712, -1325640166,\
             -8033,  -920864838,   -95709659,    -2246336,  -831901898,  -745265710,   -42128544, -1170315576, -1211442422,   -30019828,\
             -1153267679,      -40786, -1488464358,   -75556799, -1770190834, -1971926554,  -610764727,  -518653327,      -40937, -1286152673,\
             -966935,  -230025125, -1150452374,    -3332094,     -452173, -1887494430,   -63212057,  -580536527,     -409790, -1254000570,\
             -2046224851,    -4386383, -2011224659,  -138023734, -1935398245,  -565278629,       -7151,  -875064179, -1151959230,    -8618080,\
             -232998962, -1187321568,    -5291416,  -925934282,  -511806095,      -87229,  -748704015,      -92941, -1906516117, -1003962142,\
             -1641091537,     -957442,  -287444451, -1070761643,      -23524,    -5826336, -1644878293,   -34440900,     -291763, -1352907494,\
             -867180679,  -212564657,   -82699003,   -75866598,  -267364664,  -497582240,        -156,  -812099499,      -99324,       -1661,\
             -8412,      -98626,      -27461,   -34617773,  -616361887, -1682686568, -1118617433,       -3435, -1081569366, -1644606200,\
             -1984104690,     -945313,  -186647325,  -766728636,     -218138,  -774547614,    -8959964,     -438906,  -791630623,      -22894,\
             -1780973421,   -10157729,  -789676018,   -54318362,   -66346907,   -85983591,  -521644552,  -133242940,  -696909518,      -34992,\
             -127295131,  -931875421,     -155090,    -1213300,  -147729308,  -394585864,  -612591671,  -236660844,  -636706764,     -369495,\
             -6616, -2097719624,       -5870,      -33023,  -358223688,   -42474395, -1057089919,  -326414209,  -419563337,     -342512,\
             -1982837371, -1977786516, -1391704076, -1634169017, -1615063873,       -3317,  -868632019,  -321290121, -1516131132, -1247468627,\
             -45863,  -746017048,  -241315361,  -184640173, -1951883824,      -19087,    -6497488, -1497655703,  -148664396,  -572158425,\
             -9058189, -1257837879, -1920114514,  -992106364,      -67413, -1930452677, -1874691809,  -162536525,    -3033848,  -171407032,\
             -800836011,  -881942901,       -2744, -1868943425,  -270068828,      -41470, -1205336234, -1438931386,  -586269128,  -941016041,\
             -1932,    -9596378,   -43873574, -2023759077,  -455575012,  -654945586,      -58835, -2007740498,      -76892,  -718999144,\
             -761236063, -1694879016, -1548248306,    -6246771,     -148150,       -1914, -1962002805, -1158113206,     -175312,     -867974,\
             -64404402,     -767499,  -430749671, -1484451975,    -9093169, -1991233477,  -414795483, -1248707206,   -92077989,   -38338174,\
             -1153754678,   -16466953,   -53144509, -1060986890, -1101474790,      -60413,      -69020,     -620090, -1166647988,  -212501251,\
             -865695213,   -32844640, -1687375196, -1857976368,     -480773,  -231925400,      -62402,  -242180302,     -985478,       -6640,\
             -1330517351, -1921449117,  -214494574, -1429600729,   -26174966,   -24119413, -1487984506,      -27066, -1539867461,     -259815,\
             -9688,  -319119685,   -41475451,      -87826,   -83456561, -1819654172, -1250815684, -1666402463,  -247932340, -1634783359,\
             -63021385,     -710494, -1906980350,       -3088,       -8800,  -108457547,  -169938386,  -923759353,       -5771,  -199242278,\
             -430437584,  -972117641, -1832017222,  -826902948,       -3671, -2029617984, -1538831827,  -401424105,    -4718355, -1301568332,\
             -1142762050,    -5508861,  -498208580,       -3074,  -823726295, -1287570109, -1488247148, -2142307644,  -687982702, -1389527512,\
             -40782981,  -228314526,       -1511,   -21307173,    -5868401, -1344296963,      -73125,    -9622831,  -928961761,     -440130,\
             -220227698,       -8455,  -490016119, -1729746685,  -869696608,  -672480916,     -952419,  -280514070, -1785802395,    -3109724,\
             -37303, -2087999037, -2131610931,  -439541841,  -546607261, -1570376542, -1437452732,     -170580,     -192105, -1089565654,\
             -396944,      -24354,  -293560538, -1584471888, -1309172979,    -3635653, -1375316248,  -387023762, -1549621277,      -99186,\
             -470380,  -467896776, -1384890245,      -59607,       -5597, -1666966405,  -201283326, -1627330610,     -513833, -1861645570,\
             -635391531,   -63968783,  -333248507, -1874188141, -1916260567, -1844582717,  -688759758,   -96324186,  -925828529, -2057269957,\
             -596554,  -288721618,  -423844066,     -282503, -1452672917,      -49820,  -945883105,      -90882, -1661774903, -2002924724,\
             -5325,  -950671503, -1654784370, -1778214830, -1251623945, -1163587197,  -403932251, -1812821837,       -8459, -1910119467,\
             -518001833,    -9483761, -1566816473,  -227971317,  -267785396,    -2589288,  -889564919, -1045046733,  -294921223,      -50244,\
             -1701309482, -1735950416,     -578037,  -527292586,  -179873249, -1573185807,  -570697788, -1117378570,      -29968,    -7386928,\
             -706245066,  -711397701, -1740927987,      -16365, -1973036416,   -15941367,       -1433,     -418038,  -286244235, -2113425566,\
             -1471774614, -2074511451,     -428951,       -2964,  -953204581,  -371851145,    -9990167,  -238259910,  -665163188,  -217920939,\
             -7180,   -53295374, -1136303635,    -9047571,  -220646184,    -6529190,  -997109428,     -357660,  -350536634,  -865994607,\
             -1393054226,  -404669465,   -26793327, -1732701862, -1486318286,   -42993924, -1189708126,  -510950270, -2080418771, -1995554215,\
             -696467, -1464881143,  -107020082,     -707805,    -4616415, -1509000700,  -979200692,  -611348632,      -99654,    -7369545,\
             -786562654, -1012438426,  -368135436,  -319251485, -1314166252, -2108921732, -1598251908,    -4386220,   -37285542,    -6584910,\
             -200722449,  -602615522,  -373517266, -1261473386,     -823731,     -355015,  -616275478,      -86322,       -8144,  -752791861,\
             -820232247,   -83947339, -1980709928,  -146022820,     -852901,  -433297104,    -8816372,   -16367651,  -164319651,     -672923,\
             -71384, -1299750486,    -3581733,  -491928733,  -307789164,  -130455626,  -268634060,     -478043,  -670374817,   -52584733,\
             -1013659216, -1599202371,   -84772497,  -944041710,   -31990060,  -595510740, -1078264540,    -7439727, -1437143487,  -447360393,\
             -70667, -1287324941,  -800363031,  -591609281, -1983431031,  -903694597,       -5080, -2115605976, -1734113398, -2010929669,\
             -3035,  -292662974, -1076054898, -1232912249,   -87736351, -1334500605, -1757507496,     -575024,     -360053,   -56990025,\
             -2048539763, -1025767344, -1349826659, -1089409214,      -78937,  -422694413, -1516932644,      -32537, -1324507745, -1126293254,\
             -1288545057,  -880298463, -1495987344, -1894679574,  -203149793,  -632967848, -1828375689, -1689090561, -1722941435, -1680842801,\
             -1635559299, -1933492507,    -7124393,  -821346551,  -155720285, -1408135875,    -4098399, -1634581736,  -108902709,   -74257637,\
             -2005104439,  -225420446,  -620636379, -2054308666, -1611094556,  -631153020,  -949947649,  -654911484,       -8808,   -20003334,\
             -1555254878,  -386500313,    -2060263,  -748225886,   -79896503,       -5126,     -237688, -1827728418,    -2856920,      -96857,\
             -1553676213,      -92004,     -666974,      -32578, -1407699357, -1679445445,       -3735,      -90757,     -468118,   -27219859,\
             -5247,     -653482,    -8709714,   -64287328, -1635571577,        -540,   -34439697,  -714554114, -1621807794,  -114929645,\
             -24859700, -1506650989,  -270707096,       -4378,     -806323, -1223912876,       -7656, -1910202386, -1656823347,  -633737864,\
             -994620207,    -6148878, -1178127318,  -686996301,       -1819,  -565349001, -1193167408, -2104182860,       -7099,    -1850953,\
             -818804948, -1091314283,  -865323223, -1522876757, -1881885017, -2001263868,      -85063,    -3982549,    -1923807,     -699212,\
             -1214963581,    -8387646,    -9009173,   -92429979,  -447231568,  -183453250,   -62795810,      -67105,  -957912122,       -1510,\
             -96871926,    -6434142,   -58158943,  -746535685,  -135428290,      -85393, -1792032211,     -543078,    -8968228,       -5163,\
             -1039956, -1072901855, -1300657561,       -2929, -1994856556,  -436058223,    -7641069,     -225350, -1230906537, -1036313611,\
             -43655539,  -462883081, -2065803024, -1319154861,      -22120, -1942371855,  -820505799,  -418989086, -1207224962, -1047996816,\
             -1608015,  -899662262, -1827554941,      -98634,  -436148757, -1821775650, -1412913458,      -98285,      -86500,   -19389503,\
             -30256,  -987462792,   -43617202, -1028759726, -1178391951,     -669270, -1632222891,      -23587,       -7245,  -196166501,\
             -1680408729,  -117739676, -1249535166,  -835714137,       -1939, -1703343228,    -6471960, -2070087924,      -42188,  -360834150,\
             -935417015, -2016916310,        -813,    -4471262,   -26080918, -1752009714,  -491802384,     -784141, -1823641730,     -960016,\
             -49573, -1109910737,  -388515548,       -8126,  -470904453, -1324911580, -2073413946,   -97096382,       -4258, -1547040020,\
             -496653292,  -364120690,  -477988765,   -66303960,   -98806653,  -508740833, -1051688056,       -3466,    -4423770, -1189090842,\
             -916351270, -1486457173,  -591031241, -1987323528,      -75441,  -290158076,  -772933236, -1271333131,      -16823,  -694118505,\
             -1472221723,       -8982,  -860971316,  -354583960, -1726398223, -1831124472, -1881801236,  -162217207,    -6362776,      -50323,\
             -660436890,  -781004315,    -2820824,  -697987630,  -904125178,   -32481445, -1368347146,   -44641484, -1135589346, -1660200004,\
             -1388256493,  -326805486, -2140626998,  -691747096, -1380554057,    -6978802, -1031816257, -1180452212, -1403933705,   -92113824,\
             -2052539401,  -730842320,  -731367624,  -162508766,     -405304, -1450612936,  -841366887,       -9854,  -675482839,  -427585632,\
             -6559,     -768389,    -6016835, -1610094173, -1387372472,   -24328339, -1164965855,      -76615,       -1786, -1181010771,\
             -901010863,      -71647,   -98931494,  -315826059,  -639268064,  -195840417,  -897029110, -1137745128, -1483595794,  -377343517,\
             -928133824, -1327651062,  -763242275, -2068068672,  -645160128, -2040599847,      -46385,   -76610402, -1451543916,       -1712,\
             -1952718492,       -8302,     -658750,    -3252253,     -887359,  -428853124,  -500002746,  -556718969,      -19046,      -74582,\
             -1161850814, -1657445156,   -38936439,  -205567481,  -451578372,  -392153328,   -93111954,     -474829,  -862695928, -1493469316,\
             -450386149,    -2255651, -2110521278,      -23375,   -49022523, -2056113487,  -567759366,        -853, -2035167708,  -623683296,\
             -1981551436,      -83721,  -821042968,  -601809505,  -338313458,  -807962443,    -7680938,    -2505361,     -308062,  -320547523,\
             -3974, -1248847024,  -539545131,  -725720091,  -581048347,  -631840302,       -2467,  -438072604,    -8103991,  -734967416,\
             -83124428, -2033219617, -2004905572,  -601366840, -1742738177, -1301799582,    -7932647,      -39812, -1642952601, -1594119317,\
             -144607025, -2000235667, -1559791899,  -294452793,      -15551,   -29785346,     -280644, -2034190166,    -2827339,     -399031,\
             -6960,   -28063167,    -1802112,  -865182865, -1668702525,  -610248898, -2049085279,  -142135630,  -164980540,  -601226543,\
             -40962377,  -925188341,  -413306715,  -306179442,    -8112628,         -14, -2113636292,   -23599923, -2097512381, -1551403137,\
             -343200824,       -1823,     -882647,      -93484,   -63671354, -1071733035, -2101292409,  -475858903,  -568366506,     -396072,\
             -1602415689,       -6876,  -232303116,   -39605535,   -12949019,     -991807,       -1735, -1733645211,    -5514659,    -5296299,\
             -5854,    -7491968,   -32641251,   -75128479,      -94598,  -919651100, -1833654667, -1658015144,  -119394437, -1057991111,\
             -624159757,  -179628472,   -42072841,  -789786589, -1330693223,       -7349,    -8517766,  -936829411,   -72536710,      -45608,\
             -411079053,  -618441059,       -9183,  -361469385, -1062431540, -1189709487,    -1484229, -1268489815,   -22479018,       -6149,\
             -436695792,  -626347310,      -30261, -1249683759,       -9300,     -348768, -1720567261,       -7875, -1651934804,  -316521809,\
             -9063963, -1908028375,  -163968111, -1861147378,  -779862822, -1032481043, -1233115135, -1248144424,       -1151,       -9360,\
             -1637296596,    -2544245, -2122332085,      -29322,   -13530418,      -36621, -1476895910, -1476836496, -1494896845, -2019504574,\
             -895029613,   -87843842,  -322664697,  -806749407,  -588187092, -1394676974,  -137847962,  -787951075, -1268046726,       -5863,\
             -18929,  -215871124, -1777203562, -1703419034,  -179984077, -1818783370,     -679141,    -6407835,  -366248814, -1615471058,\
             -1078921292,  -168417562, -1983552511, -2064219791,  -631669585,      -36736, -1183510089,      -84608, -1341256176, -1373220730,\
             -1693939010,  -989945475, -1259555715,  -954385551,      -38635,  -483798957, -1388168086,   -35054481,      -94276,  -225247056,\
             -172370974,  -865254725,        -653,  -433036012,    -6108939, -1116698794,  -415577661,      -33117,       -8790,  -341353034,\
             -50404072,   -37582270,     -670320, -1718804120,  -886221804,       -5783, -1043019190,    -1625167,      -57601, -1268429415,\
             -305472721,  -586105645,  -510897636,  -560473671,  -533386737,       -3162,     -151254,    -6574901,    -4681562,  -499765334,\
             -71671986,     -671270,  -904081239,  -956162598,  -849269448,     -933460, -1409824605,    -6759625,  -780264878, -1112075433,\
             -49645,     -187241, -1583161577,    -6330335, -2109538905, -1520347481,       -5690, -1683770766,       -3373, -1506181081,\
             -254742499,  -176705305, -1180190022, -1644405940, -1932113004,  -618185918, -1542896285,    -6281314,  -993158036,      -41666,\
             -437862303, -1196874829, -1972370930, -2089784437,   -23138284, -1197791689,  -827335848,  -399488235,  -326776889,      -24115,\
             -238143105,     -651091,  -812119566,      -63558,  -665396949,       -2101,        -891, -2070447547,    -3397378, -1654318499,\
             -1918088317,  -204803726,     -261897,  -513815341,  -935736495,  -199224445,  -587272944, -1482697473, -1107529900,   -58352396,\
             -457949020, -1368055299, -1714735003,  -847903163, -1559772292,      -92506,  -167824812,  -203299187,   -42911207, -1194072161,\
             -489421587, -2090975546,  -272057391,   -62076573,  -245945705,  -418800866, -2009977178,  -652138140, -1284234910,  -158386998,\
             -7299, -1183786174,      -13094,  -781609886,       -2867,  -332174110, -1099210534, -1015692050, -1214584855,  -956722947,\
             -2078608819,       -2181,      -68204,     -850763,  -115139692,      -83945,      -85593, -1148899642, -2018206055,    -1528809,\
             -1640289698,  -528860089, -2131417268,  -170209054,  -496823366, -1628053957,   -19893436, -1152948211,   -68792446,      -11505,\
             -608274058,  -225050242,  -509937778,       -3002,     -368314,   -94343400,      -93515, -1801890023,      -44706,  -874136257,\
             -1546532965,   -99667782,  -802654340, -1727382026,  -522291963, -1101187095,   -95624669,       -4162,  -276096605, -1027773725,\
             -1535202211,       -8056, -1016484615,   -45771487, -1926558302,    -9596416, -1707439532,    -2182880,  -333328270,  -803355064,\
             -2011960074,  -277935244,  -258068654,  -415191894,     -823212, -1088418467,   -50578307, -1216826612,      -95776,  -505539794,\
             -429333123,  -695366341,     -474801, -2088624552, -2138227240,      -65731, -1347543638,   -31773693,  -326394900, -1534499371,\
             -1646112650,  -572239061,      -44888, -2067728443,  -855218781,       -6918,  -795645929,  -567618773,     -870046,  -894183907,\
             -1429974742,   -17419537, -1321639517, -1255684440,      -86153,  -495275720,  -125953131,  -304523399,     -296820, -1938604279,\
             -154309120, -1701987616, -1573279191,     -576898,     -684774,  -380701705,      -55532,  -206790523,  -724441156, -1377099524,\
             -7004222,   -10832940,   -38722984,  -986269391,  -149580339,  -989677303, -1364547914,  -919099558, -1398932884, -1816991326,\
             -39697299,   -62645681,  -751633574, -1718053716,  -479643191, -2039538853,    -6914678, -2138466504, -1232834221,  -667085530,\
             -831500,      -32259, -1506319085,      -37122,  -226974468,  -850083373, -1982293817,    -7444715, -1656108942,  -440450410,\
             -121963063,  -497871556,      -34505, -1655707150,    -7626739,   -73821614,    -5471323,  -860780455, -1129640215,  -789459462,\
             -673618,    -4708436,       -4568,   -60177350,       -4709,     -123797, -1275269155,  -606349166,  -170201918, -2142091348,\
             -1778193952, -1625813644,  -585220070, -1690888138,   -16816713,  -938733803,  -496866160, -1152543605,       -6331, -1664486069,\
             -7757,  -445107991,  -873866115, -1434909355,  -350139908,      -52152,       -9072,     -221904, -1960151370, -1739197728,\
             -6444,    -1180870,  -233879810,  -141396089, -1173507151, -2046328695, -1665185503,  -707951783,      -65321, -1987860486,\
             -410423077, -1235628569,   -74567999, -1720870837,  -960361066, -1819697941, -1901265643, -1317766317,      -62803, -1457827888,\
             -99305633,    -6776723,  -993575122, -2118095377,  -854583464,  -732992264,  -751039619,  -786304182,       -5985,  -975890138,\
             -735795885,   -44393659, -1248109412,  -832255666,  -316002525, -1025382116,  -456213047,  -674443682, -1029111212,      -70144,\
             -1260123887, -1002683670,    -6896436,      -51702,  -987102970,     -221069,       -5060, -1082920401,     -812000,  -503220052,\
             -28592293, -1008752407, -1788164657,    -2211065, -1929657779,  -843961949,  -917696374,       -2533,  -713602617,  -230607889,\
             -3912992, -1653208977,      -85600,  -486821318,     -494254,    -9786433,    -3787696, -2058618216,  -973698587,       -3022,\
             -131025717,  -193685983,       -2387,      -30991, -1641964453, -1907278791, -1882551882,  -571590160, -1252796130,   -30019834,\
             -733088999,  -624548146,  -270229931,    -2027265,  -578236640,       -8590,  -954573019,  -914444110,    -9430086,      -59789,\
             -481310067,  -607805949,       -4799,     -125351,     -504794,  -610408451, -2005353117, -2040247263,  -238224800,    -7733894,\
             -456312300, -1059537528, -2003649744, -1411918587,  -842249387,  -644766848,  -271686148,   -77358952,     -666883,      -17603,\
             -373294726,   -45502081,       -9533,      -93907,  -934611280,   -53746384,  -329933396,      -58676,  -531436024, -2072192979,\
             -359963669,      -36269, -1340507841,  -635914502, -2094660234,     -746348,     -365561,   -71689197,       -9307,  -223291285,\
             -1426295319, -1250142357, -1831523143, -1790177909,  -989317075,  -472842445,    -9627901,  -386987969,    -7814278,    -5965634,\
             -68581389,  -856007046,  -580164626, -1710917253,       -3611,  -634014017,     -777655, -1396240690,  -365851947, -1683779594,\
             -1981304209,   -24638060,     -326332,      -11328,  -882454213,  -874266195, -1602906484, -1481512173,   -92223680, -1966899550,\
             -225024147,     -630859,   -93302346,   -82350246,    -4608860,  -828183497,    -5010240, -1425961640, -1561192223,  -674534195,\
             -189111520, -1425572068, -1644711282,      -85296,   -81406687,     -873334,       -5926,   -13283765,     -583601,  -984705137,\
             -78441, -1797142814,  -894382767,  -889605285,    -6781387,  -491265727,  -753761208,  -648918290, -2110442161,     -475438,\
             -9966, -1546977795,  -605590107, -1015958156,  -567072508,     -721470,  -162112838, -1719679046,   -15674690,   -87595373,\
             -49094530,  -461442788,       -2117,  -493063176,   -98957039, -1075943318, -1617117262, -1472249028,    -3081123, -1208642628,\
             -976592670,  -390993180, -1554396326, -1599580047, -1455289508,   -35985856,  -911831540, -1832131810,    -2185550, -1689247767,\
             -209521544,       -1984, -2039009002,  -678083015, -1279397213,  -968188451, -1842229467,  -811274790,  -733972157,  -784570212,\
             -2702290,      -97450,  -251033015,  -113719488,    -5459668, -1707713062, -1980857174, -1832417134,       -8015, -1769772386,\
             -304783375,  -553049756,   -84724195, -1661261757,   -84294933,      -15189, -1687662089,    -2620963, -1176014503,  -505974204,\
             -1539255953,  -888786237,   -69836108,    -2362433,  -337019264,       -7690,  -448225820,      -69129, -1812115285,  -508278268,\
             -2146401847,      -34933,  -746352867,  -580503893,       -6934, -2075120467, -2028278776,  -719947071,     -433745,  -447324242,\
             -1236612798, -1714957866,    -3749566,  -335080785,   -29151549, -1469754945, -1992139544,  -828605854,    -6765388, -1115835740,\
             -1048320498,   -89039362,    -2307395, -1357463583,       -4596, -1313512372,  -435375088, -1780176625, -1504449305, -1609524986,\
             -146748,      -10289, -1347240318,       -3600, -1870875442,      -63987,  -748975927, -2128757148,     -659721,  -963675487,\
             -1894570305, -1329944012,  -581899300,  -183721146, -1698144191,       -1386,  -596961490,  -789585333,  -376077533,     -188891,\
             -907049687, -1157331254,   -31037055,  -955329760,  -769485577,  -929507356,  -460666646,  -486857107,       -8952,  -871777692,\
             -946053300,  -533001760,     -639353,    -7419285,  -873295034, -1241652283,   -82091821,     -799579,     -454774, -1817820772,\
             -129972,       -4610, -1420607544, -1041214871, -2043842234,   -88870161, -1740139288, -1350211094,  -439897952, -1688182741,\
             -810782799,  -497173317,   -35235614,     -510890,  -354280871, -1411433849, -1092500642, -1163838987,  -683383595, -1319331506,\
             -1203257579, -2112270032,       -9844,  -351016917,   -38505747, -1695452983,  -587394252,      -61906, -1057852795,  -360782169,\
             -53715,   -74497828,      -63009, -1009105404, -1832224855, -1901979625,    -2786751, -1667483718, -1607253788,   -26618517,\
             -363479745, -2136670117,  -881286959,    -3557217, -1203610745, -1213056269, -1513750113, -1709782585,     -962528,  -795295334,\
             -1458226867,     -573441, -1396858329,     -963413,  -710796232, -1425200451,  -448489596,     -303807,   -47190928,       -2289,\
             -1009453190,  -539343828,  -490176926,  -620814644,  -873768706,   -18530064, -1775604059,  -240820829,     -458489, -1671969606,\
             -1528130810, -1837090584, -1535875471, -1321707099, -1384031791,  -890404905, -1059156595,     -358110,   -76663707, -1882477466,\
             -1015220846, -1400921948,  -220276181,       -1600,    -9073309, -1922627397,      -27368,   -76543214,    -9083864, -1134720700,\
             -371147, -1777708461,  -763636142,  -456234857,  -206206198,      -44363,       -8392,  -403819696,  -113190541,  -111602991,\
             -418455,     -742980, -1017272481,  -367419700, -1351168643, -1678028042, -1266427755,  -689873298,  -581225911,   -86068811,\
             -792, -1939058191, -1833087298,    -2490946, -1839183643, -1907818763,   -85047226,      -25006, -1486255331,   -84282527,\
             -1042869163,      -97024,  -497942466, -1053944095,  -238916807,  -186085697,  -868004890, -1818831656,  -210573699,  -376089907,\
             -33157, -1146887099,       -2471, -1263382944,  -971236647, -1738630155, -1812532018,  -239909668, -1398954146, -2144180509,\
             -1721402447,  -904788200,  -667314062, -1927926575,  -957025720,   -82522321,  -757091333, -1635073637,       -9295,  -453686492,\
             -76022794,   -12969034, -1567704054,   -75623353,     -981550, -1697539088,  -722372436,       -9607,    -1607391,  -569323156,\
             -188603452,     -573359,   -70600150, -1976922962, -1369705295,   -91965413,       -1372,   -20369723,    -8212334, -1978733652,\
             -126953863,      -14037,      -86855, -1806355023, -2038014275, -1404350831, -1075367749,   -17611392,   -90151198,   -25092914,\
             -90024211, -1908239079, -1056237682,  -764902888,  -375444460,       -6864,  -718990065,  -626211208, -1690080281,      -63472,\
             -21481526,   -24535322,   -27562919,     -592396, -1968217777,       -3096,      -65961, -1152942748, -1861693732,  -443486052,\
             -9258537,  -312675997,  -118860095, -1485146141, -1808573294,  -410863957,       -9041, -2053187788,   -54807389,      -68018,\
             -593535078,   -40404253,  -266281170,      -98352, -1540675987,   -27426134, -1393138385,  -730164865,  -486637162,  -618408849,\
             -354684196,   -63825184,    -6628007,   -92567220, -1585525791, -2105297593,  -965191462,  -970544068,  -777536739, -1436415937,\
             -33760792, -1904566137,    -9859671,      -59679, -1760650939,   -27149453,    -3357518, -1186754588,   -94911213,   -65805246,\
             -1827039293,    -5147602,  -827023957,  -780818168,   -82050867,   -66213793, -1123928053,       -8754,       -1442,     -613180,\
             -441663987, -1894986354,   -43550262, -2076478632,     -670931, -1794065393, -1065515499,       -1376,     -177588,  -467868282,\
             -1819484532,  -300907144,  -929815883,    -1940860, -1760086763,     -180737,   -41445252, -1504093029,     -611844,  -330873709,\
             -1800, -1314797364,  -464646919,   -61011976, -1790713026, -1817418640,     -287712,       -7870, -1089757263, -1499327913,\
             -1746293998,    -4341621,  -590132645,   -98334995,  -364038603, -2135453351, -1980074896, -1970707501, -1672776453,  -692771526,\
             -1649998493,  -297444580,       -8480,   -56361292, -1182770153,  -539852092,  -660217246,  -892361245, -1391701776,  -783613356,\
             -187474, -1427951235,  -792892717,  -932703998,   -82401726,  -655258225,  -776328514, -1088734655,     -186249, -1363466187,\
             -534651470,  -675005013,      -22371,   -15026706,  -162940272, -1122855779,  -266952098,    -7381888,      -30156,  -735456527,\
             -575478260, -1550822399,     -923192,       -2320, -1075165903, -2059986976,  -706041229,    -6488999, -1385303994, -1753304194,\
             -444259168,  -418541539, -1308197056, -1964418090, -1166785126,      -14320, -1719453486,  -724853731,  -247625335,  -791573073,\
             -19886461, -1957002084, -1486482025,       -9066,  -866012133,  -773095917,        -850,    -5018654,       -1130,  -149523325,\
             -1854709583, -2023976062,  -946094646, -1049059689,    -4610500, -1487971681, -1997837994, -1864514252, -1735472931, -1300879677,\
             -1276,  -299417083, -1166088137,   -16401598, -2007391193,    -3433852, -1591894210,   -80053938,  -397831315,  -148081269,\
             -121522760,  -383554623, -1621922960, -1231377306, -1391155346,  -303449565,       -2632,      -20491,  -711394986,  -884858325,\
             -1271815273,    -4322367,  -345049141,     -894221, -1218268902,     -460878,  -105429163,  -841242423,  -349570402,       -9057,\
             -417470400, -1541316349, -1755793876,  -449463661,       -5704,  -495855759,    -8409443,  -212118780,  -240438217,  -235116950,\
             -665004517,    -5577311,      -89267, -1382951748, -1675336103,   -30486751,   -97081637,   -41803144,     -999485, -1092536592,\
             -1461754459,    -1745387, -2013444137,  -333993690, -1760112512,  -318493439, -1137149227,  -247539651,       -5035,  -891121705,\
             -16453,      -54992,  -570777887,    -1384712, -1222424462,  -459662433,     -578553,  -334061839,  -132643246,  -169090162,\
             -163225972,     -926357,  -411911060,  -181801326,   -73764399,       -1594, -1139786897,   -14749711, -2085408280, -1574724272,\
             -2108879290,  -279925351,   -11586603,      -45465,    -5790690,       -2108,    -2863824, -2052895775,  -998154123,       -4866,\
             -58802,  -870531358,   -68784161,  -793434296,  -581037054,  -664374681,  -440064290,  -399391149, -1020176903,   -70468773,\
             -1972738644, -1530457806,     -519608, -1844118784,  -350339650, -1941510879,       -7536,       -1256,      -98981,       -2830,\
             -3532,  -490574868,  -330309468,   -66922878,   -99572953,      -74882, -1928978788,  -890628907, -1757470342,       -6975,\
             -821762383,  -928800538,   -61200072, -1053937390, -1334335401,       -3901,      -25274,      -50291,        -256,  -357775808,\
             -1615,      -42879,  -792052873,  -795801261,  -492254466, -1020281315,    -5733228, -1933487485, -1699097457, -1145781401,\
             -929247,  -544220127,   -91922643, -1870002045, -1344500026,       -8745,     -730328,  -774916305,     -979557,  -332035814,\
             -2645,  -355645716, -1281623473,     -271188, -1160514735,  -183361731,  -932006248, -2135255383, -1506312954,     -955802,\
             -1413637867,  -845566909,  -405772167,  -588895602,  -358224828,  -599527398,   -53828152,      -51302,  -583614892, -1050061659,\
             -1015957846,       -9094, -1012830200,  -798085110, -1802760769, -1653498090, -1541966992,      -87167, -1089711694,      -70953,\
             -2304374,  -926087316,       -2392,      -83769,  -674776066,    -7581868,  -685106848,  -677449760, -1411177676, -1407711217,\
             -1048631701,    -1578206,    -1501704,  -149863764, -1831179267,      -45190, -1450589834,  -585392706,  -367539644,    -2884537,\
             -491845125,  -504578806,     -632962, -1483291403,  -661994052,      -21293,       -7848,  -576385926,  -452326620, -1328964965,\
             -49298459, -1326790479, -1372176702,    -9691917,  -249479191,  -188882077,  -252148973,    -3860780, -1378277680, -1899081825,\
             -1980904202,   -47902246,  -746291439, -2062956572,  -402314486, -1050442349,  -109467185,  -406587289,     -780995, -1408492022,\
             -1262751338, -1686676189,      -57980,    -9647358,    -4330867,      -24876,      -66838, -1281156205,  -921602240,       -4438,\
             -13716,      -18515, -1533672212, -1234555307,      -22617,       -7555,   -55455945, -1703223819, -1364932256,  -918023525,\
             -682052, -1880743487,       -7655,  -184743604, -1023497419,       -3986, -1339990060, -1857888916, -2039103160,       -8654,\
             -280114239,      -46233,    -1003965,      -12610,   -55069793,        -439,  -965883193, -1005419415,  -170645548,  -942740068,\
             -39370866,  -416259758,  -301698482,    -4549550, -1351590477, -1728369420,  -885330994,    -7408122,  -305906091,      -16113,\
             -1700149859,   -27959254,   -65177354,  -499404923,  -695071875,  -543927496,     -359061, -2109339622,    -4811691,  -140848666,\
             -942208997,      -72482, -1644913405, -1306046249,   -25469603,  -804183953,  -456937595, -1449016284, -2082582306,   -14283219,\
             -7319,  -233921530,  -101001881, -1420281919,     -975933, -2122889375, -1015720852,  -599127471,  -723254558,  -790112440,\
             -938654,  -871753035,   -30732797,        -498,  -699480538,   -44380917,   -23324030,    -7415066,  -870934541,  -738791805,\
             -1641411677,  -824953336, -1090630209,     -210928,     -867182, -1786321467, -1723724260,     -876138,  -398159994,       -4388,\
             -2037053996,      -24298,  -151345618, -1757120249,  -581266169, -1301862558,    -1411376, -1814324488, -1661708727,   -64642595,\
             -731831, -1588235687,  -697456316, -1799821337,   -34945310,   -77032521,       -3170,      -29044,    -9887889, -1220796051,\
             -431942,       -7864,      -64982,  -326885426,  -730600288,  -665032957, -1432506645,   -61340429,   -58077691,      -27319,\
             -1647125197,  -858688914,  -140638613,  -424141860,   -25430308,   -13766239,     -543785,     -362851,      -56101,   -25975179,\
             -36326778,      -81327,  -213516678, -1395832873, -1119873184,       -1829,     -385514,   -27828558,   -63282350,  -708119708,\
             -228408147,     -943089,  -649224999, -1622662959,     -112857,  -362396971,   -25341838,    -5085959,       -2891,     -891566,\
             -660141105,  -107900058,  -405753195,     -560905, -1724377099,  -485377520,   -45087436,  -128451043,  -892055675,   -10066433,\
             -152487,   -78825032,      -29217,  -842703864, -1137666991,  -197558966,  -522577809,       -9365,  -325242665,  -152131653,\
             -1002155638, -1068508248,    -1912195,  -212466557,      -70825, -1472475329, -1160280304, -1448824889,   -57679035,  -393438445,\
             -322535853,  -892917347,  -873911347,  -178107738,       -4902,  -526079479,    -3438689,     -276745,      -55688,  -249321672,\
             -81565,  -323064903, -1389172119,     -193148,  -921106672,  -288581964,   -79565858,       -5400,      -15444,  -547723377,\
             -47452759,  -487148988, -1765386164,       -8585,      -20986,    -5415051,   -45044531,     -917924, -1363825696, -1868762169,\
             -8836,   -99196061,  -589713074, -2047879074,  -428122160,  -490949525,  -149135315,      -99282,   -32003394, -1109676758,\
             -1763969467,     -132429,     -932876,  -962559247,  -117590881, -1613522454, -1737840357,        -926,  -778760989,      -83922,\
             -109206, -1660702839,       -3333, -1022129046, -1054861321,       -2169, -1399820189,  -982348557, -2086519902,  -769692842,\
             -1830141,  -976122912,  -604315517, -2109168900, -1813937443,  -749356058,  -310243720,  -321227692, -1719433487, -1869455119,\
             -1238331067,  -674187811,   -57342049, -1357563947, -1778202243,   -17072474,  -611003085, -1282869787,  -727326238, -1105234171,\
             -43652375,  -398675092,   -73994928,       -4191,  -742906886, -2032281100,     -206258,  -122901173, -1373444834,   -70825793,\
             -505849973, -1330598179,  -828318851,  -997124970,     -209198,  -722599537,  -891516094,   -97793681,  -610803730,      -57622,\
             -611330, -1580563636,  -322653463, -2092722223,  -542875375, -1045098772,       -9016,  -914485240,   -87185735, -2011656218,\
             -951230619,   -33093240,  -928708795, -1807482553, -1538654850,      -28911, -1344704087,       -7047,   -16551523,  -401416712,\
             -290465266,  -570201619,  -716457409, -1356106468,   -15558089, -1702449397, -1258938240,  -177911872,  -612561078,  -791537025,\
             -1077548132,   -74246121, -1337242708,      -10219,      -50607,       -1997,  -450283755,     -940292, -1067374498,  -951960657,\
             -618590, -1484702955, -1175971472,     -187483,  -927157747,  -495239774,  -899690649,  -454347608,       -8608,   -45387319,\
             -372477750, -1978816732,  -233247668,    -1612227, -2086060270, -1985634706, -1726645593,  -607706438,  -800038558,  -348352330,\
             -503943957,   -77290388,  -584518263,  -649675464,       -1369, -1751509003, -1285164578,  -625150218,   -34055070,   -92100251,\
             -1559534524,  -728508981,   -42686066,  -569028503, -1230842203,      -22446, -1824003896,  -752730284,       -9134, -2038314005,\
             -1958692385,  -898219222,    -4096660,       -2463,  -920269343,    -2968135,       -4317, -1482380831,     -110357,  -373701357,\
             -930044659,       -2634,  -624422590,     -473972, -1106500694,       -8062,    -4208383, -1072455904,     -958051,  -457436297,\
             -1852628411,  -247913464,  -567486594, -1138162190, -2111309587,  -897311434,      -12442, -1448006429,  -862416471,     -204449,\
             -1695298569,   -57250903, -1160488418,  -772938349,  -537224166, -1023670503, -2123017976, -1505571565,     -283713,       -6213,\
             -517136143,     -277843,  -158146076,  -222070629,  -417546258,  -641614937,       -2652, -1272278673, -1818190357, -1506060955,\
             -20289010,  -172348277,         -51, -1186923524,      -42172,     -609087,   -38865365,  -213795275, -1866123027,    -6259443,\
             -1795, -1406659343,       -6367,  -442827147,  -817482469,    -3016338,  -573601587,       -4911, -1397090963, -2005548276,\
             -888387275, -1120611081,  -468521107,     -485430,     -311190, -1731250859,  -594104062,      -32297,  -178910733, -2018570048,\
             -115722,    -9358270,     -507459,     -301532,     -194953, -1788195777, -1591772153,      -91287, -2049633671,    -8427634,\
             -34092, -1468353426,    -6183318, -1177812243, -2088547663, -1876551683,     -826066,       -7261,  -976792128,  -445573899,\
             -994487109, -1879553895,  -525785732,     -852886,   -55595185, -1955983365, -1027271655,  -621675785,   -29082258, -1742898875,\
             -671156166,  -119257378,       -3643,      -73694, -2017652025,        -156, -1234205596,     -748529,    -3110009, -1165935278,\
             -1864804462,  -682671018,  -112501954,       -1066, -1875153125,  -502233007, -1817270080,     -978245,  -180541006,  -410779436,\
             -27927, -1227959102,  -185351496, -2025543500,  -400075744,     -503321,  -483289495,      -91892, -2124510670,   -80110289,\
             -1420544398,   -62442754, -1594967095,    -4158884,   -93702858,    -5640713,       -1450,       -2630,     -722935,       -8944,\
             -45605134, -1178322376,       -2962,    -1674362,    -5324073,      -92732,  -770338648,    -4779144,   -37657364,    -3378813,\
             -594827,   -91907574, -1867606737, -1092913292,  -293634915,   -91089256,   -77102888,  -733584501,     -618418,       -6554,\
             -1828500002,   -15013455, -1789942913,   -29938268, -2039818293, -1608280658,  -504937190, -1824409650, -1706271364, -1030699814,\
             -2051292643, -1206789051,    -4129239,     -291308,     -165056, -1177984616, -1513128581, -1210401770,  -107771324,     -349893,\
             -1120578708,     -543657,  -257027418, -1630653329,    -9740473, -1340480432,  -572833581,  -847936867,  -329564045, -1653998656,\
             -112165569, -1091541675,      -78311,    -9222925,  -584798746, -1696727002,  -841855414,  -785221467, -1142834297,  -262640410,\
             -32850825, -1514435728,  -287190999,  -890631024,    -9068771,      -81448,  -866849633,     -822523,  -134338685,   -85698782,\
             -665614,  -994389019, -1595873744, -1351326102,     -344014,       -1019,      -14696,   -76915055,  -994742924, -2103370587,\
             -184279,     -162045,     -499040,  -682138661, -2085957136,  -317563060, -1949669544,    -5279152,  -704996670,  -372437110,\
             -722450625,   -70483546, -2090787237,    -1695785,  -539880207,       -8614,  -397963459,  -878864432, -1279714337, -1976670613,\
             -1073899391,  -101247687, -1009191605,  -765827020,   -58938508, -1332930964, -2124376266, -1474523695,     -919829, -2115978239,\
             -1554062065,  -818810343,  -855098293, -1984169055,     -113067, -1876850276,      -58049,  -568398198, -1665993605, -1887943552,\
             -4702480,      -46919, -1286010227,       -6058,      -44017,  -811057208,  -548727021,  -111604048, -1706900336, -1121297583,\
             -2083407471, -2123859515,   -50767483, -1729321944,     -168291, -1549538492, -1602540233,      -27066, -1930735771, -1061017008,\
             -2016797193,     -681575, -1229486070,  -670434061,    -1024221,       -4569,   -26733149,  -635557486,  -909751168,    -2625887,\
             -684169648,   -62052053, -1159966418,  -457027939,    -6228228,  -593436190,  -920939581,      -79000, -1943723131,      -80189,\
             -479598,    -6949105,       -9884, -1531701434, -1182381077, -2046425648,      -83052,  -564413079,  -298891996,     -378452,\
             -63581510, -2112232914, -1851061149, -1122519887,  -602139670,       -1226,  -758110074, -1930382602,     -505076,  -219239474,\
             -841651124, -2138377362,     -566050, -2127863842,     -879800,    -1953156,     -540093, -1187969744, -1066875725,  -633502755,\
             -605347990,       -4120, -1405165949,       -2896,   -64475767,  -410932895,  -683461135,  -471585340, -1366582938,  -554784018,\
             -1803546891, -2084983273,  -325205946,      -91116, -1239070096,     -590889,  -540173436, -1206810096, -1006587202,  -526427670,\
             -209241907, -1193703449, -1348540543, -1960179466,  -948084117, -1208814372,  -581302714,   -97552248,  -244064236,  -660069907,\
             -959433215, -1053300685, -1056499752, -1435280566,       -7965,       -8500,  -185974779,  -672559398,  -549552177,  -466320946,\
             -78647422, -2023817342,  -952354462, -1177525922,  -349684106, -1751061760,    -4841574,  -944080854, -1339834627,       -6933,\
             -3048069,    -8428124,       -3262,  -889002721, -1111612901,     -369285,      -98511,   -88628563,  -788310547,      -84822,\
             -680927281,  -482206954, -1102899392,    -2680683,  -976721721,   -91970800,  -101192165,  -723712029, -1509467624,  -697828087,\
             -18786,   -34237937,       -5018,  -463638657, -1960584107,  -512485167, -1731929340,  -980897049,  -798781883,  -982343708,\
             -30907, -1574522049, -2078679346,      -95055,     -751467,  -260024642,  -774061727,    -9062830,  -694501079,     -859505,\
             -275640723,  -883594262,   -46934559, -1435618948,       -3238,      -58844,  -135363045,    -4785948, -1272653072,    -6690821,\
             -254024839, -1514663248,   -48046105,  -229283943,  -635764844,  -855314245,      -90249,        -817, -1026870189, -1066617647,\
             -476422058,  -867686232,  -104228933,  -903483614,      -28608, -1068360711,     -437311,  -788917574,  -908354113,  -775914293,\
             -364756833,       -2384,  -239149731,  -602734567,   -54100406,  -908239548, -1055353359, -1274879793, -1540905446,    -2767415,\
             -1505301913, -1732095989,  -700973699, -1796557121, -1320920926, -1383602867,  -824602538,   -20945507,  -971553314,     -275952,\
             -1699604493,  -188239526, -1677004676,  -410693275,       -8306, -1541660848,  -647810422, -2042780068,       -3647, -1085377752,\
             -776441145,     -223292, -1436082071,  -800179409,     -642140,       -1112,   -28600486,  -626640403, -1741451056,   -69318563,\
             -1520989870,    -6561223,  -479253600, -1127478095,     -497505, -1544015028,  -334249247, -1650435514, -1588259419,       -3643,\
             -7241526,  -586257344, -1264910453,  -683792101,  -302531749,  -704744185, -1549558581,  -872032412, -1159701093,  -852059862,\
             -628549172,  -701336853, -1632009594,     -596937,       -2315, -1777262750,  -597438871,  -377973065,     -414409,  -817588207,\
             -813477,     -669361, -1779824599,  -169378898, -1666232756, -1170119877,   -29044247,    -1262845,  -533658451,      -51103,\
             -9715211,     -234811,  -285677139, -1162050594,  -369777087, -1611533298,  -703883423, -1599081826, -1874215493,       -3437,\
             -911351,  -102037720,  -783975101,   -83716630, -1024185653,  -524945958,     -732000,  -868084496,  -785912710,      -84980,\
             -7407,       -5000, -1084409493,       -3725, -1381335193, -2130687297,  -881841804, -1778606037,       -3050,  -185457137,\
             -1159419945,   -67627720, -2037345439, -1291789535, -1693213097,   -12945315, -1899838776,  -620887960,  -749609845,   -51424299,\
             -466510673,  -362564575,   -54971480, -2090253848, -1638996451,  -912151978,  -366699382,  -420545280,    -1484768,   -36475392,\
             -23122249,   -47851336, -1700089932,       -3730, -1178645609,  -257115685,  -975690787,  -173642508,  -525105509,    -1697198,\
             -77079222, -1418661827, -1368624779,    -8133724,  -944639415,       -6628, -1102015782,        -660,     -192536,     -425519,\
             -2060515834,    -3748802,     -373951,   -12679710,    -3542373,     -433080,   -52601126,  -782200735, -1129879085,  -951966524,\
             -344030,  -299109879,  -375747249,     -928492,     -726659, -1691408322, -1980852806, -1747223382,      -22851,    -4570837,\
             -3023485,      -70249,  -199718882, -1167351946, -2101675370, -1164867316, -1349961437, -1935489969,      -62206,  -531851309,\
             -7736536,  -848495037, -1854041680,       -9828,  -521585432,  -219917669, -2114252151,  -435048614,       -4238,  -677560007,\
             -1615580,  -289272375, -1583120612, -1928302140,    -5211692,       -3716,  -333568033,      -83630, -1360718436, -1214199596,\
             -865129849, -1292224314,  -796583721,      -97760,  -159990337,  -195536571, -1037719533,     -269777,   -42126082,       -7328,\
             -282396683, -1648906777, -1301838961,      -21351,      -40395, -1173680697, -1162303183, -1971339416,     -223735, -1486881164,\
             -537678935, -1810055590, -2071320284,  -307726341,       -1198,  -599702486, -1670627152,        -699, -1177306941, -1078661622,\
             -9415433,  -755513422, -1013154393,    -4991929,  -321007002,    -4274913, -1846777386, -1763932813, -1868625540, -1937811462,\
             -154383454,   -51746742,  -526684015,     -364985,  -126114093,     -317058,    -5222899,  -112770905,  -582618913,      -40232,\
             -59156384, -1544630129,       -9196, -2109175362, -1476392485,    -2435895,    -3032337, -1548238030, -1019413688,      -63185,\
             -32329,   -62250646,   -28985376,  -518792501, -1708486378, -1033600339, -1839098187,    -9880661,    -1876266,  -571271020,\
             -2056106255,    -3250131, -1149049626,    -3276739,  -990801687, -1789876728,  -474400869, -2023305725,       -1563, -1631911069,\
             -143301761,    -2346768, -1005652304,  -918198761,    -7507690,  -684334896,     -588629,      -35345,       -5551,     -670266,\
             -1759863757,  -763920376,  -150264841, -1169459066,    -2070005,   -84095702, -1633460129, -1385975438,      -76553, -1044109067,\
             -47538,  -506087116,       -9794,    -6745607,  -399894254,  -403051241, -1572247326, -1700107326,    -6693922,  -239287767,\
             -8671,  -300976038,    -9176462,      -85045, -1725288446,   -55667496,  -434265005, -1997854635,  -711715450,  -515687746,\
             -30855001,      -92422,   -72437762,  -644805543,      -21468, -1749049642,  -661546697,  -832571442, -1260785794, -1126434233,\
             -479074404,  -566974287, -1004942325,  -102499245, -1622668350, -1133345852, -2069669363,         -61, -1408999658, -1953410057,\
             -1952984974,       -6621,  -100487972,  -990602062,     -712059,    -8900619, -1991854917,  -505007512, -1214386092, -1316431095,\
             -7790, -1846031532,     -557054,  -371748383,    -8597592,  -333677322, -1115202454, -1154544283, -1676256881, -1422557037,\
             -856700541,  -643408456,  -814640888,       -5965,  -228787345,  -771159088,  -417849158,    -4330988, -2048450174,  -469032230,\
             -6370393, -1611715782, -1208048783,    -6354256,       -1261, -1713754604, -1537173170,   -39003632,  -274572819,  -626997411,\
             -7211,  -164295414, -1462709860,      -73957, -2143224315,  -503360754, -2135755093,  -537830032,   -20195558, -1898712754,\
             -1141327305, -1316719727, -1373655114,  -877459902, -1795328788, -1762550415,     -899785, -2111628068,    -1712160,   -45594140,\
             -1779429435,  -836461700,  -762793057,     -702160, -1781135621,   -27764641, -1766737825,      -52499,  -152096572,       -1960,\
             -52994878, -1951023352,    -5235790,  -809602352,       -1434, -1793854227,      -62009,    -4858739, -1801445124,      -44162,\
             -35294961,    -3634590,  -878022610,  -509774686,  -886513641,     -807283,     -410396, -2040689824,  -780265599,  -818609429,\
             -1464032291,      -80212,    -4304510,  -895143535,     -759485,   -27088335,  -538305836,   -26853676,       -6833, -1213717772,\
             -12205,       -1627,     -110304,       -3708,    -4600921, -1275187848,      -95886, -1991562704, -2134992930, -1476339457,\
             -775404163,     -965193, -1811048142,  -468793763,   -40824701,   -38823719, -1474465872, -1279579857, -1542897939,  -274844936,\
             -676999, -1882696022,  -674923397,     -661349,      -53180,  -368303163,     -570909, -1551333890, -1659892822,   -43866538,\
             -3073290,       -1491,      -98280, -1598833674,     -911881, -1056982808,     -872457,  -609185288,      -90252, -1622920690,\
             -630855075, -1267530293,      -12700,      -50002, -1826759764, -1421259752, -1609496134,     -850524,      -55290, -1772410016,\
             -1263821084, -1923941181,  -236538733, -1616970771,   -63195128, -1493481350,  -481696262,     -744457,  -943111831,       -4029,\
             -1487,    -3140809,   -51445244, -1295497472,      -45213,  -767811726, -1589828316,     -530129,     -165249,      -11443,\
             -1209273513,       -3337,       -8961,      -83024,  -792929289,  -167686256, -2081367189, -1506197550,  -357090956,    -7299094,\
             -1853471519, -1619116143, -1062803506,     -900640,    -2019384,   -51028887, -1991853733,  -833067110,  -470707107, -2081580320,\
             -381787812, -1565317448,  -168330007,   -63281001,        -173,  -769700197, -1839668275,  -108118203, -1988517019,    -2408688,\
             -489013919,   -83047069,  -775267101,      -94396, -2056687206, -1396896079,   -46705962,  -815227249,      -26788,       -9940,\
             -4336528,   -99858711,  -393318485,  -815395280,       -1929,  -844719408,       -1036,  -731415320,  -770456064, -1924261729,\
             -349114612, -1794088977,  -153439383,   -85277064,  -784393399, -1264366247,  -553967448, -1407743381, -1939188492,    -1849833,\
             -32190889,    -4328523,  -234145772,  -157189975, -1071945413,   -79792253, -1956620722, -1803794613,      -43181,      -95436,\
             -135861271,  -337887927, -2130082498,  -338007499,       -1076,  -282140142,   -29750234, -2005199795,     -753674,    -5051978,\
             -2075423574,   -28079227, -1766027184,  -984521425,  -216798627,    -9219003, -1556436588, -1402504186, -1377467506,  -794387594,\
             -647,  -468891768,    -4698688, -1134753292,   -35871374,  -464641370,  -934490054,   -12747046, -1077831960,  -811328823,\
             -552684, -1762612888,  -236239530,  -670891682,     -830148,  -227463806,     -493770,       -1884,    -4050832,  -110488451,\
             -1362673194,      -15180, -1492872694,  -279392501,  -721565912, -2146017891, -1097964493,  -408546089,       -1482,  -667572017,\
             -1110491079,  -978055340,      -47276,      -10523,   -84570392,   -47831588, -1010461551,  -249345462,   -37548639,   -68484845,\
             -341843025,      -63631,  -438974129, -1179945872,  -775418348, -1492629364,  -662286370,   -74728713,  -807251947,      -26431,\
             -153984, -1222914842,  -741548841,  -141896312,      -69940, -1225406168,   -69799236, -1348472775,       -3290,  -639528542,\
             -29035,  -485586088,  -270947343, -1563694661, -1685697586,   -44192720, -1927431239, -2059629177,  -856921525, -1548511026,\
             -2529986,  -307594599,  -881644014,  -204125666, -2087619282, -2041215261,  -235881217,  -954421893,  -479823952,  -973048186,\
             -1832001277,  -552938852,  -615668067,  -914869743,  -536084527,      -88817,  -321332076,  -947663603,  -972543906,  -518398825,\
             -513535641,     -402800,      -71934,   -38343698,     -770328,  -931168611, -1841351461, -1863372340,    -5873951,      -95952,\
             -270281,    -6807554, -1803879365,   -28830726,    -1736576,      -95604,  -424758217, -1088554702, -1626125172,  -301257488,\
             -1995737969, -1638120847, -1921857894, -1307336250, -1044686302,  -335083344,  -856717822, -1154607111,      -13688,   -61142380,\
             -879840602,  -944155059,   -30379609,    -2414869,      -95282, -1676415431, -1588601008, -1702902235,  -741601579,    -7413681,\
             -79308,  -794818755,       -9602,  -822550094,    -1515602,  -968413999, -1882888002,      -97174, -2139451960,      -69562,\
             -63139214,  -580171286,  -316779643,  -109686815,      -57243, -1761977800, -1275815865, -1675408714,    -8868535,      -24175,\
             -1612764206,    -9752299,  -345465103,  -332210940,      -58628,  -426670609,   -73492869,  -972247976,  -176098942,  -758279622,\
             -1115778374, -1326919049,  -348336164,     -889270,  -722162177, -1122027737,  -233595960,  -687386348, -2147006204,  -327249788,\
             -1213009617,  -447068695,   -71632703, -1895313812,  -600328449, -1930092762,  -874643414, -1595286910, -1312489004,      -15503,\
             -337699067, -1364193982,      -66881, -1255881467, -1092957978, -1789348315,       -5425,    -1146183, -1114655629,       -1170,\
             -235238305, -2122505734, -2033097174,  -255144430,    -8866030, -1997498439,       -2895, -1302869639,  -977281183,  -460444929,\
             -463,   -56323021, -1997332997, -1201534195,      -20178,   -74379372,  -440783288,    -1353477,      -98875, -1856183848,\
             -46130,  -821590180,      -49694,   -72737443,  -838288428, -1392763354,      -19118,    -8819843,     -361473,      -98359,\
             -1610149312, -1028510575,     -390584, -1899785621,  -238418303, -1218877229, -1863626686,  -248788967, -1224231750, -2085521935,\
             -2005223801,    -9609841,   -15171299, -1870461784,  -610947574,  -927465249, -1086344172,  -456874447,      -69960,  -969996305,\
             -4911, -1287152915,  -184718287,  -770506621,  -336953982,  -290915044,       -3522, -1892155595,    -5628531, -1877676597,\
             -26938054,   -23631285, -1769698512,  -662545248,    -2136991, -2035237117,  -272711086,     -933785,       -4190,   -27253823,\
             -2083312389,  -900757371,  -132220431,   -40428294, -1155819736, -1971604660,       -2334,  -417275762, -1865975620, -1271995527,\
             -473052669, -1223246291,  -774352003, -1226317752, -2084211464,  -268181906, -1686515732, -1578771334, -1536920155,      -50491,\
             -262390030, -1697473048, -1030860556,  -900772540, -1395006412,  -310755266,   -26092512, -1203099487,  -939056146,       -1265,\
             -119756487,   -93630711, -1400683330, -2061490520,  -616921400,  -626546763,     -650475, -1503880492, -1793727437,        -857,\
             -2062899545, -1597690033,   -77642197, -1870799977,  -560710599,     -655376,     -814555,   -95637175,       -2475,   -95018498,\
             -1840993088,     -685987,  -878057178,   -95045220,    -3659979,  -848707869, -1543150397,      -49016,  -950210206,      -27385,\
             -234839252,    -3915362, -2141764406,     -399781,   -74908745,  -549876816,      -48767,  -892304634, -1676275613, -1251576257,\
             -58657466,  -857157677, -1136465097,    -1947851,  -690867491,  -150368656,       -5845,  -348814612, -1251821289,  -385125818,\
             -64500,  -428895858,  -190763988,     -660741,  -824245850,  -397828142, -1198290948,     -759493,  -982164422, -1987865342,\
             -987873814, -1283325530,  -205340189, -2107942432,  -691893157, -1298238728,       -6611,     -119505, -1269083550,  -383188372,\
             -8696804,    -4816958, -1341634140,       -5302,     -716031,      -84495, -2074078671,  -503751860,    -9627494,   -37363581,\
             -742252283,       -2973, -1335496579,       -9007,   -52027136, -1878646378,  -743863984, -1107015559,  -992840323,      -99992,\
             -984667567,      -40339, -1082117261,  -490937658,    -4409888,  -354652910, -1302080317,    -4069294, -1957654399, -1360347911,\
             -1228422336,  -496757761,  -173919830, -2023945554, -1781241684,  -997288358, -1468921756,  -968171692, -1567239732,  -140346184,\
             -1914953416,    -4792170, -1639376900,       -7066,       -7040,       -9271, -1854477019,    -8459639,     -759435,  -433446691,\
             -167478514,  -851616200, -1820945781,       -3112, -1271513735,     -547360, -1529167649,    -7305414,      -28113, -1059898972,\
             -795920248,   -61968897, -1178721235, -1210092204, -1220305958, -2128994460,  -634565682, -1038581640,      -65189,     -472478,\
             -15727146,   -26177825,   -15850679,   -79381658,  -862641674,   -56816807, -2130948406, -1418362410,  -735413176, -1956744396,\
             -5126620,  -394377098,      -11069,  -240476184,      -78757,   -93005885,      -18932, -1704721327,  -524550436,  -534861457,\
             -1274505949, -1125003563,  -352316549,  -628039965,      -17721,   -31840763, -1972579828, -1668017027,      -78592,     -241755,\
             -1240993426, -1825250860,      -39173,     -592254,   -40309523,  -557180100, -2103639502,       -3379,  -579027024,   -91010632,\
             -2136034968,  -408497801,  -576787835,  -215030947, -2137903073,  -620900579, -1316959829,   -38531138,     -754657,   -71757638,\
             -1886052,    -3987182,  -723859182,        -349,   -61280342,  -113415911,  -488742164,  -348603918,       -8678, -1930880717,\
             -1293757139,       -7296,  -604174488,  -632066770,       -4348,    -6167372,  -147303196,   -38379676,      -43936,  -291451157,\
             -221853688,  -816464732,       -4919,  -834385118, -1942073165, -1257558291,       -6710,  -603459899,  -659170871,    -7689918,\
             -533568783,    -4664495, -1581627610,  -783636308,  -972052418,   -42735496, -2108036986,  -151559935,     -813253, -1672307234,\
             -1639420181,  -387902256, -1017752520,       -3948,     -622433,       -4049,  -902029217,  -757609249,    -2936431, -1120542192,\
             -743247, -1822492781,       -3755,      -38204,  -717986381,   -49375508,  -460956797,  -857906488,      -73679,  -986246674,\
             -227694951, -1465201579,     -353777, -2023521987,  -495123946,       -5798,      -35193,     -792639,  -401374711,     -399276,\
             -1656754241, -1467055821, -1480624270,  -218167952, -1062302301,  -495160115,  -197737977, -1471709865,  -282738842,  -823853128,\
             -152423,     -147600,     -276143,       -4209,  -124009286,       -4351,     -858516,       -3038,  -180232541,       -9876,\
             -9076,  -475004700,      -90791,      -59278,  -144438437,  -716213271, -1949839426, -1665262609,      -49622, -2065031093,\
             -722388222,  -991983402,      -15890,  -360449578,    -5933805,    -2160049, -1133659055,   -50133869,   -62205605, -1952924632,\
             -561803, -1806238594,       -9306, -2042890091,   -32379136,      -70394,     -680549,       -3496, -1022885616,   -60432085,\
             -804127994, -1150033594, -1151580021,  -433536852,  -104767853, -1954840091,       -1085, -2045295771, -1087125027,       -6342,\
             -333883468,  -166282046,  -198818573,    -1547660,   -31342890, -1583183851,   -15500439,   -95476458, -2076779772,  -676858765,\
             -62326186,    -6946769,  -380422803,   -30312639,  -530030072,  -441826514,       -3634, -1229042671,  -360268021, -1524024508,\
             -93840,   -20527291,    -8319476,  -817280326,  -241787828,    -2673259, -2141529187,      -50031,  -980980926,  -964054676,\
             -68463, -1350695925,  -594293107,     -379017,  -818211299,    -8286707,    -7108128,  -507068896,    -6475352,  -634280247,\
             -2021714342, -1193723759,   -43441977,  -470322860,  -485156085,  -253886183,  -865810503,    -5058749,   -34668899,   -44653851,\
             -18537, -1417443361,   -40534054,  -170897774,  -751812085,     -343554,  -612194667,    -5794504,  -897283942, -2108763330,\
             -2141033006,       -4871, -1182507897,     -322707,  -413592348, -2042795046,       -9530,   -63034308, -2089499793, -1894348132,\
             -978417777, -1624306141,       -1882,    -1095761,    -2385517,  -327209261,  -593512541, -1562941555,  -904755996,  -138805233,\
             -1914691466,     -139828, -1485758912, -2004202133, -1009880560,    -6202658,  -272542636,  -948048662, -1156152361,  -405422263,\
             -1173472331, -1862703066, -1671183505,  -293577941, -1183301499,  -101619882,   -33415656,  -495651336, -1063400245,  -169686980,\
             -292643, -1063853649,       -8851,   -14735297, -1643347203,   -36472255,   -68963842,  -200486212,     -231391, -2060533851,\
             -1460037067, -1821523487, -1649684271,    -3489262,   -79748517,      -32616, -1960218695,  -401108219,  -135677605,    -4256287,\
             -2094095250,       -6158,       -3753,    -9525344,       -1792,     -862909,  -960742911, -1212064230,  -378181357,       -4431,\
             -90076189,  -678121621,  -440808122,       -4175,  -358891145,  -641474747,  -549025429,  -505203399,  -357306482,        -692,\
             -1978346832,  -760232430,  -645440434, -1575284579, -1795853520, -1212434121, -1197511756,     -356775,  -656384612,   -34921390,\
             -540535093,   -24709249,       -9846, -1944975402, -2147259353,  -554548589,      -68585, -1384041569,  -196014239,     -225798,\
             -77052,  -932871714,  -113285060,       -5391,  -712296579,  -842873735,  -667220498,     -974359, -1517769510, -1337233570,\
             -565446,   -81564378,    -3225949,    -4473051,   -47727644, -2048716725, -1269324437,      -40540, -1946997925,    -5432280,\
             -77208,  -322881024,  -338554429,     -638792,  -367750931,         -71,  -269813247,      -52431, -1555218593, -1039050214,\
             -8667849,      -44246, -2133990805,  -651332166,      -57466,  -257180228, -1828396892,    -5542210,  -860938946, -1810484865,\
             -644229361,    -8167770, -1109895061, -1505514827,  -823088457,    -2075605,  -834622962,    -9953644, -1759862246,  -451671718,\
             -2080676225,      -21480,       -5736,       -1403,    -1007229, -1663306505,  -675031406, -1496579016,  -736752791, -1414683843,\
             -710380356,  -106302029,  -483422663,   -99496117,  -942475128, -1563217168,  -351519169,   -91127687,      -88401, -1960532339,\
             -3688535,   -53852059,      -42323,       -8109,  -278419874, -1641538926,    -7937004, -1079404031, -1859078060,      -24979,\
             -974135, -2059563543,  -180786389,      -31046,  -990137055,   -85793757,  -989983754,     -520182,  -958761254,    -2704778,\
             -1495632288,       -3351, -1855406220,      -30729,  -703805630, -2122580301,      -19150,    -4836675,  -716466753,  -440616103,\
             -5665627,  -500925028,   -95805781,     -273973, -1442167666,      -19616,         -91, -1810197956, -1145585836,   -32804319,\
             -455706291,  -571913628,       -8522, -2029608456,   -31475371,  -476986819,  -554542767,        -626, -2059526901,  -977662599,\
             -548048172,      -68041, -1608194057,     -730756,    -6594720,      -29350,    -2837749,  -888131069,      -79709, -1787373099,\
             -9312043, -1508241306,  -456324987, -1222489898,  -582571123,     -164196,      -52744,  -965759277,    -5079817,    -2209228,\
             -9890427,  -497143807,  -816817092,   -33979390,     -311989,  -601498202,     -972243,  -601960091, -1157688692, -1076132614,\
             -878582470,  -978357641,   -87418220,       -3497,   -44741583,  -125340591,  -985517290,    -5409132, -2134877321,  -205092893,\
             -1176972162, -1299718247, -1945011829,  -985622007, -1348111176,  -110451692,     -857551,    -2974638,  -521502343,   -46996795,\
             -1152105493, -1998313859, -1972104855,     -455675,  -535688541,      -92303,  -192442398,  -333653121,       -7113, -1334293641,\
             -1288964514,  -222834294,  -577942761, -2099428475,     -438485, -1271385352,  -916502464,    -2660442,     -184678,    -5697335,\
             -395669770,  -743751565,       -6159,  -384639757,     -950955, -1426100147,  -811260815,  -148463701,  -155856853,  -234022296,\
             -41626796,  -190174595,  -940887361,       -8636,  -621758807,      -62522,  -906300453,     -920909, -2099015227,   -86414701,\
             -1223288836,    -4133371,       -3795, -1620341093, -2016405195, -1108388655,     -518236,  -836354206,  -677856439,    -7079365,\
             -327968091,       -4181,  -111484290,     -105800, -1900923668,     -533133,      -13213, -2057263808,    -9121073,  -651706504,\
             -592989166, -1673606157,      -28154,       -3045, -1746325901,       -6124,  -649706165,    -1102624,    -6670506,  -621961433,\
             -1241791806,  -947289121, -1320250807, -1740767981,       -7348,  -944388488,       -1194, -1146523869,    -1577964, -1254544266,\
             -1126796182,  -240054645,     -531446,       -5487, -1043382677,  -394022644, -1686937095, -1747777938, -1475837659, -1726403531,\
             -673598193,     -610872, -1845596423, -1183592749,   -56175743, -1239427653,       -7602, -1366537400, -1104934845,       -9031,\
             -9444,  -218213401,  -327689017,      -77625,      -89135, -1830292123, -1840505208,   -60078182,  -940233465, -1246779776,\
             -811596807,  -697870372, -1538133360, -1673627625, -1465907024,  -767113823, -1628312780, -1912726749, -1053495645,  -648220455,\
             -1069460739,    -5895564,  -337335771,      -23246, -1394948141,      -53625,     -856281,  -741867376,  -150256127,  -236848540,\
             -1524935550,    -5151041, -2137335420,    -6570502,       -5826,  -702734673,    -5462953,  -947039828, -2108209434, -2103237828,\
             -8738, -1323815489, -1123203323,  -898806475,  -845369879,  -605031321,  -844623753,       -6106,   -26290808,  -299834374,\
             -1788934984, -1547048233,      -71832, -1674066214,  -330584275,  -961666987, -1720119897,      -79004,    -7625841,      -85878,\
             -1116361719,  -716844748,      -36584,  -428647149,      -50083,      -10077,     -924149, -1936682063,  -873525750, -1535167555,\
             -599934782,   -82887704,      -12885, -1889518497,  -607255920, -2114480646, -1844731553,    -9980058, -1686158693,  -483762600,\
             -533913794,  -867672019, -1324670085, -1728513883,    -6245776,  -945507901, -1715478119,  -410204632,  -430765591, -1044469756,\
             -4206053,  -395533622,     -115752,  -485376216, -1025631241, -1107660146,  -173079906, -1853542302,  -426313671, -1345028980,\
             -954833,  -643776743,  -773389996,      -39641,  -227363283,  -633773998, -1916298764, -1483598814,   -94123664,   -21529337,\
             -910371357, -1618761823,  -498492245, -1327316476,  -291437021,  -183508510,  -702540818, -1540687920, -1421844740,  -690017057,\
             -8645,   -87799914,   -64238451,   -30504613,    -6895756, -1435781444,   -68595287,       -4490,  -949320608,     -275798,\
             -1864337290, -1096510634, -1206391252,  -733241274,   -48262548, -1915088364,  -933222463,     -611372,      -34990,  -197481185,\
             -720728801,  -461847358,   -90621526,  -960804671,       -6517, -1213695848,      -66443,       -4553,       -2505,     -361938,\
             -1689067900,  -535048253, -1814666325,   -73689724,  -175578247,  -723509059,  -387822703,   -61399162,  -342528424,  -281429271,\
             -541,  -845291546,    -7337843,  -471960459, -1841719522,   -80630420, -2113670449,    -3607714,  -743052978,  -172718452,\
             -2033998225,  -687797759,   -56494114,      -61016,   -89829120,      -23493,      -77379,  -548830797,    -6888965,  -667857185,\
             -99656166,   -32647857,    -2244158, -1836627505,  -271878626, -2136748417, -1855360225, -1437569088,      -48366,      -89625,\
             -219967,     -267631,   -29920778,       -7573,       -7032, -1436890049,      -56344,  -712131162, -1464445203, -2016083886,\
             -2777,  -164887216,  -813462872,  -335011561, -1090360492, -1721584132, -1551714447,  -172015401,  -988048707, -1377750323,\
             -727803702, -1710071768,    -2311828, -1743587968,        -863,       -9481,     -443938,      -16607,  -282681686,  -588323487,\
             -284962516,  -826669818,  -257940680,       -9555,  -817705175,  -387106474,      -27531, -2014856754,  -167444805,  -437423234,\
             -116107734, -1831234013,       -4481, -1009453359,  -551009188, -1194241335,  -341061313, -1180458531,   -58667994,    -8685935,\
             -495205859,       -5044,  -236688100,    -9683501,      -64218,  -484968679,      -43583,   -12311508, -1699256277,      -62823,\
             -8526338,     -107386,  -828882207, -1810483111,     -220194,  -243317459, -1210782218,       -8604,  -973890074, -1504716891,\
             -517959797, -1427781617, -1042566864, -1026207184,     -574559, -1414674708,  -504321706, -1396337916,    -8169668, -1897767617,\
             -316513399, -1284112209,     -233285,     -467455, -1494079669,       -9074, -1312953330,  -880230720, -1833159086,      -48378,\
             -3868112, -1342700333, -1859569431,   -82611561,       -4503, -1642102729, -2138191849,   -74739083,    -7624054,      -84079,\
             -3996, -1046044534,   -51485717,  -391098006, -1436357683, -1944158211,       -3849,      -46485,  -512068732,       -2885,\
             -769541212,  -522464538,  -843020206, -1259312911,   -61454675, -1878379793,  -474297204,       -5615, -1112742996,  -975048966,\
             -1287689467,  -246195312,   -32759055, -1852410708,  -594676433,   -27421059, -2044326300, -1644591642,  -914564339,   -74575980,\
             -438712429,      -10869, -1610561751,  -438508520,       -6827,  -320657732,  -470222830,       -6958,     -396666, -1513820585,\
             -381114590, -1937210767,     -808666, -2134714202,    -8802479,     -630764,  -249901676,   -24869200, -1579549608,  -573598986,\
             -121604606,     -548218,      -22932, -1639991366,  -725646241, -2023372732,  -891537346,     -757805, -1960844394,  -979331758,\
             -1798910990,     -973870, -2030376824,  -185286545,  -446636833,      -68105,     -398707,   -12445078,  -990939739,  -523877721,\
             -554023706,  -535532471,  -796865743,  -268534031, -2014929492,  -151338531,      -98113,    -6141977,  -720323406,    -4063613,\
             -35859,   -85929501,  -462358760, -1099986790,  -135439259,     -804670,       -4815,     -680182,  -129574805, -1016175992,\
             -1114625918,      -45510,       -7581,   -27083638,  -497414828,       -4507,  -500961195,       -6422,  -853337343,       -5494,\
             -907636090,  -310301341,    -4241218,  -265311590,      -62127,  -148994280,      -93674,  -395044490,       -6334,      -68749,\
             -744401220, -1255186368,       -1790,     -762745, -1487375991,  -406527227, -1549680335,    -2002927,     -745460, -1502368202,\
             -1350093279, -1305374491,       -6664,  -256082005, -2049531270,   -28477726,     -435465,       -8838, -1933932367,     -942219,\
             -2083952324,  -632495914,   -58490426,      -72860,   -44701464,    -4074254,      -22077,  -337213666,  -100104976, -1581218718,\
             -1255832808,  -322862036,     -707226,  -740439118, -1689786888,  -167314603,  -359530153,    -2382010,       -4134,  -984274488,\
             -86224, -1507939909,       -8887,   -25180892,       -3961,  -205461974,      -53937, -1217740710,  -701693293,  -487271714,\
             -7240,  -132990147, -1587091256,  -810341573,  -693891192,   -70614318,     -759869,      -17168,     -257346,  -384324429,\
             -1794513635,       -4329,        -880,    -1242757,    -4063918, -1132343414,     -160527,      -29241,  -521684410,  -713832984,\
             -102566341,  -466849484,      -63007,  -124709986,  -606471886,     -423136, -1507771268,   -36080840,       -8807,   -25350741,\
             -136811615, -1681238018,  -944656281,  -514967839,  -806255677, -1155676797, -1989215390,  -492665549, -1350700203,       -9805,\
             -10447,    -7556174, -1784311423, -1179680245,  -294175505, -1404943882, -1419661928, -1426937007,  -897023174,  -475151648,\
             -145119846,       -7149, -2076880019,  -676557045,   -19998180, -1534204892,  -608311436,     -633462, -1355656368,     -506236,\
             -1403465063,  -678379301, -2133620010, -1266245999, -1818143339,  -575200370, -1300334579, -2109985207,       -5053,  -426470523,\
             -1186024060, -1089005276,  -902128129, -1599236133, -1602295827,       -8312,     -521295,  -809203738, -1873000212, -2123022959,\
             -1483741930,  -921085235,  -608931295,     -696207,  -819491008,  -933107491,     -602017, -1490265377,     -262169,  -987581329,\
             -1094215764, -1706128440,  -584726285,     -414245,   -15030445, -1514404254,     -589574,  -269314607, -1808031848, -2038588387,\
             -31158,       -5242,   -32547832, -1364066397, -1330865146,    -3679969,  -154442683, -2035145838,  -762235664, -1407373437,\
             -1734279185,  -283884837,  -571958134,  -913413464, -1259020153,  -460352102,  -870958698, -1383325378,   -29034564,  -727166846,\
             -1178125846,  -831869010,     -776641,  -864006435,   -10319101,   -19149778,     -738212, -1666126621,  -694697166,  -546729842,\
             -1971102075,      -49205, -1763596830,      -42502, -1902055204,  -934295189,  -818224775,  -421837402,       -7134,      -56571,\
             -66264063,       -5708,    -8459681, -1824564045,  -810912764,     -452833,  -815654621, -1862803581,      -51197, -1599262431,\
             -659169441, -1091365879,    -6674058,  -465766777,        -512, -1097218290,       -3146,  -374926999, -1194069919, -2081043557,\
             -5260,   -66620728, -1229728947,  -257077465,     -253393,  -177314900,    -1969688,  -742167393,     -596322,  -475581659,\
             -630888177,       -9537,       -3612,  -468164302,      -54902,      -71187, -1621488921,       -2625,  -500472790, -1792125019,\
             -218371504,     -250264,     -128833,  -606480032,   -53601631, -1304573776,  -308110997,   -17820810, -1857726680, -1215015003,\
             -857191,  -153102744,    -4215429, -1859883595,    -5424700,   -42755010, -2044567700,  -645109168,  -129965184,    -7819505,\
             -813248531,     -173154,  -545632271,    -6352648,     -474726, -2054233253,    -5263056, -1781869977,     -139391,  -614505022,\
             -194,  -347081955,     -608259,       -3053, -2051912795,  -454504959,     -588562,    -9549512,      -49880,  -977975312,\
             -1718059163, -1082457626,  -407567941, -1200413698, -1398808740, -1472500786,  -275985121,     -524243, -1191857254, -1907368845,\
             -1617956718,  -184799038, -1848476252,       -7563,  -744077362,  -907317910,  -268079573,       -5890,       -1712, -1894239000,\
             -583016,    -7449908,       -2011,  -836608370, -1000045616,  -190163333,  -764382190, -1007673034,  -348632132,  -973816023,\
             -2034007062,    -5013079,    -1668603, -1580030761,  -864580728,    -3430430, -1486681070, -1588555982,  -838598287,  -136625922,\
             -708100016,     -561800,     -420417,  -508919303, -2032626237,  -692795487,     -442253, -1197679001,  -767284727, -1605865727,\
             -921115872,   -64399092, -1005670565,  -260032163,      -20837, -1363702504,     -443583, -1180002347, -1115874924, -1492328596,\
             -1590490034,  -435157461, -1154245583, -1314782796,  -300377117, -1082622116,     -382639,  -439338151,  -292967136,      -85027,\
             -1422282347,  -257818392,   -90117119, -1006829739,    -8558407, -1642625378,    -3133891, -2014063772,  -982948836,  -400547384,\
             -1760727891,  -711311759, -1246776970,      -99538,   -59292249,  -778917705, -1924651341,  -324155346, -1568587197,      -36721,\
             -229217954,  -300572485,  -630471385, -1048628383,      -17430, -1101432922,  -918595174,       -5868,  -135645868, -1451549275,\
             -3675497, -1420411625, -2075235033,        -269,   -54997709,  -370192391,   -48295794,        -281, -1289640642,    -5595314,\
             -636496991,    -3129931, -1469612426,  -359992166,  -166322124,       -9701,    -2514773,       -8238,   -36090254,  -733044955,\
             -1615425453,       -7951,  -102051580,  -450704088, -1069348841,       -4451,    -2415004, -1026609426, -1593456226,   -64316733,\
             -846910705,  -980904953,    -8311267, -1165526287,  -241019886,  -342591437,  -432322362, -1245601700,    -5089653,    -4463991,\
             -154477193, -2036872425,  -263617234, -2037095598,   -60705807,   -44792720,    -1524044,      -82026,      -50141,   -10772439,\
             -60083,    -1210050,       -2213,   -21678070,     -517439, -2008173531, -1177583413, -1074032642,        -706,    -6675971,\
             -1652224578,      -92108,  -689665559,  -891485899, -2009439777,    -5106353,   -81280236,       -4399,   -53863165, -2093643072,\
             -560348554,     -997786,  -766300774, -1796424969, -1521428658,  -183042521, -1520133303,  -523852280,       -9401,  -395534000,\
             -1243724691,      -21213,   -33410921,  -701168749,       -4056,      -71008,     -501043,       -8381,       -2561,  -179172608,\
             -34877, -1107757031, -1605350128,   -82489021,  -762646797,      -58108,    -6355566,      -90994, -1353201581,    -8693061,\
             -1638692717,      -40812, -1676402079,   -80909621,  -200935319,  -951149456, -1000481148,  -265462675,       -4766,     -511598,\
             -7415,  -146378944,  -263435204, -1394749686,  -985539234, -1258456333,   -54143560,   -30803427, -1702341816, -1910048119,\
             -930009,    -8026822, -1871279895,  -909127145,  -658032208,    -5456113, -2045709601, -1445763657,       -4499,  -220377360,\
             -1107474244,       -6731,   -87391408, -1222893348,    -2098889, -1009599881,       -2610, -1871161610,    -6668546, -1387023731,\
             -99713,    -2570105,      -35971, -1268679297,  -321315642, -1882633399,  -966030917,    -2430081, -1118952278,    -3390603,\
             -374986014, -2028310584,  -392592260,    -3259459,      -83222,  -760146696,  -758669323,    -9225846, -1754624070, -2104533266,\
             -7783970,      -92646,       -8802,  -492337418,    -5804974,  -864054649, -1663548279,      -48298,      -81005, -1741879313,\
             -3917, -1506415660, -1006172069,       -5505,   -50727187,  -101377395,   -60392261, -1159038212,   -10573651,  -568261202,\
             -7140, -1762069029,   -75885433,      -96600,   -91409034,      -19081, -1827611348,  -413693070, -1570573646, -1061716799,\
             -1882073767,  -358002071,      -84929,      -83990,    -8081610, -1707473941,  -917696838, -1005655502,    -6076469, -1015904603,\
             -4121883, -1409883099, -1971365977,    -2443289, -1862350159,  -568188365, -1966563226,    -4715089,  -220892115,  -295759968,\
             -646475, -1404755820,  -333394586, -1890312879,   -96323090, -1870265126,     -809068,    -4998253,      -39037,    -2031443,\
             -8201,       -3618,  -123963053,     -586642,  -570193818,    -5768647, -1829567322, -1776282170, -1760740642,  -555559758,\
             -11073884,  -902376174,    -6519526,  -644824169, -1364945479,       -5492, -1440024534, -1265926603,      -70775, -1055267007,\
             -529377978, -1398666273,   -27916113,  -760373979,  -430756171, -1265158269,  -729898721,        -151, -1623475958, -1491212569,\
             -1465877076, -1259028494,  -953460911, -1584480044, -1357708830, -1073952918,  -943338901,     -618693,  -577441951,    -9739461,\
             -93890, -1538821749,  -318646584,  -554508908,  -438359510,     -886245,  -261690786, -1673079781,     -532479,  -259257347,\
             -1580560806, -1446766452,     -704867,  -621700597,  -908271546,  -361811708,  -724029154,      -47837, -1724867862,  -141287149,\
             -1497725620,  -970882814,  -451265451, -1426581167, -1743742408,  -279200399,   -80363242,      -44110, -1907364014,  -792096407,\
             -719091963,  -868594167,  -126066048,       -8502, -1538986184, -1553129761,       -6191,     -179925,    -5021641,  -604431059,\
             -5173,       -2394,     -193560,   -20298944,   -76028819,        -788,       -4992, -1980315497,  -243590685,  -507766434,\
             -62548,  -315219221,     -395435,       -8442,  -285857271,   -11318440,       -1410, -1947489963,      -51608,  -255972727,\
             -23058,  -714804905,     -469609,    -7649348,  -325054125,    -2788908,   -96028942, -1212106283, -1192943458,  -644061601,\
             -1552285122,    -7071880,      -15102, -1675507766,   -35643833,       -8357, -1774211642,  -326997045, -1944420889,  -745968089,\
             -61772155,       -9925,   -77191803,     -403233,  -587955008, -1484272554,  -998746913,  -107284912,  -208956956, -1710778092,\
             -381258250,  -817406731,    -6725402,     -916209,       -6845, -1319483463, -1282556458, -1857715000,  -990151445,  -673327242,\
             -1993508752,  -947982292,    -6751821, -1125267485,       -9694, -1868891544,   -13846814, -1143128879,        -210,      -63925,\
             -2103,  -397805597, -1830171704,  -160837854,    -6820640,       -4963,  -834643423,      -23860, -1853691917, -1163041490,\
             -3264,  -754314705, -1429017370,   -68128233, -2120356844, -1547301707, -1002655892,      -21626,  -674798031,     -783539,\
             -3952, -1263959250, -1159671829, -1552074866,      -58430, -2021962651,    -3632815,       -3448,    -1256570,  -621584325,\
             -774115524,  -376369162,  -573657347,      -91203,        -984, -1130120612,  -764025747,  -186218363, -1775127616, -1568971287,\
             -98427508,    -7362042,       -8671,    -1199933, -1667163129,       -8738,    -4938631, -1299888048,  -189280717,  -102866178,\
             -2791, -1097935608, -1616174309,    -3965858,  -592150631, -1018664933,   -51209473, -2068857467,  -867795446,   -91077087,\
             -7020,   -15095563,  -576979774,  -709558251,  -467772948, -1109710238, -1430896582,  -544641864, -1306429568,    -7946681,\
             -140143026,  -811888180, -1986812381, -1675497522, -1121153687,  -527670897,      -71976,  -540232266, -1617784909,     -408625,\
             -9055,       -9749,       -7753,  -721987553, -2021368315,   -46673222, -1604994010,  -207041053,      -83393,  -851665401,\
             -2017366289, -1856007221,    -7414224, -1266549567, -2095406478,     -267176, -1980768214,  -772758799,     -517817,      -48358,\
             -1049453983, -1825414726,  -214017942, -1783417245, -1787938681, -2132932435, -1930224554,   -20671742,  -412238047,  -520889543,\
             -794897,  -218557328, -1715210381,     -960107, -1597864605,    -8403163,  -422703266,  -259635237, -1110633999, -1926281897,\
             -4748765,  -300202157,    -1928049,       -7415, -2093189624,      -29501,       -3650,   -80371878,  -109671714, -1565820414,\
             -522837937,  -537385344,     -698203,     -930523,    -1817586,      -55198,  -817698762,  -873506176,      -79968,  -684677883,\
             -130072325,   -78015363,  -467437718,       -7195,      -22476,       -1716,  -438800226,  -720510403,    -1497775,    -4778337,\
             -481729906, -2062553138,    -4027065, -1364717776, -2011266380,    -2408573, -1635197568,       -8880,  -440370894,   -15698730,\
             -1518861890,    -8074793,   -25460722,  -806411207,    -7431976, -1278664791,   -60234527,  -168554768,    -9792936,  -722363263,\
             -703307582,  -319857554,      -15245, -1068488212,     -983849, -1157791749,     -200344, -2015782466, -1517991716,  -939517061,\
             -1634582079, -1434678805,      -77474,      -70237,  -322709589,   -82670957,  -130609386, -1818549442,  -458814407,     -579566,\
             -797090355,  -148194096, -1414103148, -1161661310,  -144893479,  -719400443,   -39331534,   -81834504,  -969855492, -1843414501,\
             -851664061,  -913424568, -1745719660,   -80237730, -1157625686,       -5534,  -165028871,   -47485770,  -935526477,     -913053,\
             -3231,     -652214,     -454645,       -5506,  -137805705,   -62256289,  -278321072,  -314208869,  -962039547,     -314227,\
             -1816009248,  -637983458,      -53998,   -78148215, -2012002212,      -93704,      -19049, -1273786004,     -563647,  -241784411,\
             -15760046, -1478132313, -1702259127,  -730508954,  -912359095,   -62726041, -1180958584,     -472954,      -13088,  -961644762,\
             -8134,       -2252,     -984896,    -1973137,    -4742999,   -63030428, -2086770909, -1333298822,  -597805132, -1100755154,\
             -55415,  -169466633,     -441326,  -335661140, -2135505597, -1251470882,  -775279669,  -379396538,  -734217172,  -446095505,\
             -1675108586, -1331494306,      -12746, -1484378007, -2093666065,  -210866841,      -69106,   -73430334,      -24013,     -771637,\
             -58492270,      -16808, -1528959493, -2109069860, -1182632579,  -635287161, -2023024876,       -3967, -2147268790,  -513259617,\
             -291126745,  -344339601,    -3794624,  -389583543, -1619822574,  -257579221,  -781250946,      -21257, -1933452650, -1971577510,\
             -237468374, -1617560335,       -2142,      -80416,  -487322725,    -1410969,   -98448882,  -214529505,      -12743,     -738583,\
             -93608843,      -20641,  -108666078, -1001515977,   -89113973,  -270510461,     -458486,  -144991288,  -553210515,     -580743,\
             -61483,   -19460047, -1312731709,  -292001626, -1306585215,  -868048403,   -61431292, -2017262712, -2018996270,  -636664642,\
             -739726262, -1679696594, -1623365803, -1112914233,  -535290745,  -900420038,     -848762, -1179415074, -1002635483,  -476809701,\
             -7692,   -79390358, -1510488839, -2034420179,   -50595763,  -761748179, -1401479103,  -709783720, -1376694964, -1266466391,\
             -2121185,   -88851413,     -725174, -1536873517, -1029979720,   -59602008, -1524041897, -1165408560, -1447522090,  -202022469,\
             -494101211, -1959033553,    -5925584,  -419231745, -1092877498,  -977552516,  -341410705,      -34494,      -57422,     -616638,\
             -1868294538,   -28570218,  -850281853,     -809626, -1268142576, -1080449697,     -533645,   -55058897,      -65869,      -68800,\
             -742737264, -2098190906,  -639218560,  -333098940,  -781930105, -1194973580,  -820774091,     -743016, -1912853872,  -131548344,\
             -1384702676,       -1821,  -916417239,     -480803,  -359187894,  -741217677,    -3609427,      -99643, -2057116356, -2131380963,\
             -472349378,  -208538760,  -420857352,  -148209602,  -811766407, -1999380671, -1710574802, -1816842577,  -558783340,     -596376,\
             -2036759674,     -286184,       -6120,  -799044574,    -8543641,  -178247912,       -1159,   -76797746, -1749119207,     -217370,\
             -60097457, -1448507625, -1907091506, -1484843176,  -574234169, -1894476680,       -2979,      -33144,      -21257, -1320940208,\
             -1303989204,  -546138332,      -24453,   -26792965,  -527908704,     -386678,  -639631214,  -634129110,   -92740557,  -845107602,\
             -155955340,   -59239286,  -280246532,  -783806254, -1435799709,  -772211410,    -7812500,      -81371, -1718931890,  -980236092,\
             -76284624,      -73549, -2077759433,  -324966441, -1932167939, -2143082911,  -242782470, -1352658865,     -804806,  -997396284,\
             -642618253, -2117888591, -2011876942,       -3458, -1867878015, -1319697086, -1825030838,   -17767430,   -61093488,       -7787,\
             -610982147,   -65209897, -1773964827,      -38364,      -44653,  -428923330,    -9691599,  -558027367, -1287373414,     -313521,\
             -908318024,   -38303944,  -622005321, -1905856990,  -262065941,  -378575348,      -94863,  -510451108,      -76617, -1141524628,\
             -549526472,  -713282796, -1469841059,  -162309060, -1473239827,   -78123950,       -2298,  -141208640,     -990609, -1824308773,\
             -950476425, -1944887067,       -8333,  -111661604,      -31513,      -32856,       -7828,  -364875839,     -539601,   -13802468,\
             -227440394,      -56340,        -311,  -997816437,  -270607494,       -9532,    -6249699,      -16114, -1887561200,  -793235247,\
             -655913, -1163531030,  -856258640,      -84374, -1377226160, -1623019263,  -451856702, -1818512118,  -223634525,   -75655100,\
             -566821049,      -99809,     -716710, -1371382844,  -537861335,  -432785053, -1338153158,  -798872636,  -353967457, -1335656703,\
             -621649420,  -862987934,  -655756266, -1786707397,     -461144,    -4372379, -2053895313, -1783268186,  -299978840,      -84392,\
             -966043588,  -103715313,  -588603023,      -28940,  -930778283, -1930195102,  -307848311,  -334027241,  -220616171, -1941189148,\
             -1147589388, -1393579145, -1811422481,  -739844015,    -4345296,  -466924664,    -5387551, -1251413118,     -118597,     -282078,\
             -17669,   -31894192, -1778737564, -1201789608, -1763893221, -1975507563,       -1577,       -6487, -1885914363,    -4038589,\
             -1354979603, -1450036377,  -903793031,    -9316947, -1534415288,  -134074475,  -340684089,  -782663681,       -5953,       -2068,\
             -1425482206,  -933053706,    -4508650, -1569346766,  -594728158,      -16113, -1520739529, -1659989404, -1020098966, -1422568611,\
             -332235828,   -96592372,  -178114640,     -514918, -1994807095,   -29377562, -1879260675,  -357292711, -1595177444,      -10491,\
             -8662910, -1708389072,       -1006,  -736292522,      -86959,   -35866700,     -890712,  -969905493,  -555602926,  -659151543,\
             -1928870094, -1678848769, -1857139178, -1280962182,  -409209265,        -849,     -168181,     -399375, -1699079123,  -952572586,\
             -6293, -2068010438,     -980276, -1565681164,  -488862966, -1743156993,  -579005271,       -8354,   -15937949,  -237748429,\
             -2071656729,    -5606259,       -6202,  -441520985, -1114615151,  -405319305,   -42630921,  -189619015,  -816718306,  -211280642,\
             -329387718, -1354391834,  -358177314, -1827844176,  -141700566,     -454383, -1492371668,       -9820,  -196010527,  -895517069,\
             -411913002, -2085778176,       -6774, -1898664591, -1925738142,  -240243609,    -7543686,   -66476366,  -972340301,    -5274354,\
             -1597236490,  -374360603,     -195485,  -321566807, -2087629158, -1703018277,  -615271374, -1325312842,  -844124382,  -252473577,\
             -893543090, -1873987310,   -68576495,    -2154778,  -267516117, -1883862475,  -985611067, -1438451264,       -1479,  -697116592,\
             -355100240, -1112514868,    -9939633,  -425126616,       -5153, -1103155313, -1490619151,  -363308031,  -933349750,      -63828,\
             -1041244911,  -375579667,  -190476389,  -322358077,  -561266023,   -51283251,       -8542, -1934179668, -1371160569,  -302593127,\
             -440326423,     -687227, -1674719287,  -635607591,       -9098,       -5910, -2141485677,    -4687634, -1464380768,   -59778902,\
             -1651730070, -1306062210,   -40791073,  -273683336,  -392494798,     -268049,  -438157115,       -7991,  -952249638,   -65986125,\
             -728113816,    -8676426,    -5452930, -1939625966,   -68148112,       -5466,  -203997019,   -11271183,   -64508978,     -316291,\
             -5349492, -1554250756,      -98050,  -374344592,   -37050014, -1843878056,      -44357, -1368482859,    -2338360,     -542248,\
             -865967862,      -11782,     -692544, -1322856227,      -57256,  -563587911,    -4857380,  -426045532,  -165208107, -1616252212,\
             -607437444,   -78974899,  -780720733,    -5024252, -1345905147,      -65388, -1380780108,      -32657, -2012133224,       -6036,\
             -456, -2044439907, -1196578744, -1572797483,  -348109644,  -928725096, -2002520046,  -367904846,     -793904,    -4081707,\
             -58219259, -1752635300,  -658337665, -1440858081,  -525967807,  -203599107, -1644121077, -2053177540,      -83739,      -36908,\
             -263116023, -1545739276, -1004854428,    -6613679,       -3312,      -55718,  -235909657,    -6685418, -1345685813,     -137233,\
             -43298034,      -16191,    -6784382,       -2164,  -166228467,  -857483204,       -3326,  -842850990,  -713237480,  -755395440,\
             -1208902575,  -719717706,  -586816712,  -912820526, -1639169954,   -96220520,    -2648428,  -429502312,     -116575, -1620601395,\
             -1705464618,       -1586,   -68996167,  -231101934, -1300147201,  -316333566,       -7150, -1421667537,    -2406710, -1391977709,\
             -2133075221,    -3950384,  -440073323,   -99257311,  -895513929,  -260798711,  -368942477, -1753383794,     -298358, -1831592602,\
             -695107,    -7575252,   -25156273,    -4469865, -1007617487,  -807693641,       -6790,  -171831991, -1484733984,   -36803544,\
             -29126,  -907479229, -1074779755,    -2756041, -1368618728,      -75816, -1655965141,  -170878840,   -34898755,  -336828069,\
             -95292688,      -57280,     -608089,    -6879032, -1072426465,  -178074837,  -594858230,      -59745, -1080871063,  -102012965,\
             -1460409430,  -775824238, -1760931909,    -3117869, -2043895743, -1090586312, -1921588004,      -93895,      -70434,      -31202,\
             -25230486,       -6264,       -1899,   -46337643, -1256419717,     -106158,  -511376902,      -71537, -1791810738,  -980661372,\
             -928346953,  -850753236,    -1179643,   -41957045,  -991530244,  -138700874, -1508519573, -1578808443,  -634989640,  -410476499,\
             -2032872496,  -765399663,   -50271595,  -848435030,  -259949170,  -281474888,  -560537763, -1668594966, -1606453862,   -15437977,\
             -9185796,      -14009,     -931010,    -1202199,       -6537,        -505,       -6393,  -852540502,  -292222066,        -222,\
             -1683802,    -8680822,     -110671, -1778259676,    -6004133,  -610298807,       -8677,    -4901916, -1128714441,      -39798,\
             -1907823591,  -799240819,    -2349091, -2012487577, -1283744730,  -567029239, -1944668135,  -546495274, -1302520820,  -996692724,\
             -2025107487, -1766804668,  -736515607,     -821955,       -9311,   -34759642, -1343194222, -1521367737,  -580967731,      -41008,\
             -757801806,   -73711562, -1153816548, -1609303116,    -6153086,  -131831145,  -444189461,   -81315305, -1585958372,    -5621327,\
             -181312540,        -533, -1303835725,  -885652695,   -68147888,      -53738,  -737803705,  -260973624,      -81903,    -2232115,\
             -578768122,     -992307, -1852053680, -2012431430,    -3806258,  -370625996, -1499359118,   -97353013,  -347194239, -1513318305,\
             -6391372,  -239455873,      -88164,       -3794, -1472807295, -2045292683,       -9909,     -617804,  -912474915,  -298387504,\
             -58076,    -7540061, -1963462438,      -13738, -1219258239, -1804165669, -1832202518,   -87487812,     -999513,      -37174,\
             -71597788, -1731992644,  -255232676,     -138801,  -580444288,  -536101369,    -1502961,  -918793647,     -820360, -1432878857,\
             -8594,  -749804972,   -31564372,  -858815640, -1210653399, -1709666062, -1835308550,    -6316643,  -210026469,       -9440,\
             -90650145,  -142525324,    -8145382,     -675874,  -721851396,  -174541936,       -7202,  -104896061,      -53021, -1390095017,\
             -319920075,  -286238738, -1775775008,  -743587417,  -334085288,     -601195,   -56214644,     -124616,  -801867947,   -99036154,\
             -2890352,       -9694,      -13409,       -1565,      -77419,  -995075115, -1941944640,  -733608854, -2098827287, -1175202914,\
             -59553, -1679668365,       -8597,     -336895,  -627292149,      -87338,     -725479,  -110131071,      -46199,    -4308786,\
             -1412193660,  -272682031,      -95758,      -11753,    -3826508,  -881558564, -1513195320,  -340015317,       -9763, -1103212597,\
             -806345119, -1002993725,   -72262424, -1315315577,  -816071049, -1851115894,    -2315058,    -8692570, -1734642240,  -503418903,\
             -323591730, -1616912845,    -1556126,       -1730,       -9181,  -481679687,       -3044, -1625036768,     -570779,     -682888,\
             -26462,  -856876528,       -5076,       -3908, -1196139267,   -52036890,  -971315380,  -235529085, -2020785664,      -14268,\
             -103987803,    -3092461,      -47387,   -58754100,       -2342,  -696437757, -1048854217,     -974086,    -5381698,   -90556723,\
             -1154513282,   -69797792, -1562718058,  -183762391,  -656096524, -1495015009, -1167531289,     -643018, -1571809728,  -950090234,\
             -3153469,  -552429595,  -735633629,  -707816849,  -722440760,    -7663605,     -703282,      -57360,      -92500,     -777297,\
             -12620571, -1271788423,  -731179873, -1784703276,  -772963684,   -48578127,  -497238287,   -23367509, -1889754256, -1311508655,\
             -1798028, -1255190635, -1440660975,  -963509495, -1157892390,    -1393219, -1787130092, -1745315398,       -5540,  -613174530,\
             -1957426643,      -95829, -1772919321, -1419176538,  -274646805,  -426797981,       -7088, -1726671742,    -8415445, -1840061643,\
             -1214051282,    -3204930, -2008636124,  -217912439, -1858679529,   -82512917,       -1912,  -687637289,       -6232,      -14325,\
             -96348401,       -4584, -1874011812,  -337553692,  -580334434, -2011182294, -1527732467,  -255953711,  -676634441,    -6667069,\
             -9249, -1248728041, -2070986076,     -571301,  -335073284,   -31262938, -1713221441,        -342,  -680860943,  -954921769,\
             -674134766, -1052692885,        -758, -1347785800,  -169264922,    -9752926,  -548923795,     -662380,   -98526634, -2107111326,\
             -1191973123, -2070320133, -1865476591, -1024270640,   -15862101,  -866454382,     -285041,  -889618542,  -739200608,  -840930105,\
             -1223982120,   -52081216, -1315875337,    -9176040,    -7544166,  -880076563, -1388482113,     -823994, -2137483319, -1509385684,\
             -1363523001,     -195691,       -3757,      -19111, -1660643057,  -500508999,      -23019, -1120076086,      -19057, -1177102205,\
             -1636835960,  -508267573,     -697709,  -966412842,      -48859,    -9465469,  -763741807,  -124776598,   -56858305, -1645552705,\
             -367213,  -995374322,  -969108664, -1259631434, -1356769489, -1370455435,  -929448212, -1398904561, -1285464590,      -41594,\
             -10018,      -86904, -1179021247, -1993595047,    -3364854,     -176232, -1012499763,  -191635102,       -8783, -1076590531,\
             -182776926,  -201427483,  -504597498, -1503709267,      -23745, -1809894147,     -718969,   -83750506,  -633741118, -1810679780,\
             -8475705,      -51753,   -14804957, -1487696569, -1387767440,  -441021089, -1590097504,      -53509,  -620301929,  -289541163,\
             -469796636,       -9236,    -4056015,  -330591729,      -34026,  -207962539,  -577833403, -1471330551, -1653545853,  -974635065,\
             -257620480, -1058958544,  -115297030,  -461654678,  -431604271,       -8964,     -413734,     -609950,  -724614946, -1835397755,\
             -127440743,  -938976103,       -7018, -2018612439, -1312464793,  -720177373,   -83436559,       -1055,  -792825390,   -57538370,\
             -62295,  -653001245, -1755421693,  -489191971,      -81921,     -660083,      -86063, -1371869723,      -83250,    -3433743,\
             -388625094, -1503331682,  -745584990, -1789590386, -1395046701,      -94732, -1134558005, -1089981132,  -110314552, -1912926358,\
             -1834386430,       -2471,     -450617,  -780963436,     -181199,  -412576504, -1804154649,    -2687452,  -705382635, -1354503665,\
             -280684319, -1501773053, -1439960295,   -72395671,  -584411246,  -239133827,  -742510415,     -979896,        -819, -1685505559,\
             -40224804,  -699692639,      -98842,   -64528744, -2075204792, -2051654740, -2030982493,   -30602002,   -74816590,      -63638,\
             -1210229479, -1436774653,  -140230028, -1790442114, -1144046309,  -867447092, -1182888268,      -52932,   -87655820,   -12823882,\
             -57, -1097232055, -1689239699, -1790381115, -2075656345, -1946467290, -1985365896,  -513867284,       -9750,   -86354249,\
             -8198607,   -82459043, -2123109646,   -87479474,   -75882188, -1140202177,   -76318739,  -941161127,  -108165445,  -336151469,\
             -1410479032, -1821608326,      -31972,  -537917274,    -2778739, -1025840504,   -74546618,  -103830576,    -8201409,  -564581243,\
             -961824, -1597952780,       -9446,  -618363029,       -6624,       -6103, -1144801772, -1629018970, -1596318362,  -397562682,\
             -128678860,   -70742939,  -696592968,  -472543842,    -2943411,      -58938,  -146490781,      -61384,    -8976736,     -751193,\
             -1840026752, -1643580892, -1677191627, -1473029261, -1154073381,  -822454842,  -286026539,    -8411426, -1545049676,       -1802,\
             -240891521,    -2614634, -2074254684,       -6629,    -6701427,   -11319053,       -6007,  -480466041,    -2246955, -1987052075,\
             -1999645517,      -61302, -1178655698, -1764843755,  -499590779, -1812699914,  -566637766, -1853507100,      -36713,  -179408161,\
             -615567925,  -287149984, -1717276258,  -581278640,  -956904807,  -833769335,   -12191741, -1481252413, -1761528482,   -98969306,\
             -631501897,  -557211886,    -7038630, -1512747051,      -68899,     -214113,     -619779,  -993878014,      -72451,     -266294,\
             -236924,       -2704, -1001268721, -1698857749, -1482478182,   -16691751,  -500534248, -1684236327,       -6855,  -208451595,\
             -722759, -2000986991, -1664817665,  -897638163, -2017384376, -1118496655, -1134541272,  -914857691,  -425129268, -1326862517,\
             -546991,     -636127,  -126202429,      -80450, -1767335950,  -909681828, -2011753998,  -506480749,      -64919,  -627422878,\
             -1323292811,   -48630021,    -7920656, -1718811869, -1084948550,  -467746584, -2031811589,    -5902458,       -4221, -2013570132,\
             -1574228182,   -23918604,       -6788,    -9645283,  -675445403,  -206085698,   -22238487,       -3739, -1755182536,  -270608257,\
             -113979521, -2085205942,  -838231943,      -43418, -2015392365,     -950600, -1943232055,  -615717191,     -599925, -2097128059,\
             -1174038517,  -543561246,   -46620156,  -267863377,     -214429,  -269793670, -1865384836,  -667525042, -1658150905,   -88864849,\
             -2088531470,       -2026, -1250859594,      -71010, -1720549204,      -35173, -1259420819,  -790608862,  -169387257,  -169361831,\
             -661894591, -1219509216,       -6815,     -735597, -1713946962,  -272920562,  -358435817,    -7204010,   -73447023,  -320333250,\
             -1366487,    -7022884,       -3264,  -258050200,  -768974722,  -545041320, -1562252519,   -49574379,     -709046,       -6292,\
             -1889560260, -1300358671,  -984511143,  -663562672,  -248937454,     -397380, -1591472571, -1591962329,  -444619273, -1228833923,\
             -466853,       -1250,  -805669719, -1268130546,  -165763259,         -65,    -2366457, -1926756553,  -210560029,     -710910,\
             -75668,  -290288068, -1216432622, -1855940990, -1503030117,   -81880014, -1216784728,  -821406869,  -374320788,  -836580338,\
             -103871990,      -34257, -1115537410,       -2513, -1997937299,     -673053,  -778490355,  -462823392,    -9526597,    -5544671,\
             -1163193588,       -8555, -1572073608,     -452261,  -185055017,       -2521, -2072677180,  -967305624, -1062149636,       -6043,\
             -290065175,     -841934,  -649772028,  -597609586, -1837577784,  -313268901,  -206352311, -1594025808,   -63641399,  -678687906,\
             -2090816282,  -811918125, -1488312822,  -938351188,  -609848917,      -11774, -1702963613,  -412675219,       -4183,  -583845491,\
             -995014083,       -2487,    -5138069,  -545040336,  -654640512, -1972369331,       -3785,    -4022192,    -8295522,       -4285,\
             -381451657,       -8703,  -268864946,  -776166492,    -5253941,  -863830762,  -327063769, -1934772488, -1780689999,   -55166943,\
             -924991127, -1744998533,      -90145, -1141516843,  -800314495,    -8275844, -1635894245, -1686392529,    -2112751,     -478226,\
             -688732994,  -594256902,  -468197253,  -579115669, -1425202701,       -7665, -1338436372,      -84005,   -38108340,  -444055333,\
             -3294,  -651222125,   -74251262,      -68723, -1736807184, -1918073152, -1758802156,    -2111240, -1363014526, -1527966261,\
             -9801,  -348904977,      -67066, -1643952266,  -395314730,   -54924232,  -401260710,       -6986,      -33411,       -9532,\
             -1736253604,       -9117, -1919202310,   -71927797,  -656309348, -1605877390,       -1266,  -445618720, -1234992865,  -630190495,\
             -96858, -1908931246,    -4544770,  -940234023,  -643382990, -1105058273,  -358653526,  -412502619,    -3021992,   -47528721,\
             -1481999582,   -46329438,     -663495,   -81678193,    -7512179,     -342167,  -540935474,   -41205485,   -66133114, -1271114944,\
             -27037323,  -112164575, -1207227487, -1080311525,       -6602,     -682657,  -767891248,     -748875,    -6529558,      -72072,\
             -34574552,  -757144935,  -700114396,  -258539448,   -44970316,    -2166512,    -8679390, -1323719508,  -869451721, -1753379977,\
             -1233377352,  -602169823,    -3326012, -1566771566, -1879455140,  -503277412,  -562725189, -1934218797,  -624611904,  -748415794,\
             -1852,  -895178439,  -977072218,  -967456664,  -531306804,  -923685579, -1864256645,      -81394,          -4, -1935341884,\
             -546679589,        -164,  -240014689, -1576062140,  -956935672,  -257970310, -1004052665,      -44471,       -6360,  -802621034,\
             -699169739,  -461053227, -1590798189,  -132582834, -1871244074,      -25594,  -611603185, -1767793332, -1820289050,       -8700,\
             -69746772,  -959505470,  -623969809,       -7342,  -533314489, -1970250972,      -63693, -1094968376,  -457655817,     -689022,\
             -122829157,  -167257161,   -16416664,     -305753,  -142986916,     -156369, -1376248778,   -33132186,  -115272252,   -95827552,\
             -1648186120, -1857642903,    -8718126, -1616054461,       -3605, -1555257368, -1942442768, -1891321226,  -994472683,      -13787,\
             -1615441469,      -90955,      -61662,  -732063407, -1195741935,      -42009, -1001427860, -1948093271,       -7622, -1478158750,\
             -1205999987,      -78012,  -487946767,  -124348731, -1968706988, -1848810279,  -933115123,  -486977466,     -220527, -1776618549,\
             -540576675,      -18930, -1397569410, -1579500818,  -332611184, -1735032086,  -766472246, -1934651532, -1111660717,      -26886,\
             -484416,    -6273276,  -650804179,  -809948243,   -80959679, -2088064091,       -3857,  -625168380,  -194111780, -1744521475,\
             -1215762243,   -42305204,    -9810562,  -489932398,          -2,   -30855894,  -694584030,  -735252195, -1407534635,  -561348213,\
             -298631847,  -453018705,      -98391,       -5540,      -28719, -2106102652,    -4569205,  -590024044,      -79913,  -617563772,\
             -1130,    -7926048,  -472279899, -1185961600,      -79689,      -51304, -1582020650, -1721530698,   -35626146,  -213156468,\
             -46360470,     -410795, -1090503143,      -66077,  -358257334,     -959945, -2032898453, -2023883434,      -25594,  -577326401,\
             -385842229,       -7285, -1841588602,  -564466079,      -84692, -1672611553,      -88609, -1172146810, -2123015565,   -96650924,\
             -2049379998, -1662409935,   -40703962,     -640723,  -916362756, -1932628973, -1679169374,   -15152255,      -49827, -1603022509,\
             -1549305955,  -937699024,      -33840,  -198157999,  -475556920,      -52146,    -4442298,    -8897488,     -108004,    -5555554,\
             -80910139, -1726640374,  -518328417, -1200885230,  -836455381, -1078393744,   -87367613, -1118552273,   -76712052,  -296310594,\
             -56490, -1346137278, -1965017879, -1908856453,   -80329863, -1639867614,  -486652091, -2022683171, -1142063951,  -184248639,\
             -821943141,     -957180,  -451686729,      -69979, -1347857950,  -368213360,      -51894, -1150476585,     -587119,  -335316858,\
             -1694349737,  -137383800,      -90454,    -4059879,       -8156,      -61156,      -72943,  -762663444,  -553383971, -1645682057,\
             -23970,   -38866692,   -82602446,       -8862, -1599437763, -2099224211, -1913710269, -1703165461,     -667245, -1352033237,\
             -8924894,  -383101254,  -370952450, -2021127971, -1976413799,   -58245142,  -916855361,   -22550067, -1389874316,       -4861,\
             -97920,   -26720984,    -9162951,  -768276826, -1573321709,  -656361794,     -419394,  -738941523,       -9186,   -67580079,\
             -6046206,   -63799357,       -9704, -1873124961,  -396002723,     -872686,      -12070,    -1439551,       -4703,  -381308493,\
             -1612807628,  -111889445,  -204585799,    -6178602,       -2957,  -123020572,   -62150723,       -8205,  -687153803,   -27242643,\
             -1631369014, -1600420510,   -41075625,  -889844125,    -9299810,  -674101213,  -439205662,   -95220465,      -63590, -1176512409,\
             -1399743787,      -50325,       -1720,    -1072681,  -575601918, -1244265073,  -717729709, -1160988343,  -862450395, -1033496410,\
             -6658,     -758948, -1041349139, -1865407925,  -561235926,  -528385288,      -12911,  -610544828, -1321361138, -1546314898,\
             -65759817,  -269701971,    -9974921, -1228274526, -1865850518,  -390887172,    -4294043,     -214453,  -640964904, -2096457722,\
             -1064171009,  -400150033,  -229636623, -2131734018,  -971034405,  -550406195,   -38698129,    -9472186, -1743817384, -1871909165,\
             -3230680, -1927548515, -1076403223,      -91935,      -76191,  -662024468, -2084576460,    -2175653,  -566209617,  -208481413,\
             -38141922, -1179319129,       -6818, -1891129287, -1662515180,       -5027,  -962292180,   -16398601,      -46623, -1557296473,\
             -450789399,   -75880594,     -737414, -1283283185,  -269136892,    -3101842,      -36846, -1073580810,   -17875550,  -491409905,\
             -498216,  -593406793, -1246258824,     -206678,  -492451504,     -207299,  -398458632, -1134511702,    -5696192,  -188139257,\
             -41586,  -807270733,      -88559,       -8509, -1359489399,     -751175, -1306921104, -1395149750, -1346508836,  -900204939,\
             -8944240,  -506391520,       -5915,  -938191184,       -9920,  -966881803, -2004135346, -2018771287,   -77749735,  -431998188,\
             -9520995, -1349218101,   -37895212,      -78053,   -32400219,  -642571817,    -4185311,      -89420,  -719358202,    -3671783,\
             -37935581,  -533498759,   -77631471,  -615948423,      -39322,   -85303531,  -600876961,       -1522,  -197116880,      -33056,\
             -51096365,     -669314, -1412972627,      -48055,  -960896579,    -6272052,  -560811572, -1309144783, -1369508550,  -561166210,\
             -591266,  -168706224,   -62513267,     -819425,       -6864,  -146976500,  -489903590, -1536952381, -2047555665,    -9723173,\
             -569518100, -1922978353,  -603497319,  -608491480, -1451498972,  -490162150,  -150036413,   -43220827,     -693218,        -381,\
             -1499645448,  -166749878, -1347463976,    -4641170,        -495,    -6934269,    -6483375,  -547162214,   -53079471,   -86259246,\
             -425791892,   -59697560, -1807371908,     -300744,  -106045288,  -900588876,  -569746911, -1410673724,  -151365288,     -385028,\
             -199893112,  -411685548,  -433400446, -1211069225,       -9746, -1106599228,  -706749838,     -877162,     -171155,      -80443,\
             -99888065, -1096911720,       -4674,  -837189769,  -594242999,  -815518568, -2046354114,  -751380706,   -64385750,  -104583831,\
             -4234331,      -18659, -1550072264,  -218964340, -1728975184,  -222439854,  -412047728,      -32859, -1387755562,  -244351780,\
             -1759992579,  -175962589, -1164414560, -1254319856,  -171001414, -1516250911, -1490307761,  -964616260, -2036861366,     -626221,\
             -37376,      -20030, -1954475275,  -482608473,  -419080697,  -804505887,     -492123, -2108187134, -1123813679,  -379136358,\
             -1530870577,   -55929190, -1564746454,  -614274225,   -10897906,  -339989716,    -8354099,  -943011180,    -6491900,    -6420711,\
             -7074611,    -5549486, -2016632865,  -754497519,  -556912311,  -786950863, -1255184183, -1434143208,  -645348753,  -777231759,\
             -704323365,   -66936960,      -43985,  -148054275,      -31483,   -57807689, -1882626153,  -215372735, -1872736500,  -731446640,\
             -87414, -1590212257,  -398250257, -1083059484,       -7886,  -302673855,       -5132,   -38526777,  -685090184, -1290196158,\
             -92971,      -60058,      -34039, -1712632249,  -263387898, -1382108482,      -40390,  -278090257,  -377589965,  -469867976,\
             -873633993,     -752399, -1152864164,   -36093779, -1740651239, -1073542717,   -94615214,   -19301077,   -33451167, -1434209947,\
             -279077917, -2064838472,  -152772666,  -415630609,     -674072,  -745589228, -1421066973,  -893342334,        -710,  -520936798,\
             -1383308444,      -99455,       -8921,       -4426,  -274479161,       -9388,        -444, -2053407293,   -41365138, -1410704823,\
             -152202522, -1328589301, -1632902724,  -156078175,   -89913151,  -951196899, -1248648262, -1406526764, -1917516708,   -82170552,\
             -666295383,      -22053, -1326565974,  -895354599,     -387176,     -609323,   -22023261, -2059057186, -1388669328,     -898518,\
             -1620631014,    -9820950,   -17118735,  -847717998, -1312986197,    -4520304,  -950333788,  -629219462,  -971739119,    -8053785,\
             -2007910403,   -72189933, -1149957413, -1505130883, -1634891222,    -9097717,   -39343089,      -57567, -1777931621,  -444656132,\
             -200805,  -506244383, -1238290377,   -65434894,      -28464, -1489579279, -1154860820,       -3760,  -264638267, -1237581041,\
             -45398379,  -494101039, -1926015236,  -130426219,  -525054016, -1633661191,  -211541699,  -563378568,  -530651086,     -432345,\
             -897776707,  -471466802, -1488893205, -1403016217,   -82570316, -1688834568,   -20159143,  -108644007,     -427462,  -903847969,\
             -3763135, -1179894021,  -356731165,  -680629748,   -66063889,  -985337587, -1522520096,  -202504959, -1228419242, -1431589738,\
             -49737157, -1918910839, -1233298690,  -907428222,     -266996,  -173681441, -1743476726,  -822078793,  -746165370,    -9362751,\
             -9591766,   -43983161,       -7663,  -251288854,  -992941344,   -74201513,  -515274022, -1868441519,     -695886,      -62498,\
             -5995,     -871756,    -7736505,      -73936,       -9113,  -922635086,    -5862760,  -558892350, -1099841372,        -316,\
             -29190197,  -897259091,  -772454911,     -615462, -1002522423,     -598408, -1336790343,  -295923432, -1974022309,   -83388317,\
             -677734,    -8067331,     -741751,    -3396159,   -71850100, -1824863881,   -11363871,    -2764276,    -9573035,       -6804,\
             -387679, -1228122718,  -137022705,     -708191,    -8226621,  -819371701,    -7310003, -1717034994,  -970940549,  -641026183,\
             -17655, -2104073231,  -202484866, -1007078080,      -41310, -1167701778,   -32685540,  -871073805,   -81278133,       -2561,\
             -1211148723,  -903241072, -1336647079,   -10462322,  -155416005, -1203924772,  -753624101,       -2371, -1129407188, -1296456248,\
             -1378816807, -1286392853,  -744321295, -1800153247,   -70098562,  -384482140, -1389020437, -1146655953,   -39032232, -1748878155,\
             -1066852460,  -436337633,  -854672640,  -763723277,  -192649413,   -95326714,  -505942579, -1828659669,    -3114250,  -501695143,\
             -997,  -484197393,     -590599, -1755392908, -2139281872,     -329886,  -957199265, -1615641792,   -36641316,     -466901,\
             -2000973130,  -539212142,  -386716565,    -9443344,      -42660, -2093141175,  -433287464, -2080149998,  -111105335,  -198804110,\
             -1106649754,   -90535111,     -837321,  -352880290, -1660566223,  -107464800, -2042040301, -1279808134,  -509502503,       -3867,\
             -637473774,   -65801293,  -375867653,  -744769370, -1117009256,     -888800, -1803469262, -1508844729,  -804821560,  -393917882,\
             -8002959,  -965099373, -1584752780, -1884664845,   -18748710, -1306975788, -1206115220, -2028591617,      -30555,         -44,\
             -74128097, -1793659162,   -36415202, -1951065306,   -84734243,       -3303,  -648436427, -1720005383, -1463074645,   -19689433,\
             -1486379366, -1998205353,  -940679762,     -582059,   -42022985,      -39555,  -371597453, -1449295966,     -409141, -1986953343,\
             -1581920127,  -958360304,  -244348925,     -735286,   -88243436, -1645635315,    -2972419,    -7540432,  -193500534,    -2904777,\
             -1664059513, -1594192675,  -976105569, -1447315918,     -699769,   -48641606,    -2466235,  -387078759,   -54708077, -1947013502,\
             -5318741,      -35174,    -1690553,       -3779,     -310319,       -5570,  -677245020,   -33356709,  -616311788,     -280518,\
             -900960296,  -575692762,  -163127157,  -581774622, -1982698981,  -986101144,  -691047314, -2029815430, -1156747140,      -84740,\
             -473833979, -1898926502,   -74745173,  -847392959,      -59283,  -875496985,     -743344,  -710349490, -1889051152,       -2980,\
             -71110,     -986836, -1455995046, -1430437650, -2066104648,  -894340269,     -174132, -1154368327,  -986856718,      -14543,\
             -5616, -1953779884,        -703,      -16670,   -32028350, -1999536606,  -130285005,       -5013,  -991910833, -2038101329,\
             -507426173, -1949921418,  -529007910,     -514870, -2100540103,  -656229750,   -42467280,   -94950837,  -394791619, -2122337996,\
             -550717353,       -4169,   -22258211,       -2167,  -134877650,   -87244223,  -782127701,   -56191906,  -611203539,  -464862799,\
             -749394823,   -51441782,       -9661,     -421454, -1626809873,    -7528988,       -7460, -1621603071,  -494747209, -1788210796,\
             -418448,  -311733001,  -552689252,  -389522625, -2117756962,  -800621466,   -89395610, -1012270215,   -44366572,       -9253,\
             -1498506512,  -234069132,     -390559,  -403081748,  -240387639,   -18989739,     -430495,  -765414471,  -670182775,       -4755,\
             -3604,  -254758019, -1447191460, -1697595571,  -113681857,   -57307852,  -332482767, -1419118000,   -51853661,  -265242707,\
             -1980092566,  -166641203,      -53168,      -69343,  -345415563,      -65988, -1299377339, -1148628699,  -889049168,     -545414,\
             -804196673, -1315647375, -1705233055,  -881007502, -2068428353, -1351159522,       -8823,      -25591,    -1742956,    -4995618,\
             -32031,      -10530,    -2299341,   -60899380,   -25580311,       -1663, -1524840066, -1291430158,   -77217634,  -676609938,\
             -82023389, -1710603642, -1616304592,  -291912644, -1938847476,   -72694355,  -763361516, -1223661166,    -2076316,  -874174586,\
             -83067, -1792850821,  -468330086, -1844843364,   -79303532, -1537108582,     -700592, -1579320179, -1755911721,       -3934,\
             -8198709,    -6248845,  -791725704, -1357320000,  -802708558,  -957039565, -1271358511, -1626606338, -1992793177,       -5437,\
             -32108,   -39685295,  -517081372,     -167034,  -982304018, -1968482997,     -250132, -2041435965, -1877548852,  -111938496,\
             -406526550, -1626616419,  -166495477,  -345886805, -1493804924,  -360167231, -1780908302,  -621516989,   -32299159, -1596548650,\
             -1480643862,   -49955912,       -5860,  -765829593,  -179178517, -2005732936,       -2472,   -87843066, -1606466211,        -539,\
             -1235675561,  -897160108, -1060415821,  -770602464,    -1852641,  -860252036,    -1040123,     -285762,   -60378816,  -850781421,\
             -1230652124,       -7883,  -265293202,      -48588,       -1378,    -9465545,  -759126610,       -1825,   -48885229,  -996081055\
             ]
    return my_bank[int_idum]

def interp2D(Nold,Nnew,B):
    Btmp = np.zeros((Nold,Nnew))
    Bnew = np.zeros((Nnew,Nnew))

    xold=r_[0:Nold]/(Nold-1.0)
    xnew=r_[0:Nnew]/(Nnew-1.0)

    for i in r_[0:Nold]:
        Btmp[i,:] = interp(xnew,xold,B[i,:])
        
    for i in r_[0:Nnew]:
        Bnew[:,i] = interp(xnew,xold,Btmp[:,i])

    return Bnew
