'''module to comute the covariance matrix of observed 2PCF multipoles

    functions are dived in 2 categories:
        - Theoretical covariance
        - BestFit covariance
    note: GaussianCovariance implementation from
                https://gitlab.com/veropalumbo.alfonso/gaussiancovariance
'''
         
#==== import needed libraries =====
import BeXiCov.Models as Models
from GaussianCovariance import *
from iminuit import Minuit
from scipy.linalg import cholesky, solve_triangular
import numpy as np


#===== Theoretical covariance
def ThCovariance(s, volume, number_density, cosmo, space, rectype=''):
    '''
        routine to compute 2PCF Gaussian covariance Grieb et al 2016) given the expected non clustering signal
        GaussianCovariance implementation from
                https://gitlab.com/veropalumbo.alfonso/gaussiancovariance
                
        s: array of separation bins
        volume: (float) survey volume
        number_density: sample number density
        cosmo: list of cosmological parameters, example in Models.DefaultCosmology()
        space: 'RealSpace' if the template has to be computed in real-space,
                'RedshiftSpace' if in redshift-space
        rectype: string of reconstruction type. leave empty for no-reconstruction,
                 'rec-sym', 'rec-iso' for the Zel'dovich reconstruction without and
                  with RSD removal
        returns:
            xy, yx = np.meshgrid(sconc, sconc, indexing='ij'),
            covariance matrix,
            corr matrix
    '''
    
    kh, p_k_mu = Models.P_k_mu_models(cosmo, space, rectype)
    
    #set output bins
    
    Dr=s[1]-s[0]
    rad_edges = np.linspace(min(s),max(s)+Dr,len(s)+1)+0.5*Dr
   
    n_bins_r = len(s)
    l_list = [0, 2, 4] #multiple list
    n_data_points = n_bins_r * len(l_list)
    
    
    xi_covariance = TwoPointGaussianCovariance(rad_edges, l_list, deg=51, deg_k=6)
    cov = xi_covariance(p_k_mu, volume, number_density)
 
    std = np.sqrt(np.diagonal(cov))
    corr = cov / np.outer(std, std)
    sconc=np.concatenate((s,s,s))
    xy, yx = np.meshgrid(sconc, sconc,indexing='ij')
    
    return xy,yx,cov,corr


#====== BestFit covariance ======

#class to obtain an estimate of the covariance matrix reproducing the data
class BestFitCovariance:
    '''
        class to compute 2PCF Gaussian covariance (Grieb et al 2016) from the best fit of the data
        GaussianCovariance implementation from
                https://gitlab.com/veropalumbo.alfonso/gaussiancovariance
                
        s: array of separation bins
        xi_elles: (3,len(s)): array measured 2PCF multipoles (monopole,quadrupole,hexadecapole)
        volume: (float) survey volume
        number_density: sample number density
        cosmo: list of cosmological parameters, example in Models.DefaultCosmology()
        space: 'RealSpace' if the template has to be computed in real-space,
                'RedshiftSpace' if in redshift-space
        rectype: string of reconstruction type. leave empty for no-reconstruction,
                 'rec-sym', 'rec-iso' for the Zel'dovich reconstruction without and
                  with RSD removal
        fit_range: [smin,smax] separation range used for the fit, default (10-200)
        returns:
        
            xy, yx = np.meshgrid(sconc, sconc, indexing='ij'),
            covariance matrix,
            corr matrix
    '''
    
    def __init__(s, xi_ell_data,volume, number_density, cosmo, space, rectype='',fit_range=[10,200]):
        self.s=s
        self.xi_ell_data = xi_ell_data
        self.volume = volume
        self.number_density = number_density
        self.space = space
        self.rectype = rectype
        self.fit_range = fit_range
        
    def run(self):
    
        #get theoretical covariance
        print('generating theoretial covariance ...')
        self.xy,self.yx,self.covTH,self.corrTH = self.GetThCov()
        #initial guess, can be modified in function
        initial_guess=self.SetInitialGuess(self.space,self.rectype)
        
        print('performing best fit ...')
        self.best_fit_params,self.param_errors=self.FitMinuit(self.s,self.xi_ell_data, self.xy,self.yx, self.covTH,\ self.space,initial_guess,self.fit_range)
        
        print('construct covariance ...')
        xy,yx,self.cov_it,self.corr_it = self.ConstructCovariance()
        return
        
        
        
        
    #==== intermediate routines
    def GetThCov(self):
        '''
            get theporetical covariance to obtain best fit
        '''
        xy,yx,cov,corr =  ThCovariance(self.s, self.volume, self.number_density, self.cosmo, self.space, self.rectype)
        return xy,yx,cov
        
    def SetInitialGuess(self,space,rectype):
        '''
            set initial guess for best fit search
            space: 'RealSpace' if the template has to be computed in real-space,
                'RedshiftSpace' if in redshift-space
            rectype: string of reconstruction type. leave empty for no-reconstruction,
                 'rec-sym', 'rec-iso' for the Zel'dovich reconstruction without and
                  with RSD removal
            returns: dictionary of physical parameters values
        '''
        Phpar = {
                'alpha_par': 1,
                'alpha_perp':1,
                'bias': self.cosmo['bias'],
                'f': 0.8
                'Sigma_par': 10.,
                'Sigma_perp': 10.,
                'Sigma_s': 4.,  # Fix Sigma_s to 0
                'Sigma_rec': 0.
            }
        if rectype == 'rec-iso': Phpar['Sigma_rec']: 15.
        if space == 'RealSpace': Phpar['f']: 0.
        return Phpar
        
    #====== subroutines to obtain best fit ====
    def FitMinuit(self,s,xi_ell_data, xy,yx, cov, space,initial_guess,fit_range):
        '''
            find BF parameters using iminuit
            s: array of separation bins
            xi_ell_data: (3,len(s)): array measured 2PCF multipoles (monopole,quadrupole,hexadecapole)
            xy,yx: 2D coordinates grid of cov matrix
            cov: covariance matrix to be used ro fit
            space: 'RealSpace' if the template has to be computed in real-space,
                'RedshiftSpace' if in redshift-space
            initial_guess: dictionary of initial_guess parameters for best fit
            fit_range: [smin,smax] separation range used for the fit, default (10-200)
            
            returns: best_fit_params, param_errors
        '''
        
        par = self.InitMinimization(s,space,xy,yx,cov,initial_guess,fit_range)

     
        # Create Minuit object and set the objective function
        minuit = Minuit(self.chi2,tuple(par))
        minuit.errordef = Minuit.LEAST_SQUARES
        
        #define priors
        minuit = self.setPriors(minuit)

        # Perform the fit
        minuit.migrad()

        # Get the best-fit parameters
        best_fit_params = np.array(minuit.values)

        # Get the errors on the best-fit parameters
        param_errors = np.array(minuit.errors)

        #getchi2
        chi2_value = minuit.fval/float(2 * len(self.sub_s) - len(par))

        print("Best-fit parameters:")
        print(self.GetParClass(best_fit_params,self.space))
        print("Parameter errors:")
        print(param_errors)
        print("chi2_red:")
        print(chi2_value)
    

        return best_fit_params,param_errors
        
        
    def InitMinimization(self,s,xi_ell_data,xy,yx,cov,space,initial_guess,fit_range):
        '''
            initialize minimization: inpose fitting range, register initial guess
            s: array of separation bins
            xi_ell_data: (3,len(s)): array measured 2PCF multipoles (monopole,quadrupole,hexadecapole)
            xy,yx: 2D coordinates grid of cov matrix
            cov: covariance matrix to be used ro fit
            space: 'RealSpace' if the template has to be computed in real-space,
                   'RedshiftSpace' if in redshift-space
            initial_guess: dictionary of initial_guess parameters for best fit
            fit_range: [smin,smax] separation range used for the fit, default (10-200)
            
            returns par: initial parameter array
            
        '''
        #mask data

        mask = (s>=fit_range[0])&(s<=fit_range[1])
        self.sub_xi=xi_ell_data[:,mask]
        self.sub_s=s[mask]
       
        ellemax=4. #adding exadecapole
        nb=len(s)
        ells=np.concatenate((np.zeros(nb),np.ones(nb)*2,np.ones(nb)*4))
        lk,kl=np.meshgrid(ells,ells,indexing='ij')
        
        mask2D=(xy.flatten() >= fit_range[0]) & (xy.flatten() <= fit_range[1]) & \
            (yx.flatten() >= fit_range[0]) & (yx.flatten() <= fit_range[1]) &\
                          (lk.flatten()<ellemax+1)&(kl.flatten()<ellemax+1)
        
        self.sub_cov = (cov.flatten()[mask2D]).reshape(len(self.sub_s)*2,len(self.sub_s)*2)
        print('shape subcov', self.sub_cov.shape)

        #guess
        if space=='RedshiftSpace':
        
            par=np.array([initial_guess['bias'],initial_guess['f'],\
                         initial_guess['Sigma_par'],initial_guess['Sigma_perp'],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #with exadecapole
        else:
            initial_guess['f']=0
            par=np.array([initial_guess['bias'],\
                         initial_guess['Sigma_par'],initial_guess['Sigma_perp'],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])#with exadecapole
                         
                         
        #Load reference power spectrum for templates
        self.kh, self.Pk_l, f, sigma8,T_cmb = Models.Pk_linear(self.cosmo)
        self.Pk_now = Models.Pk_nowiggle(k,Pk_l,cosmo,sigma8,T_cmb)
                         
        return par
    
    def chi2(self,par):
        '''
            evaluates chi2 function by comparing data with model,
            models generated via self GetModel
            par: parameters of the model
        '''
    
        xi_ell_model_temp=self.GetModel(par)

        xi_ell_model=np.zeros(self.sub_xi.shape)
    
        for i in range (0,3):
            xi_ell_model[i]=xi_ell_model_temp[i]

        residuals = self.sub_xi[[0,1,2],:].flatten() - xi_ell_model[[0,1,2],:].flatten() #with hexadecapole
        
        # Perform Cholesky decomposition on the covariance matrix
        L = cholesky(self.sub_cov, lower=True)

        y = solve_triangular(L, residuals, lower=True)
        chi2 = np.dot(y, y)
       

        return chi2
    
    def setPriors(self,minuit):
        '''
            sets priors of fitting parameters, can be modified
        '''
  
        if self.space=='RedshiftSpace':
            index=1
    
            minuit.limits = [(0., 4),(0., 4),(-3, 20),(-3, 20),(0, 20),\
                               (-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),\
                               (-20,20),(-20,20),(-20,20),(-20,20),(-20,20)]#with hexadecapole
        else:
            minuit.limits = [(0., 4),(-3, 20),(-3, 20),(0, 20),\
                              (-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20),\
                              (-20,20),(-20,20),(-20,20),(-20,20),(-20,20),(-20,20)]#with hexadecapole
            
        return minuit
        
    def GetModel(self,par):
        '''
            evaluate model at the fiducial parameters:
            par: ordered array of fiducial parameters
        '''
        
        Phpar,bbpar=self.GetParClass(par,self.space)
        #get model
        xiTemplate=Models.model_xi_ell(self.s,Phpar,self.kh, self.Pk_l, slelf.Pk_now)
        BB=Models.BroadBand(s,bbpar)
        return xiTemplate+BB

    def GetModelBF(self):
        '''
            evaluate model at best fit:
        '''
        Phpar,bbpar=self.GetParClass(self.best_fit_params,self.space)
        #get model
        xiTemplate=Models.model_xi_ell(self.s,Phpar,self.kh, self.Pk_l,self.Pk_now)
        BB=Models.BroadBand(self.s,bbpar)
        return xiTemplate+BB
    
    def GetParBF(self):
        Phpar,bbpar=self.GetParClass(self.best_fit_params,self.space)
        return Phpar,bbpar
    
    def GetParClass(self,par,space):
        index=0
        if space=='RedshiftSpace':
            Phpar = {
                'alpha_par': 1,
                'alpha_perp':1,
                'bias': par[0],
                'f': par[1],
                'Sigma_par': par[2],
                'Sigma_perp': par[3],
                'Sigma_s': par[4],  # Fix Sigma_s to 0
                'Sigma_rec': 0.
            }
            index=1
        else:
            Phpar = {
                'alpha_par': 1,
                'alpha_perp':1,
                'bias': par[0],
                'f': 0,
                'Sigma_par': par[1],
                'Sigma_perp': par[2],
                'Sigma_s': par[3],  # Fix Sigma_s to 0
                'Sigma_rec': 0.
             }
        bbpar={
        'b0-2': par[4+index],
        'b0-1': par[5+index],
        'b00': par[6+index],
        'b01': par[7+index],
        'b02': par[8+index],
        'b2-2': par[9+index],
        'b2-1': par[10+index],
        'b20': par[11+index],
        'b21': par[12+index],
        'b22': par[13+index],
        'b4-2': par[14+index],
        'b4-1': par[15+index],
        'b40': par[16+index],
        'b41': par[17+index],
        'b42': par[18+index],

        }
     

        return Phpar,bbpar
     
    def ConstructCovariance(self):
        #construct anysotropical BF physical Pk
        '''
            construct the best fit covariance of 2PCF multipoles
        '''
        P_mu_k_Ph=self.get_physical_Pmuk()
        P_mu_k_BB=self.get_BB_Pmuk()
        P_mu_k_all=lambda mu,k:P_mu_k_Ph(mu,k)+P_mu_k_BB(mu,k)
        self.P_mu_k_all=P_mu_k_all
        
        xy,yx,self.covBF,self.corrBF=self.BuildITCovariance(self.s,self.kh, P_mu_k_all,self.snap)
      
        return xy,yx,self.covBF,self.corrBF
    

        
    #==== subroputines to construct BF covariance
    def get_physical_Pmuk(self):
        '''
            gets the anasotropycal physical powe spectrum given the best fit parameters
        '''
        Phpar,bbpar=self.GetParClass(self.best_fit_params,self.space)
        P_mu_k_Ph=Models.P_mu_k_parametric(Phpar,self.kh, self.Pk_l,self.Pk_now)
      
        return P_mu_k_Ph
        
    def get_BB_Pmuk(self):
        '''
            gets the anasotropycal  power spectrum corresponding to the broad band terms
        '''
        Phpar,bbpar=self.GetParClass(self.best_fit_params,self.space)
        
        elles=np.array([0,2,4])
        sk,t=hankl.P2xi(self.kh, 1, l=0, lowring=True)
        kh_test,Pk_tet=hankl.xi2P(sk, 1, l=0, lowring=True)

        xi_elle_BB=self.BroadBand(sk,bbpar)
        Pk_elle_BB=np.zeros((3,len(kh_test)))

        norm=0.0015#norm rappresenting the value of xi at r=rref
        rref=80.

        smax=np.min(np.array([np.max(self.s),self.fit_range[1]]))
        smin=np.max(np.array([np.min(self.s),self.fit_range[0]]))
        for l in range (0,3):
            Pk_l=np.zeros(len(kh_test))
            xi_l=np.zeros(len(sk))
            for i in range (-2,3):
                bli=bbpar['b'+str(elles[l])+str(i)]
                xi_li=bli*sk**(-i)*norm*rref**(i)*np.heaviside(smax-sk,1)*np.heaviside(sk-smin,1)
                kh_test,Pk_test=hankl.xi2P(sk, xi_li, l=elles[l], lowring=True)
               
                Pk_l+=Pk_test.real
                # xi_l+=xi_li
            Pk_elle_BB[l]=Pk_l
          
            if l==0:
                Pk_0=interp1d(kh_test,Pk_elle_BB[l],fill_value='extrapolate')
            else:
                if l==1: Pk_2=interp1d(kh_test,Pk_elle_BB[l],fill_value='extrapolate')
                else: Pk_4=interp1d(kh_test,Pk_elle_BB[l],fill_value='extrapolate')
        #resum the multipoles
        Pk_mu_k_BB= lambda mu,k: Pk_0(k)*Models.legendre(0)(mu)+Pk_2(k)*Models.legendre(2)(mu)+Pk_4(k)*Models.legendre(4)(mu)
        
        return Pk_mu_k_BB
        
    def BuildITCovariance(self,s,kh, p_mu_k):
        
        '''
          constructs covariance matrix from BF 2PCF
          s (array): data array separation
          kh (array): at which the power spectrum is evaluated
          p_mu_k (func): best fit anisotropic power spectrum
          returns: xy,yx,cov,corr
        '''
           
    
        p_k_mu=lambda k,mu:p_mu_k(mu,k)
        
        #set output bins

        Dr=s[1]-s[0]
        rad_edges = np.linspace(min(s),max(s)+Dr,len(s)+1)+0.5*Dr
       
        n_bins_r = len(s)
        l_list = [0, 2, 4] #multiple list
        n_data_points = n_bins_r * len(l_list)
        
        xi_covariance = TwoPointGaussianCovariance(rad_edges, l_list, deg=51, min_k = 1.e-4, max_k =  99,deg_k=10)
        cov = xi_covariance(p_k_mu, self.volume, self.number_density)
     
        std = np.sqrt(np.diagonal(cov))
        corr = cov / np.outer(std, std)
        sconc=np.concatenate((s,s,s))
        xy, yx = np.meshgrid(sconc, sconc,indexing='ij')
        
        return xy,yx,cov,corr
        
        
    def show tests():
        '''
            show the goodness of the procedure
            - plotBF: overplots  the results of the best fit on top of the data
            - Plot_CovModel_test: plots the hankle of the fiducial Pk used to build BF covaria
            

        '''
        xi_ell_BF=self.GetModelBF()
        print('best fit model ...')
        self.plotBF(self.s,self.xi_ell_data,self.covTH,xi_ell_BF)
        print('hankle transform of fiducial Power-spectrum used to build the covariance ...')
        self.Plot_CovModel_test():
        print('comparison between theoretical and best fit covariance ...')
        self.PlotCovComparison(self.covTH,self.corrBF)
        return
        
    def plotBF(self,s,xi_ell_data,cov,xi_bf):
        '''
            overplots  the results of the best fit on top of the data
            s: separation array
            xi_ell_data: 2pcf multipoles
            cov: covariance used for the fit
            xi_bf: best fit model for 2PCF multipoles
            
        '''
        
        std=np.sqrt(np.diag(cov)).reshape(3,len(s))
        
        color=['#009ffd','#ffa400']
       
        ylabels=['$r^2\\xi_0~(h^{-1}\mathrm{Mpc})$','$r^2\\xi_1~(h^{-1}\mathrm{Mpc})$']

        labels=['$r^2\\xi_0$','$r^2\\xi_2$']
        labelsM=['$r^2\\xi^\mathrm{BF}_\mathrm{0}$','$r^2\\xi^\mathrm{BF}_\mathrm{2}$']
        alph=0.2
        ft=12
        figure, axs = plt.subplots(nrows=2, ncols=1, sharex='col', sharey='row',figsize=(6, 4))


        smin=0.01
        smax=200
        for m in range (0,2):
            axs[m].set_ylabel(ylabels[m],fontsize=ft)
            axs[m].axhline(0,ls=':',lw=.5,c='k')
            axs[m].tick_params(labelsize=ft-3)

            axs[m].plot(s,xi_ell_data[m]*s**2,c=color[m],alpha=1,lw=1.5,label=labels[m])
            axs[m].fill_between(s,(xi_ell_data[m]-std[m])*s**2,\
                                        (xi_ell_data[m]+std[m])*s**2,color=color[m],alpha=0.1)

                
            m=0
            axs[m].plot(s[(s<smax)&(s>smin)],xi_bf[m][(s<smax)&(s>smin)]\
                         *s[(s<smax)&(s>smin)]**2,c='k',alpha=1,lw=0.8,ls='--',label=labelsM[m])
            m=1
            axs[m].plot(s[(s<smax)&(s>smin)],xi_bf[m][(s<smax)&(s>smin)]\
                         *s[(s<smax)&(s>smin)]**2,c='k',alpha=1,lw=0.8,ls='--')


        axs[0].legend(loc="upper right",fontsize=ft-2,frameon=False)
        axs[0].set_xlabel('$r~(h^{-1}\mathrm{Mpc})$',fontsize=ft)
        plt.tight_layout()
        figure.subplots_adjust(hspace=0)
        
        return
        
        def Plot_CovModel_test(self):
            '''
                plots the hankle of the fiducial Pk used to build BF covariance
            '''
    
            rtemp, xi_mu_r=self.ximur_ext(self.kh,self.P_mu_k_all)
            par_temp={'alpha_par':1.,'alpha_perp':1}
            xi_l_ref=T.xiell(self.s,par_temp,xi_mu_r)
            self.plotBF(self.s,self.xi_ell_data,self.cov,xi_l_ref)
        
        return
        
    def PlotCovComparison(self,xicov_th,xicov_BF):
        '''
            plots comparioson between theoretical covariance and best fit covatiance
            xicov_th: theoretical covariance
            xicov_BF: best fit covatiance
        '''
        ft=12
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex='col', sharey='row',figsize=(5,7))
        colors=['#21b0fe','#fe218b','#fed700']
        labels=['mock','th','it']

        yaxis=['$s^2Cov_{ii,0}~(\mathrm{Mpc}h^{-1})^2$','$s^2Cov_{ii,2}~(\mathrm{Mpc}h^{-1})^2$',\
        '$s^2Cov_{ii,4}~(\mathrm{Mpc}h^{-1})^2$']

        diag_it=np.diag(xicov_BF)
        diag_th=np.diag(xicov_th)
        subarray_size = len(diag_it) // 3

       
        errxi_th=np.zeros((3,subarray_size))
        errxi_it=np.zeros((3,subarray_size))

        errxi_th[0]=diag_th[:subarray_size]
        errxi_th[1]=diag_th[subarray_size:2*subarray_size]
        errxi_th[2]=diag_th[2*subarray_size:]
        errxi_it[0]=diag_it[:subarray_size]
        errxi_it[1]=diag_it[subarray_size:2*subarray_size]
        errxi_it[2]=diag_it[2*subarray_size:]
        s=np.linspace(0,200,len(errxi_th[0]))


        for i in range (0,3):
            axs[i].plot(s,s**2*errxi_th[i],ls='--',color=colors[1],label=labels[1])
            axs[i].plot(s,s**2*errxi_it[i],ls='--',color=colors[2],label=labels[2])
            axs[i].set_ylabel(yaxis[i],fontsize=ft)
            axs[i].tick_params(labelsize=ft)
        axs[0].legend(loc="upper left",fontsize=ft,frameon=False)
        axs[2].set_xlabel('$s~(\mathrm{Mpc}h^{-1})$',fontsize=ft)

        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)

        plt.show()
        
        fig, axs = plt.subplots(1, 2,sharex=True, sharey=True,figsize=(10,6))
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)
        xicorr_th = get_correlation_matrix(xicov_th)
        xicorr_it = get_correlation_matrix(xicov_BF)
        im1 = axs[0].imshow(xicorr_th, cmap='coolwarm',vmin=-1,vmax=1,origin='lower')
        im2 = axs[1].imshow(xicorr_it, cmap='coolwarm',vmin=-1,vmax=1,origin='lower')
     

        axs[0].set_title('Th')
        axs[1].set_title('It')


        # Create a common colorbar for both subplots
        cbar = fig.colorbar(im1, ax=axs, shrink=0.6)
        cbar.set_label('Corr')

        plt.show()
        return
            
            
    
