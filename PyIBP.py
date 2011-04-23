"""
PyIBP

Implements fast Gibbs sampling for the linear-Gaussian
infinite latent feature model (IBP).

Copyright (C) 2009 David Andrzejewski (andrzeje@cs.wisc.edu)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as NP
import numpy.random as NR
import scipy.stats as SPST
import scipy.special as SPS

# We will be taking log(0) = -Inf, so turn off this warning
NP.seterr(divide='ignore')

class PyIBP(object):
    """
    Implements fast Gibbs sampling for the linear-Gaussian
    infinite latent feature model (IBP)
    """

    #
    # Initialization methods
    #

    def __init__(self,data,alpha,sigma_x,sigma_a,
                 missing=None,useV=False,initZV=None):
        """ 
        data = NxD NumPy data matrix (should be centered)

        alpha = Fixed IBP hyperparam for OR (init,a,b) tuple where
        (a,b) are Gamma hyperprior shape and rate/inverse scale
        sigma_x = Fixed noise std OR (init,a,b) tuple (same as alpha)
        sigma_a = Fixed weight std OR (init,a,b) tuple (same as alpha)
        
        OPTIONAL ARGS
        missing = boolean/binary 'missing data' mask (1=missing entry)
        useV = Are we using real-valued latent features? (default binary)
        initZV = Optional initial state for the latent         
        """
        # Data matrix
        self.X = data
        (self.N,self.D) = data.shape
        # IBP hyperparameter
        if(type(alpha) == tuple):
            (self.alpha,self.alpha_a,self.alpha_b) = alpha
        else:
            (self.alpha,self.alpha_a,self.alpha_b) = (alpha,None,None)
        # Noise variance hyperparameter
        if(type(sigma_x) == tuple):
            (self.sigma_x,self.sigma_xa,self.sigma_xb) = sigma_x
        else:
            (self.sigma_x,self.sigma_xa,self.sigma_xb) = (sigma_x,None,None)
        # Weight variance hyperparameter
        if(type(sigma_a) == tuple):
            (self.sigma_a,self.sigma_aa,self.sigma_ab) = sigma_a
        else:
            (self.sigma_a,self.sigma_aa,self.sigma_ab) = (sigma_a,None,None)
        # Are we using weighted latent features?
        self.useV = useV                            
        # Do we have user-supplied initial latent feature values?
        if(initZV == None):
            # Initialze Z from IBP(alpha)
            self.initZ()
            # Initialize V from N(0,1) if necessary
            if(self.useV):
                self.initV()
        else:
            self.ZV = initZV
            self.K = self.ZV.shape[1]
            self.m = (self.ZV != 0).astype(NP.int).sum(axis=0)
        # Sample missing data entries if necessary
        self.missing = missing
        if(missing != None):
            self.sampleX()

    def initV(self):
        """ Init latent feature weights V accoring to N(0,1) """        
        for (i,k) in zip(*self.ZV.nonzero()):
            self.ZV[i,k] = NR.normal(0,1)

    def initZ(self):
        """ Init latent features Z according to IBP(alpha) """
        Z = NP.ones((0,0))
        for i in range(1,self.N+1):
            # Sample existing features
            zi = (NR.uniform(0,1,(1,Z.shape[1])) <
                  (Z.sum(axis=0).astype(NP.float) / i))
            # Sample new features
            knew = SPST.poisson.rvs(self.alpha / i)
            zi = NP.hstack((zi,NP.ones((1,knew))))
            # Add to Z matrix
            Z = NP.hstack((Z,NP.zeros((Z.shape[0],knew))))
            Z = NP.vstack((Z,zi))
        self.ZV = Z
        self.K = self.ZV.shape[1]
        # Calculate initial feature counts
        self.m = (self.ZV != 0).astype(NP.int).sum(axis=0)

    #
    # Convenient external methods
    #

    def fullSample(self):
        """ Do all applicable samples """
        self.sampleZ()
        if(self.missing != None):
            self.sampleX()
        if(self.alpha_a != None):
            self.sampleAlpha()
        if(self.sigma_xa != None):
            self.sampleSigma()

    def logLike(self):
        """
        Calculate log-likelihood P(X,Z)
        (or P(X,Z,V) if applicable)
        """
        liketerm = self.logPX(self.calcM(self.ZV),self.ZV)
        ibpterm = self.logIBP()    
        if(self.useV):
            vterm = self.logPV()
            return liketerm+ibpterm+vterm
        else:
            return liketerm+ibpterm

    def weights(self):
        """ Return E[A|X,Z] """
        return self.postA(self.X,self.ZV)[0]

    #
    # Actual sampling methods
    #

    def sampleV(self,k,meanA,covarA,xi,zi):
        """ Slice sampling for feature weight V """
        oldv = zi[0,k]
        # Log-posterior of current value
        curlp = self.vLogPost(k,zi[0,k],meanA,covarA,xi,zi)
        # Vertically sample beneath this value
        curval = self.logUnif(curlp)
        # Initial sample from horizontal slice
        (left,right) = self.makeInterval(curval,k,zi[0,k],
                                         meanA,covarA,xi,zi)
        newv = NR.uniform(left,right)
        newval = self.vLogPost(k,newv,meanA,covarA,xi,zi)
        # Repeat until valid sample obtained
        while(newval <= curval):
            if(newv < zi[0,k]):
                left = newv
            else:
                right = newv
            newv = NR.uniform(left,right)
            newval = self.vLogPost(k,newv,meanA,covarA,xi,zi)
        return newv        

    def makeInterval(self,u,k,v,meanA,covarA,xi,zi):
        """ Get horizontal slice sampling interval """
        w = .25
        (left,right) = (v-w,v+w)
        (leftval,rightval) = (self.vLogPost(k,left,meanA,covarA,xi,zi),
                              self.vLogPost(k,right,meanA,covarA,xi,zi))
        while(leftval > u):
            left -= w 
            leftval = self.vLogPost(k,left,meanA,covarA,xi,zi)
        while(rightval > u):
            right += w
            rightval = self.vLogPost(k,right,meanA,covarA,xi,zi) 
        return (left,right)

    def vLogPost(self,k,v,meanA,covarA,xi,zi):
        """ For a given V, calculate the log-posterior """        
        oldv = zi[0,k]
        zi[0,k] = v
        (meanLike,covarLike) = self.likeXi(zi,meanA,covarA)
        logprior = -0.5*(v**2) - 0.5*NP.log(2*NP.pi)
        loglike = self.logPxi(meanLike,covarLike,xi)
        # Restore previous value and return result
        zi[0,k] = oldv
        return logprior + loglike

    def sampleSigma(self):
        """ Sample feature/noise variances """
        # Posterior over feature weights A
        (meanA,covarA) = self.postA(self.X,self.ZV)
        # sigma_x
        vars = NP.dot(self.ZV,NP.dot(covarA,self.ZV.T)).diagonal()
        var_x = (NP.power(self.X - NP.dot(self.ZV,meanA), 2)).sum()
        var_x += self.D * vars.sum()
        n = float(self.N * self.D)
        postShape = self.sigma_xa + n/2
        postScale = float(1) / (self.sigma_xb + var_x/2)
        tau_x = SPST.gamma.rvs(postShape,scale=postScale)
        self.sigma_x = NP.sqrt(float(1)/tau_x)
        # sigma_a
        var_a = covarA.trace() * self.D + NP.power(meanA,2).sum()
        n = float(self.K * self.D)
        postShape = self.sigma_aa + n/2
        postScale = float(1) / (self.sigma_ab + var_a/2)
        tau_a = SPST.gamma.rvs(postShape,scale=postScale)        
        self.sigma_a = NP.sqrt(float(1)/tau_a)
        if(self.sigma_a > 100):
            pdb.set_trace()
        
    def sampleAlpha(self):
        """ Sample alpha from conjugate posterior """
        postShape = self.alpha_a + self.m.sum()
        postScale = float(1) / (self.alpha_b + self.N)
        self.alpha = SPST.gamma.rvs(postShape,scale=postScale)

    def sampleX(self):
        """ Take single sample missing data entries in X """
        # Calculate posterior mean/covar --> info
        (meanA,covarA) = self.postA(self.X,self.ZV) 
        (infoA,hA) = self.toInfo(meanA,covarA)
        # Find missing observations
        xis = NP.nonzero(self.missing.max(axis=1))[0]
        for i in xis:
            # Get (z,x) for this data point
            (zi,xi) = (NP.reshape(self.ZV[i,:],(1,self.K)),
                       NP.reshape(self.X[i,:],(1,self.D)))
            # Remove this observation
            infoA_i = self.updateInfo(infoA,zi,-1)
            hA_i = self.updateH(hA,zi,xi,-1)
            # Convert back to mean/covar
            (meanA_i,covarA_i) = self.fromInfo(infoA_i,hA_i)            
            # Resample xi
            (meanXi,covarXi) = self.likeXi(zi,meanA_i,covarA_i)                
            newxi = NR.normal(meanXi,NP.sqrt(covarXi))
            # Replace missing features
            ks = NP.nonzero(self.missing[i,:])[0]
            self.X[i,ks] = newxi[0][ks]

    def sampleZ(self):
        """ Take single sample of latent features Z """
        # for each data point
        order = NR.permutation(self.N)
        for (ctr,i) in enumerate(order):
            # Initially, and later occasionally,
            # re-cacluate information directly
            if(ctr % 5 == 0):
                try:
                    (meanA,covarA) = self.postA(self.X,self.ZV)
                    (infoA,hA) = self.toInfo(meanA,covarA)
                except Exception,e:
                    pdb.set_trace()                
            # Get (z,x) for this data point
            (zi,xi) = (NP.reshape(self.ZV[i,:],(1,self.K)),
                       NP.reshape(self.X[i,:],(1,self.D)))
            # Remove this point from information
            infoA = self.updateInfo(infoA,zi,-1)
            hA = self.updateH(hA,zi,xi,-1)
            # Convert back to mean/covar
            (meanA,covarA) = self.fromInfo(infoA,hA)            
            # Remove this data point from feature cts
            newcts = self.m - (self.ZV[i,:] != 0).astype(NP.int)
            # Log collapsed Beta-Bernoulli terms
            lpz1 = NP.log(newcts)
            lpz0 = NP.log(self.N - newcts)
            # Find all singleton features
            singletons = [ki for ki in range(self.K) if
                          self.ZV[i,ki] != 0 and self.m[ki] == 1]
            nonsingletons = [ki for ki in range(self.K) if
                             ki not in singletons]
            # Sample for each non-singleton feature
            #
            for k in nonsingletons:
                oldz = zi[0,k]
                # z=0 case
                lp0 = lpz0[k]
                zi[0,k] = 0
                (meanLike,covarLike) = self.likeXi(zi,meanA,covarA)
                lp0 += self.logPxi(meanLike,covarLike,xi)
                # z=1 case
                lp1 = lpz1[k]
                if(self.useV):
                    if(oldz != 0):
                        # Use current V value
                        zi[0,k] = oldz
                        (meanLike,covarLike) = self.likeXi(zi,meanA,covarA)
                        lp1 += self.logPxi(meanLike,covarLike,xi)
                    else:
                        # Sample V values from the prior to 
                        # numerically collapse/integrate
                        nvs = 5
                        lps = NP.zeros((nvs,))
                        for vs in range(nvs):
                            zi[0,k] = NR.normal(0,1)
                            (meanLike,covarLike) = self.likeXi(zi,meanA,covarA)
                            lps[vs] = self.logPxi(meanLike,covarLike,xi)
                        lp1 += lps.mean()
                else:
                    zi[0,k] = 1
                    (meanLike,covarLike) = self.likeXi(zi,meanA,covarA)
                    lp1 += self.logPxi(meanLike,covarLike,xi)
                # Sample Z, update feature counts
                if(not self.logBern(lp0,lp1)):
                    zi[0,k] = 0
                    if(oldz != 0):
                        self.m[k] -= 1
                else:
                    if(oldz == 0):
                        self.m[k] += 1
                    if(self.useV):
                        # Slice sample V from posterior if necessary
                        zi[0,k] = self.sampleV(k,meanA,covarA,xi,zi)
                if(self.m[k] != ((self.ZV[:,k] != 0 ).astype(NP.int)).sum()):
                    pdb.set_trace()
                if(self.m[k]>self.N):
                    pdb.set_trace()
            #
            # Sample singleton/new features using the
            # Metropolis-Hastings step described in Meeds et al
            #
            kold = len(singletons)
            # Sample from the Metropolis proposal
            knew = SPST.poisson.rvs(self.alpha / self.N)
            if(self.useV):
                vnew = NP.array([NR.normal(0,1) for k in range(knew)])
            # Net difference in number of singleton features
            netdiff = knew - kold
            # Contribution of singleton features to variance in x
            if(self.useV):
                prevcontrib = NP.power(zi[0,singletons],2).sum()
                newcontrib = NP.power(vnew,2).sum()
                weightdiff = newcontrib - prevcontrib
            else:
                weightdiff = knew - kold
            # Calculate the loglikelihoods
            (meanLike,covarLike) = self.likeXi(zi,meanA,covarA)
            lpold = self.logPxi(meanLike,covarLike,xi)
            lpnew = self.logPxi(meanLike,
                                covarLike + weightdiff*self.sigma_a**2,
                                xi)
            lpaccept = min(0.0, lpnew-lpold)
            lpreject = NP.log(max(1.0 - NP.exp(lpaccept), 1e-100))
            if(self.logBern(lpreject,lpaccept)):
                # Accept the Metropolis-Hastings proposal
                if(netdiff > 0):
                    # We're adding features, update ZV
                    self.ZV = NP.append(self.ZV,NP.zeros((self.N,netdiff)),1)
                    if(self.useV):
                        prevNumSingletons = len(singletons)
                        self.ZV[i,singletons] = vnew[:prevNumSingletons]
                        self.ZV[i,self.K:] = vnew[prevNumSingletons:]
                    else:
                        self.ZV[i,self.K:] = 1
                    # Update feature counts m
                    self.m = NP.append(self.m,NP.ones(netdiff),0)
                    # Append information matrix with 1/sigmaa^2 diag
                    infoA = NP.vstack((infoA,NP.zeros((netdiff,self.K))))
                    infoA = NP.hstack((infoA,
                                       NP.zeros((netdiff+self.K,netdiff))))
                    infoappend = (1 / self.sigma_a**2) * NP.eye(netdiff)
                    infoA[self.K:(self.K+netdiff),
                          self.K:(self.K+netdiff)] = infoappend
                    # only need to resize (expand) hA
                    hA = NP.vstack((hA,NP.zeros((netdiff,self.D))))
                    # Note that the other effects of new latent features 
                    # on (infoA,hA) (ie, the zi terms) will be counted when 
                    # this zi is added back in                    
                    self.K += netdiff
                elif(netdiff < 0):
                    # We're removing features, update ZV
                    if(self.useV):
                        self.ZV[i,singletons[(-1*netdiff):]] = vnew
                    dead = [ki for ki in singletons[:(-1*netdiff)]]
                    self.K -= len(dead)
                    self.ZV = NP.delete(self.ZV,dead,axis=1)
                    self.m = NP.delete(self.m,dead)
                    # Easy to do this b/c these features did not
                    # occur in any other data points anyways...
                    infoA = NP.delete(infoA,dead,axis=0)
                    infoA = NP.delete(infoA,dead,axis=1)
                    hA = NP.delete(hA,dead,axis=0)                    
                else:
                    # net difference is actually zero, just replace
                    # the latent weights of existing singletons
                    # (if applicable)
                    if(self.useV):
                        self.ZV[i,singletons] = vnew
            # Add this point back into information
            #
            zi = NP.reshape(self.ZV[i,:],(1,self.K))
            infoA = self.updateInfo(infoA,zi,1)
            hA = self.updateH(hA,zi,xi,1)

    #
    # Output and reporting
    # 

    def sampleReport(self,sampleidx):
        """ Print IBP sample status """
        print 'iter %d' % sampleidx
        print '\tcollapsed loglike = %f' % self.logLike()
        print '\tK = %d' % self.K
        print '\talpha = %f' % self.alpha
        print '\tsigma_x = %f' % self.sigma_x
        print '\tsigma_a = %f' % self.sigma_a
                
    def weightReport(self,trueWeights=None,round=False):
        """ Print learned weights (vs ground truth if available) """
        if(trueWeights != None):
            print '\nTrue weights (A)'
            print str(trueWeights)
        print '\nLearned weights (A)'
        # Print rounded or actual weights?
        if(round):
            print str(self.weights().astype(NP.int))
        else:
            print NP.array_str(self.weights(),precision=2,suppress_small=True)
        print ''
        # Print V matrix if applicable
        if(self.useV):
            print '\nLatent feature weights (V)'
            print NP.array_str(self.ZV,precision=2)
            print ''
        # Print 'popularity' of latent features
        print '\nLatent feature counts (m)'
        print NP.array_str(self.m)

    #
    # Bookkeeping and calculation methods
    #

    def logPV(self):
        """ Log-likelihood of real-valued latent features V """
        lpv = -0.5*NP.power(self.ZV,2).sum()
        return lpv - len(self.ZV.nonzero()[0]) * 0.5 * NP.log(2*NP.pi)

    def logIBP(self):
        """ Calculate IBP prior contribution log P(Z|alpha) """
        (N,K) = self.ZV.shape
        # Need to find all unique K 'histories'
        Z = (self.ZV != 0).astype(NP.int)
        Khs = {}
        for k in range(K):
            history = tuple(Z[:,k])
            Khs[history] = Khs.get(history,0) + 1
        logp = 0
        logp += self.K * NP.log(self.alpha)
        for Kh in Khs.values():
            logp -= self.logFact(Kh)
        logp -= self.alpha * sum([float(1) / i for i in range(1,N+1)])
        for k in range(K):
            logp += self.logFact(N-self.m[k]) + self.logFact(self.m[k]-1)
            logp -= self.logFact(N)        
        if(logp==float('inf')):
            pdb.set_trace()
        return logp    
    
    def postA(self,X,Z):
        """ Mean/covar of posterior over weights A """
        M = self.calcM(Z)
        meanA = NP.dot(M,NP.dot(Z.T,X))
        covarA = self.sigma_x**2 * self.calcM(Z)
        return (meanA,covarA)

    def calcM(self,Z):
        """ Calculate M = (Z' * Z - (sigmax^2) / (sigmaa^2) * I)^-1 """
        return NP.linalg.inv(NP.dot(Z.T,Z) + (self.sigma_x**2) 
                             / (self.sigma_a**2) * NP.eye(self.K))

    def logPX(self,M,Z):
        """ Calculate collapsed log likelihood of data"""
        lp = -0.5 * self.N * self.D * NP.log(2*NP.pi)
        lp -= (self.N - self.K) * self.D * NP.log(self.sigma_x)
        lp -= self.K * self.D * NP.log(self.sigma_a)
        lp -= 0.5 * self.D * NP.log(NP.linalg.det(NP.linalg.inv(M)))
        iminzmz = NP.eye(self.N) - NP.dot(Z,NP.dot(M,Z.T))
        lp -= (0.5 / (self.sigma_x**2)) * NP.trace(
            NP.dot(self.X.T,NP.dot(iminzmz,self.X)))
        return lp

    def likeXi(self,zi,meanA,covarA):
        """ Mean/covar of xi given posterior over A """
        meanXi = NP.dot(zi,meanA)
        covarXi = NP.dot(zi,NP.dot(covarA,zi.T)) + self.sigma_x**2
        return (meanXi,covarXi)

    def updateInfo(self,infoA,zi,addrm):
        """ Add/remove data i to/from information """
        return infoA + addrm * ((1/self.sigma_x**2) * NP.dot(zi.T,zi))

    def updateH(self,hA,zi,xi,addrm):
        """ Add/remove data i to/from h"""
        return hA + addrm * ((1/self.sigma_x**2) * NP.dot(zi.T,xi))
    
    #
    # Pure functions (these don't use state or additional params)
    #

    @staticmethod
    def logFact(n):
        return SPS.gammaln(n+1)
    
    @staticmethod
    def fromInfo(infoA,hA):
        """ Calculate mean/covar from information """
        covarA = NP.linalg.inv(infoA)
        meanA = NP.dot(covarA,hA)
        return (meanA,covarA)

    @staticmethod
    def toInfo(meanA,covarA):
        """ Calculate information from mean/covar """
        infoA = NP.linalg.inv(covarA)
        hA = NP.dot(infoA,meanA)        
        return (infoA,hA)

    @staticmethod
    def logUnif(v):
        """ 
        Sample uniformly from [0, exp(v)] in the log-domain
        (derive via transform f(x)=log(x) and some calculus...)
        """
        return v + NP.log(NR.uniform(0,1))

    @staticmethod
    def logBern(lp0,lp1):
        """ Bernoulli sample given log(p0) and log(p1) """
        p1 = 1 / (1+NP.exp(lp0-lp1))
        return (p1 > NR.uniform(0,1))

    @staticmethod
    def logPxi(meanLike,covarLike,xi):
        """
        Calculate log-likelihood of a single xi, given its
        mean/covar after collapsing P(A | X_{-i}, Z)
        """
        D = float(xi.shape[1])
        ll = -(D / 2) * NP.log(covarLike)
        ll -= (1 / (2*covarLike)) * NP.power(xi-meanLike,2).sum()
        return ll
    
    @staticmethod
    def centerData(data):
        return data - PyIBP.featMeans(data)

    @staticmethod
    def featMeans(data,missing=None):
        """ Replace all columns (features) with their means """
        (N,D) = data.shape
        if(missing == None):
            return NP.tile(data.mean(axis=0),(N,1))
        else:        
            # Sanity check on 'missing' mask
            # (ensure no totally missing data or features)
            assert(all(missing.sum(axis=0) < N) and
                   all(missing.sum(axis=1) < D))
            # Calculate column means without using the missing data
            censored = data * (NP.ones((N,D)) - missing)
            censoredmeans = censored.sum(axis=0) / (N-missing.sum(axis=0))
            return NP.tile(censoredmeans,(N,1))
