import numpy as np
import pandas as pd
import warnings
import random
from utils import cdot, safe_cast, to_df, set_labels, symmetrise, check_symmetry
from sklearn.covariance import ShrunkCovariance


class FMCA:
    def __init__(self, n_components="auto", cov_estimator=ShrunkCovariance, cov_params=None, max_pos=True, as_df=True):
        """Initialise the FMCA constructor and determine the appropriate form of the cov_estimator function.
        Params:
        n_components: Determines the number of eigen-vectors output by transform. If set to 'auto' n_components will be set to the square root of the number of features of the input data.
        cov_estimator: Determines the algorithm used to estimate the covariance matrix of the data. Any estimator passed must implment a fit function and posess a covariance_ attribute. If set to None, the standard maximum likelihood estimator will be used, but you shouldn't do this because the standard ML estimator is very bad.
        cov_params: Use this to pass a dictionary of any additional parameters you may wish to pass the covariance estimator.
        max_pos: Maximises the variance of the positive samples divided by the variance of the negative samples if true.
        as_df: If true, passes the decomposed dataset as a df. Otherwise, passes it as a numpy matrix. """
    
        self.nc = n_components
        self.cv = cov_estimator
        self.cv_prms = cov_params
        self.mp = max_pos
        self.ad = as_df
        #Set cv_mat to None just in case symmetrise gets called before the covariance matrix is called for whatever reason
        self.cv_mat = None
        if self.cv is not None:
            if 'fit' not in dir(self.cv):
                warnings.warn("Covariance Estimator must support 'fit' method. Resorting to default estimator.")
                self.cv = ShrunkCovariance
            else:
                if self.cv_prms is not None:
                    self.cv_est = self.cv(**self.cv_prms)
                else:
                    self.cv_est = self.cv()
                if 'covariance_' not in dir(self.cv_est):
                    warnings.warn("Supplied covariance estimator does not posses covariance_ attribute. Resorting to default estimator")
                    self.cv = None
                else:
                    self.cov_fit = lambda x: self.cv_est.fit(safe_cast(x)).covariance_
        if self.cv is None:
            self.cov_fit = lambda x: np.cov(safe_cast(x))
                    
    def fit(self, X, y):
        """Here we determine the eigenvalues and eigenvectors of the product of the covariance matrix of 
        one class with the precision matrix of the second and store the resulting product in the spec
        attribute"""
        
        #We convert the data to a pandas dataframe just to make the class split simple
        if type(X) != pd.DataFrame:
            #The purpose of casting the data to a numpy array here is to ensure that the data structures
            #passed to this function are of a suitable format. Suitability being defined as having the
            #basic underlying structure of an array. Probably not necessary but there may be cases where
            #some unsuitable ds gets passed to fit and in those cases you want to exit asap with a clear
            #error message 
            X = safe_cast(X)
            y = safe_cast(y)
            y = set_labels(y)
            cols, df = to_df(X, y)
        else:
            cols = X.columns
            y  = safe_cast(y)
            y = set_labels(y)
            y = pd.DataFrame(y, columns=["Class"])
            df = pd.concat([X, y], axis=1)
        df_neg = df[df["Class"] == 0]
        df_pos = df[df["Class"] == 1]
        cv_neg = self.cov_fit(df_neg[cols])
        cv_pos = self.cov_fit(df_pos[cols])
        if self.mp:
            #This branch executes if you want to maximise the ratio of positive to negative class samples
            prod = symmetrise(np.dot(cv_pos, np.linalg.inv(cv_neg)))
        else:
            #This branch executes if you want to maximise the ratio of negative to positive class samples
            prod = symmetrise(np.dot(cv_neg, np.linalg.inv(cv_pos)))
        U, s, V = np.linalg.svd(prod)
        #Eigenvectors are given columnwise so we need to transpose here
        Ut = np.transpose(U)
        self.eigvals = s
        #Check for repeat eigenvalues
        if len(set(s)) != len(s):
            warnings.warn("Repeated eigenvalues found. Please check your data for collinearity.")
        spec = list(zip(s, Ut))
        spec.sort(key=lambda x:x[0], reverse=True)
        self.spec = spec
        self.eigmat = Ut
    

    def transform(self, X, y=None):
        """Rotate X into the F-maximal (sub)-basis. Preserve class labels in y if not None"""
        columns = []
        dct = {}
        vecs = {}
        if n_components == "auto":
            n_components = int(np.sqrt(len(X.columns)))
        for val in range(self.nc):
            st = "FMCA_" + str(val)
            columns.append(st)
            dct[st] = []
            vecs[st] = self.spec[val][1]
        Xt = safe_cast(X)
        for row in Xt:
            for col in columns:
                dct[col].append(cdot(row, vecs[col]))
        Xout = pd.DataFrame(dct)
        if y is not None and len(y) == len(Xt):
            yt = safe_cast(y)
            if self.ad:
                yout = pd.DataFrame(yt, columns=["Class"])
                return pd.concat([Xout, yout], axis=1)
            else:
                yout = pd.DataFrame(yt, columns=["Class"])
                return np.array(pd.concat([Xout, yout], axis=1))
        else:
            if self.ad:
                return Xout
                
            else:
                return np.array(Xout)

        
        
class WMCA:
    def __init__(self, n_components="auto", cov_estimator=ShrunkCovariance, cov_params=None, as_df=True):
        """Initialise the WMCA constructor and determine the appropriate form of the cov_estimator function.
        Params:
        n_components: Determines the number of eigen-vectors output by transform. If set to 'auto' n_components will be set to the square root of the number of features of the input data.
        cov_estimator: Determines the algorithm used to estimate the covariance matrix of the data. Any estimator passed must implment a fit function and posess a covariance_ attribute. If set to None, the standard maximum likelihood estimator will be used, but you shouldn't do this because the standard ML estimator is very bad.
        cov_params: Use this to pass a dictionary of any additional parameters you may wish to pass the covariance estimator.
        as_df: If true, passes the decomposed dataset as a df. Otherwise, passes it as a numpy matrix. """
    
        self.nc = n_components
        self.cv = cov_estimator
        self.cv_prms = cov_params
        self.ad = as_df
        #Set cv_mat to None just in case symmetrise gets called before the covariance matrix is called for whatever reason
        self.cv_mat = None
        if self.cv is not None:
            if 'fit' not in dir(self.cv):
                warnings.warn("Covariance Estimator must support 'fit' method. Resorting to default estimator.")
                self.cv = ShrunkCovariance
            else:
                if self.cv_prms is not None:
                    self.cv_est = self.cv(**self.cv_prms)
                else:
                    self.cv_est = self.cv()
                if 'covariance_' not in dir(self.cv_est):
                    warnings.warn("Supplied covariance estimator does not posses covariance_ attribute. Resorting to default estimator")
                    self.cv = None
                else:
                    self.cov_fit = lambda x: self.cv_est.fit(safe_cast(x)).covariance_
        if self.cv is None:
            self.cov_fit = lambda x: np.cov(safe_cast(x))
                    
    def fit(self, X, y):
        """Here we determine the eigenvalues and eigenvectors of the product of the covariance matrix of 
        one class with the precision matrix of the second and store the resulting product in the spec
        attribute"""
        
        #We convert the data to a pandas dataframe just to make the class split simple
        if type(X) != pd.DataFrame:
            #The purpose of casting the data to a numpy array here is to ensure that the data structures
            #passed to this function are of a suitable format. Suitability being defined as having the
            #basic underlying structure of an array. Probably not necessary but there may be cases where
            #some unsuitable ds gets passed to fit and in those cases you want to exit asap with a clear
            #error message 
            X = safe_cast(X)
            y = safe_cast(y)
            y = set_labels(y)
            cols, df = to_df(X, y)
        else:
            cols = X.columns
            y  = safe_cast(y)
            y = set_labels(y)
            y = pd.DataFrame(y, columns=["Class"])
            df = pd.concat([X, y], axis=1)
        df_neg = df[df["Class"] == 0]
        df_pos = df[df["Class"] == 1]
        mean_pos = np.diag(df_pos.mean())
        mean_neg = np.diag(df_neg.mean())
        mean_diff = (mean_pos - mean_neg)**2
        cv_neg = self.cov_fit(df_neg[cols]) / (np.sqrt(len(df_neg)))
        cv_pos = self.cov_fit(df_pos[cols]) / (np.sqrt(len(df_pos)))
        cv_tot = cv_neg**2 + cv_pos**2
        
        prod = symmetrise(np.dot(mean_diff, np.linalg.inv(cv_tot)))

        U, s, V = np.linalg.svd(prod)
        #Eigenvectors are given columnwise so we need to transpose here
        Ut = np.transpose(U)
        self.eigvals = s
        #Check for repeat eigenvalues
        if len(set(s)) != len(s):
            warnings.warn("Repeated eigenvalues found. Please check your data for collinearity.")
        spec = list(zip(s, Ut))
        spec.sort(key=lambda x:x[0], reverse=True)
        self.spec = spec
        self.eigmat = Ut
    

    def transform(self, X, y=None):
        """Rotate X into the Welch-maximal (sub)-basis. Preserve class labels in y if not None"""
        columns = []
        dct = {}
        vecs = {}
        if n_components == "auto":
            n_components = int(np.sqrt(len(X.columns)))
        for val in range(self.nc):
            st = "WMCA_" + str(val)
            columns.append(st)
            dct[st] = []
            vecs[st] = self.spec[val][1]
        Xt = safe_cast(X)
        for row in Xt:
            for col in columns:
                dct[col].append(cdot(row, vecs[col]))
        Xout = pd.DataFrame(dct)
        if y is not None and len(y) == len(Xt):
            yt = safe_cast(y)
            if self.ad:
                yout = pd.DataFrame(yt, columns=["Class"])
                return pd.concat([Xout, yout], axis=1)
            else:
                yout = pd.DataFrame(yt, columns=["Class"])
                return np.array(pd.concat([Xout, yout], axis=1))
        else:
            if self.ad:
                return Xout
                
            else:
                return np.array(Xout)

        

        
