from sklearn.base import BaseEstimator, TransformerMixin
from PyFNRI.snplogit.snp_allele_genotype_freq import *
import numpy as np

class AlleleTransformer(BaseEstimator, TransformerMixin, snp_frequency_calc):
    
    def minor_genotype(self, X):
        x = []
        for v in X:
            x.append(X[v].value_counts().idxmin())
        return x

    def major_genotype(self, X):
        x = []
        for v in X:
            x.append(X[v].value_counts().idxmax())
        return x 

    def allele_freqs(self,X):
        Afreq = []
        for x in X.columns:
            Gfreq = self.calculate_genotype_freq(X,x)
            Afreq.append(self.calculate_allele_freq(Gfreq))
        return Afreq
     
    def fit(self, X, y=None):
        self.Afreq = self.allele_freqs(X)
        self.major = []
        self.minor = []
        for Afreq in self.Afreq:
            r = list(Afreq.values())
            k = list(Afreq.keys())
            if Afreq[k[0]] >= Afreq[k[1]]:
                self.major.append(k[0]+"/"+k[0])
                self.minor.append(k[1]+"/"+k[1])  
            else:
                self.major.append(k[1]+"/"+k[1])
                self.minor.append(k[0]+"/"+k[0])  
        return self

    def transform(self, X, y=None,dominant=False,recessive=False):
        if not dominant and not recessive:
            X_enc = (np.array(X != self.minor)*np.array(X != self.major)).astype(np.int64)
            X_enc = X_enc + np.array(X == self.minor)*2 
        elif dominant:
            X_enc = np.array(X != self.minor).astype(np.int64)           
        else:
            X_enc = np.array(X != self.major).astype(np.int64)          
        return X_enc
    
    def fit_transform(self, X, y=None, dominant=False,recessive=False):
        return self.fit(X).transform(X)
    
class NoTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X
    
    def fit_transform(self, X, y=None):
        return X