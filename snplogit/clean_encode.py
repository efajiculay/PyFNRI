import re as rgx
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import pandas as pd
from scipy.stats import chi2_contingency
from PyFNRI.snplogit.snp_allele_genotype_freq import *
from PyFNRI.snplogit.snp_ordinal_transform import *

from joblib import Parallel, delayed
from tqdm import tqdm
import sys

class data_cleaning_encoding(snp_frequency_calc):
    
    def __init__(self,data):
        self.data = data
       
    def set_outcome(self, outcome):
        self.target = outcome
        self.data = self.data[self.data[outcome].notna()]
        return self

    def encode_outcome(self, thr): 
        self.data[self.target][self.data[self.target]<=thr] = 0
        self.data[self.target][self.data[self.target]>thr] = 1
        return self
        
    def filter_call_rate_less_than_pthr(self, pthr=10):
        "Filter data based on genotype call rate"
        table = self.data
        res_row = table.isnull().sum(axis=1)
        res_row = (100*res_row/table.shape[0])
        res_row.to_excel("percent_null_row.xlsx")

        res_col = table.isnull().sum(axis=0)
        res_col = (100*res_col/table.shape[0])
        res_col.to_excel("percent_null_column.xlsx")

        to_remove_col = list(res_col[res_col>pthr].keys())
        to_remove_row = list(res_row[res_row>pthr].keys())
        self.data = self.data.drop(to_remove_col,axis=1).drop(to_remove_row,axis=0)
        return self
        
    def feature_typing(self,exclude=["biocode"]):
        self.categorical_features = []
        self.snp_categorical = []
        self.numeric_features = []
        
        col_dtypes = self.data.dtypes
        for i, x in enumerate(self.data.columns):
            key = x.strip()
            if "unnamed" in key.lower():
                pass
            else:
                if rgx.search(r"rs\d+",key.lower()):
                    self.snp_categorical.append(key)
                elif x.strip() in exclude:
                    pass
                elif col_dtypes[i] == 'object':
                    self.categorical_features.append(key)
                else:
                    self.numeric_features.append(key)
        
        self.selected_features = self.categorical_features + self.snp_categorical + self.numeric_features
        return self
               
    def chi2_test_HWE(self,Gfreq):
        allele_freq = self.calculate_allele_freq(Gfreq)
        Efreq = self.expected_genotype_freq(allele_freq)
        x = [ Gfreq.loc[c].Frequency for c in Gfreq.index ]
        y = [ Efreq[c] for c in Gfreq.index ]
        return chi2_contingency([x,y])

    def filter_HWE(self,thr=0.05,normal=True):
        if normal:
            data = self.data[self.data[self.target]==0]
        else:
            data = self.data
        snp_features = set(self.snp_categorical)
        for x in snp_features:
            Gfreq = self.calculate_genotype_freq(data, x)
            chi2 = self.chi2_test_HWE(Gfreq)
            pval = chi2.pvalue
            if pval>=thr:
                data = data.drop(x,axis=1)
                snp_features = snp_features -{x}
                        
        self.snp_categorical = list(snp_features)
        self.selected_features = self.categorical_features + self.snp_categorical + self.numeric_features
        self.data = self.data[self.selected_features]
        return self
        
    def calculate_LD_r2(self,data_slice):
        col = data_slice.columns
        Gfreq1 = self.calculate_genotype_freq(data_slice,col[0])
        Gfreq2 = self.calculate_genotype_freq(data_slice,col[1])
        allele_freq1 = self.calculate_allele_freq(Gfreq1)
        allele_freq2 = self.calculate_allele_freq(Gfreq2)
        tot1 = sum(allele_freq1.values())
        tot2 = sum(allele_freq2.values())
        
        p1 = {}
        prod = 1
        for x in allele_freq1:
            p1[x] = allele_freq1[x]/tot1
            prod = prod*p1[x]
            
        p2 = {}
        for x in allele_freq2:
            p2[x] = allele_freq2[x]/tot2
            prod = prod*p2[x]

        haplo_freq_est = {}
        for x in allele_freq1:
            for y in allele_freq2:
                key = x+"_"+y
                haplo_freq_est[key] = p1[x]*p2[y]

        haplo_freq_obs = self.calc_haplo_freq_obs(data_slice)
        tot3 = sum(haplo_freq_obs.values()) 
        
        D = {}
        for g1 in allele_freq1:
            for g2 in allele_freq2:
                key = g1+"_"+g2
                D[key] = haplo_freq_obs[key]/tot3 - haplo_freq_est[key]

        numD = [D[x]**2 for x in D] 
        D2tol = max(numD)
        
        return D2tol/prod 
        
    def calc_LD_r2_all_pairs(self,nj=4,limit=None):
        data = self.data[self.data[self.target]==0]
        snp_features = self.snp_categorical
        if not limit:
            N = len(snp_features)
        else:
            N = 100
        
        def parallel_run_LD_r2(i):      
            LDR2_table = {}
            key1 = snp_features[i]
            LDR2_table[key1] = {}
            for j in range(i+1,N):
                try:
                    key2 = snp_features[j]
                    data_slice = data[[key1, key2]]
                    LD_R2 = self.calculate_LD_r2(data_slice)
                    LDR2_table[key1][key2] = LD_R2
                except:
                    pass
            return LDR2_table 
        
        res = Parallel(n_jobs=nj, prefer="processes")(
            delayed(parallel_run_LD_r2)(i) for i in tqdm(range(N-1), colour='GREEN', file=sys.stdout)
        )

        res2 = {}
        for x in res:
            for k, v in x.items():
                res2[k] = v
        self.LD_r2_table = res2
        return self
        
    def snp_to_ordinal(self,dominant=False,recessive=False):
        data = self.data
        snp_features = self.snp_categorical
        data[snp_features] = AlleleTransformer().fit_transform(data[snp_features],dominant,recessive)
        self.data = data
        return self
        
    def calc_LD_r2_all_pairs2(self,nj=4):
        data = self.data[self.data[self.target]==0]
        snp_features = self.snp_categorical
        genotype_matrix = AlleleTransformer().fit_transform(data[snp_features])
        p = np.mean(genotype_matrix / 2, axis=0)  
        q = 1 - p
        genotype_centered = genotype_matrix - np.mean(genotype_matrix, axis=0)  
        covariance_matrix = (genotype_centered.T @ genotype_centered) / (genotype_matrix.shape[0] - 1)
        variances = np.var(genotype_matrix, axis=0, ddof=1)  
        epsilon = 1e-10
        r2_matrix = (covariance_matrix ** 2) / (variances[:, None] * variances[None, :] + epsilon)   
        res = pd.DataFrame(r2_matrix,columns=snp_features,index=snp_features)
        np.fill_diagonal(res.values, np.nan)
        self.LD_r2_table = res
        return self
        
    def filter_LD_r2(self,thr=0.8,nj=4,LD_r2_file="LD_r2_res_new.pkl",recalc=False,limit=None,fast=True):
        if recalc:
            if not fast:
                self.calc_LD_r2_all_pairs(nj,limit)
                LDR2_table = pd.DataFrame(self.LD_r2_table)
            else:
                self.calc_LD_r2_all_pairs2()
                LDR2_table = self.LD_r2_table
            LDR2_table.to_pickle("LD_r2_res_new.pkl")
        else:
            LDR2_table = pd.read_pickle(LD_r2_file)
 
        LD_snp = []
        data = self.data
        for row in LDR2_table.index:
            for col in LDR2_table.columns:
                val = LDR2_table.loc[row, col]
                if val:
                    if val>0.8:
                        LD_snp.append([row, col, val])    

        for row in LD_snp:
            try:
                col = row[0:2]
                Gfreq1 = self.calculate_genotype_freq(data,col[0])
                Gfreq2 = self.calculate_genotype_freq(data,col[1])
                allele_freq1 = self.calculate_allele_freq(Gfreq1)
                allele_freq2 = self.calculate_allele_freq(Gfreq2)
                tot1 = sum(allele_freq1.values())
                tot2 = sum(allele_freq2.values())
                
                MAF1 = min([x/tot1 for x in allele_freq1])
                MAF2 = min([x/tot2 for x in allele_freq2])
                if MAF1 > MAF2:
                    data = data.drop(col[1],axis=1)
                else:
                    data = data.drop(col[0],axis=1)
            except:
                pass
        self.data = data
        return self
                     
    def col_transform(self):
        numeric_transformer = Pipeline(steps=[
            #('imputer', SimpleImputer(missing_values = np.nan, strategy='mean')),  
            ('imputer', IterativeImputer(random_state=0)),  
            #('scaler1', MinMaxScaler()),
            #('scaler2', StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values = np.nan, strategy='most_frequent')),  
            #('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat1', categorical_transformer, self.categorical_features),
                ('cat2', categorical_transformer, self.snp_categorical),
                ('num', numeric_transformer, self.numeric_features)
            ],
            remainder='drop'
        )
            
        self.preprocessor.fit(self.X_data)
        self.X_data = pd.DataFrame(
            self.preprocessor.transform(self.X_data),
            columns = self.selected_features
        )  
        return self
        
    def define_XYdata(self):
        self.X_data = self.data.loc[:, self.selected_features].fillna(np.nan) #  
        self.Y_data = self.data[self.target]
        self.X_data.drop(self.target,axis=1)
        self.categorical_features = list(set(self.categorical_features)-{self.target})
        self.snp_categorical = list(set(self.snp_categorical)-{self.target})
        self.numeric_features = list(set(self.numeric_features)-{self.target})
        self.selected_features = self.categorical_features + self.snp_categorical + self.numeric_features
        return self
        
    def get_data(self):
        return self.data
        
    def get_Xdata(self):
        return self.X_data
        
    def get_outputs(self):
        return self.X_data, self.Y_data, self.categorical_features, self.snp_categorical
