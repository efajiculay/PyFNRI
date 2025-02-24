import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import sys
import os
from PyFNRI.snplogit.helper_funct import generate_random_color
from statsmodels.stats.outliers_influence import variance_inflation_factor

class univariate:
    
    def __init__(self,X_train,Y_train,X_train_dummies,column_map,col_dict,
                 X_test=None,Y_test=None, X_test_dummies=None,working_folder=".",univar_folder=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_train_dummies = X_train_dummies
        self.col_dict = col_dict
        self.column_map = column_map
        self.X_teset = X_test
        self.Y_test = Y_test
        self.X_test_dummies = X_test_dummies        
        if not univar_folder:
            univar_folder = "univar_folder"
        if not os.path.exists(univar_folder):
            os.makedirs(univar_folder)
        self.univar_folder = univar_folder    
        self.working_folder = working_folder

    def get_LogReg_Univar(self,X,Y,col):
        col1 = self.column_map[col]
        x = np.array(X)
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        y = np.array(Y)  
        x = pd.DataFrame(x,columns=col1)
        x=sm.add_constant(x)
        logit_model=sm.Logit(y,x)
        res0 = logit_model.fit(disp=0,method='newton')
        
        file = self.univar_folder+"/"+col.replace("/","").replace("|","_")+".txt"
        with open(file,"w") as f:
            f.write(str(res0.summary()))
        coef = res0.params
        pval = np.nan_to_num(res0.pvalues,nan=9999)
        ci_val = np.nan_to_num(res0.conf_int(0.05),nan=9999)
        return {
            'x1'     : col1,
            'OR'     : np.exp(coef)[1:],
            'p-vals' : pval[1:],
            'CI_0.025' : ci_val[:,0][1:],
            'CI_0.975' : ci_val[:,1][1:]
        }
    
    def run_univar(self):
        
        log_reg_univariate = []
        failed_cols = []
        column_list = self.X_train.columns

        def univar_parallel(i):
            col = column_list[i]
            if col in self.col_dict:
                try:
                    x_train = self.X_train_dummies[self.column_map[col]]
                    y_train = self.Y_train
                    res = self.get_LogReg_Univar(x_train,y_train,col)
                    for i in range(len(self.column_map[col])):
                        try:
                            log_reg_univariate.append({
                                'x1'       : res['x1'][i],
                                'OR'       : res['OR'][i],
                                'p-vals'   : res['p-vals'][i],
                                'CI_0.025' : res['CI_0.025'][i],
                                'CI_0.975' : res['CI_0.975'][i]
                            })
                        except:
                            print(col)
                            pass
                except:
                    pass
                
        res = Parallel(n_jobs=1, prefer="threads")(
            delayed(univar_parallel)(i) for i in tqdm(range(column_list.shape[0]), colour='GREEN', file=sys.stdout)
        )
   
        res = pd.DataFrame(log_reg_univariate)#.drop('model',axis=1)
        res.reset_index(inplace = True)

        res["main_column"] = [None]*res.shape[0]
        res["dummy_column"] = [None]*res.shape[0]
        res["main_description"] = [None]*res.shape[0]

        for i in range(res.shape[0]):
            row = res.loc[i][1]
            if len(row.split("_"))>1:
                col0 = row.split("_")[0]
                col1 = row.split("_")[1]
                res.loc[i,"main_column"] = col0
                res.loc[i,"main_description"] =  self.col_dict[col0][2]
                try:
                    res.loc[i,"dummy_column"] = self.col_dict[col0][1][col1]
                except:
                    res.loc[i,"dummy_column"] = col1
            else:
                res.loc[i,"main_column"] = row
                res.loc[i,"main_description"] =  self.col_dict[row.strip()][2]

        univar_with_pval = res.drop("index",axis=1).reindex().sort_values(
            by=['main_column','p-vals', 'OR'],
            ascending=[True, True, False]
        )
        self.univar_with_pval = univar_with_pval

        univar_groups = {}
        for i in range(univar_with_pval.shape[0]):
            key = univar_with_pval.loc[i].x1.split("_")[0]
            if key in univar_groups:
                univar_groups[key].append(univar_with_pval.loc[i])
            else:
                univar_groups[key] = [univar_with_pval.loc[i]]
                
        univar_groups_pval = []
        for x in univar_groups:
            group = pd.DataFrame(univar_groups[x])
            min_pval = group["p-vals"].min()
            univar_groups_pval.append([x,min_pval])
            
        sorted_group_names = sorted(univar_groups_pval, key=lambda x: x[1])

        sorted_univar_by_group = []
        for x in sorted_group_names:
            sorted_univar_by_group.append(pd.DataFrame(univar_groups[x[0]]).sort_values(by="p-vals"))
            
        res = pd.concat(sorted_univar_by_group)
        res = res.reset_index().drop("index",axis=1)
        res.insert(1,"ref_var",[None]*len(res))

        for i in range(len(res)):
            key = res.loc[i,"x1"].split("_")[0]
            res.loc[i,"ref_var"] = str(list(self.col_dict[key][1])[0]) + " : " + str(list(self.col_dict[key][1].values())[0])
            
        all_codes = {}
        def style_ko(val):
            res = ""
            if val.main_column in all_codes:
                res = 'background-color: '+all_codes[val.main_column]
            else:
                c = generate_random_color()
                res = 'background-color: '+c
                all_codes[val.main_column] = c
            return [res for x in val]
        res2 = res.style.apply(style_ko, axis=1)
        self.univar_res = res2
        res2.to_excel(self.working_folder+"/Univariate_new6.xlsx", engine='openpyxl')
        return self
        
    def save_univar_result(self,f_name="Univariate_new6.xlsx"):
        self.res2.to_excel(self.working_folder+"/"+f_name, engine='openpyxl')  
        return self
        
    def FDR_Holm_Bonferroni(self,thr=0.05,pass_filter=[]):
        
        if len(pass_filter)>0:
            univar_with_pval_sorted = self.univar_with_pval.loc[:,:]           
            univar_with_pval_sorted["x-index"]= univar_with_pval_sorted["x1"]
            univar_with_pval_sorted = univar_with_pval_sorted.set_index("x-index")
            univar_with_pval_sorted = univar_with_pval_sorted.loc[pass_filter]          
            univar_with_pval_sorted = univar_with_pval_sorted.sort_values("p-vals").reset_index()
        else:
            univar_with_pval_sorted = self.univar_with_pval.sort_values("p-vals").reset_index()

        col_passing_univar = []
        col_failing_univar = []
        log_reg_univariate = univar_with_pval_sorted.loc[:,:]

        m = log_reg_univariate.shape[0]
        for i in range(log_reg_univariate.shape[0]):
            d = log_reg_univariate.loc[i,'x1']
            pval =  log_reg_univariate.loc[i,'p-vals']
            k = i + 1
            if pval<thr/(m+1-k):
                col_passing_univar.append(d)
            else:
                col_failing_univar.append(d)
                
        self.pass_fdr_hb = list(pd.unique([x for x in col_passing_univar ]))
        pass_uni_ori = pd.unique([x.split("_")[0].strip() for x in col_passing_univar ])
        self.pass_FDR_HB = pass_uni_ori

        return self
        
    def FDR_Benjamini_Hochberg(self,thr=0.05,pass_filter=[]):
        
        if len(pass_filter)>0:
            univar_with_pval_sorted = self.univar_with_pval.loc[:,:]           
            univar_with_pval_sorted["x-index"]= univar_with_pval_sorted["x1"]
            univar_with_pval_sorted = univar_with_pval_sorted.set_index("x-index")
            try:
                univar_with_pval_sorted = univar_with_pval_sorted.loc[pass_filter]    
            except:
                uni_dict = set(self.univar_with_pval.index.values)
                new_cols = []
                for x in pass_filter:
                    if x in uni_dict:
                        new_cols.append(x)
                try:
                    univar_with_pval_sorted = univar_with_pval_sorted.loc[new_cols]
                except:
                    column_list = []
                    for col in new_cols:
                        if col in self.col_dict:
                            column_list.extend(self.column_map[col])
                    univar_with_pval_sorted = univar_with_pval_sorted.loc[column_list]
                
            univar_with_pval_sorted = univar_with_pval_sorted.sort_values("p-vals").reset_index()
        else:
            univar_with_pval_sorted = self.univar_with_pval.sort_values("p-vals").reset_index()

        col_passing_univar = []
        col_failing_univar = []
        log_reg_univariate = univar_with_pval_sorted.loc[:,:]

        m = log_reg_univariate.shape[0]
        for i in range(log_reg_univariate.shape[0]):
            d = log_reg_univariate.loc[i,'x1']
            pval =  log_reg_univariate.loc[i,'p-vals']
            p_corr = pval*m/(i+1)
            if p_corr<thr:
                col_passing_univar.append(d)
            else:
                col_failing_univar.append(d)
                
        self.pass_fdr_bh = list(pd.unique([x for x in col_passing_univar ]))
        pass_uni_ori = pd.unique([x.split("_")[0].strip() for x in col_passing_univar ])
        self.pass_FDR_BH = pass_uni_ori

        return self
        
    def FDR_Benjamini_Yekutieli(self,thr=0.05,pass_filter=[]):
        
        if len(pass_filter)>0:
            univar_with_pval_sorted = self.univar_with_pval.loc[:,:]           
            univar_with_pval_sorted["x-index"]= univar_with_pval_sorted["x1"]
            univar_with_pval_sorted = univar_with_pval_sorted.set_index("x-index")
            univar_with_pval_sorted = univar_with_pval_sorted.loc[pass_filter]          
            univar_with_pval_sorted = univar_with_pval_sorted.sort_values("p-vals").reset_index()
        else:
            univar_with_pval_sorted = self.univar_with_pval.sort_values("p-vals").reset_index()

        col_passing_univar = []
        col_failing_univar = []
        log_reg_univariate = univar_with_pval_sorted.loc[:,:]

        m = log_reg_univariate.shape[0]
        c_m = sum([1/(j+1) for j in range(m)])
        for i in range(log_reg_univariate.shape[0]):
            d = log_reg_univariate.loc[i,'x1']
            pval =  log_reg_univariate.loc[i,'p-vals']
            p_corr = pval*m*c_m/(i+1)
            if p_corr<thr:
                col_passing_univar.append(d)
            else:
                col_failing_univar.append(d)
                
        self.pass_fdr_by = list(pd.unique([x for x in col_passing_univar ]))
        pass_uni_ori = pd.unique([x.split("_")[0].strip() for x in col_passing_univar ])
        self.pass_FDR_BY = pass_uni_ori

        return self
    
    pass_uni = []    
    def filter_pvalue(self,thr=0.05):

        univar_with_pval_sorted = self.univar_with_pval.sort_values("p-vals").reset_index()

        col_passing_univar = []
        col_failing_univar = []
        log_reg_univariate = univar_with_pval_sorted.loc[:,:]

        for i in range(log_reg_univariate.shape[0]):
            d = log_reg_univariate.loc[i,'x1']
            pval =  log_reg_univariate.loc[i,'p-vals']
            if pval<thr:
                col_passing_univar.append(d)
            else:
                col_failing_univar.append(d)
                
        self.pass_uni = list(pd.unique([x for x in col_passing_univar ]))
        pass_uni_ori = pd.unique([x.split("_")[0].strip() for x in col_passing_univar ])
        self.pass_UNI = pass_uni_ori

        return self
        
    def calculate_vif1(self,nj=4,selected=[]):
        if len(selected)==0:
            x1 = self.pass_vif
        else:
            x1 = selected
        X = self.X_train_dummies[x1]
        
        def vif_parallel(i):
            return variance_inflation_factor(X.values, i)  
               
        res = Parallel(n_jobs=nj, prefer="processes")(
            delayed(vif_parallel)(i) for i in tqdm(range(X.shape[1]), colour='GREEN', file=sys.stdout)
        )
 
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [x for x in res]
        self.vif_data = vif_data
        self.vif_data.to_pickle("vif_data.pkl")
        return self
        
    def calculate_vif2(self,selected=[]):
        if len(selected)==0:
            x1 = self.pass_vif
        else:
            x1 = selected
        X = self.X_train_dummies[x1]
        X_cor = X.corr()
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        xcorr = np.nan_to_num(X_cor.values)
        vif_data["VIF"] = np.diag(np.linalg.inv(xcorr+np.eye(xcorr.shape[0])*1.0e-10))
        self.vif_data = vif_data
        self.vif_data.to_pickle("vif_data.pkl")
        return self
        
    def filter_VIF(self,thr=10,nj=4,which=2,selected=[]):
        
        check = True
        if len(self.pass_uni)>0:
            self.pass_vif = self.pass_uni
        elif len(selected)==0:
            column_list = []
            for col in self.X_train.columns:
                if col in self.col_dict:
                    column_list.extend(self.column_map[col])
            self.pass_vif = column_list       
        else:
            self.pass_vif = selected
        while check:
            check = False
            if which == 2:
                self.calculate_vif2()
            else:
                self.calculate_vif1()
            vif_res = self.vif_data
            remove_var = None
            max_vif = 0
            for index, row in vif_res.iterrows():
                x, y = row["Variable"], row["VIF"]
                if y>thr:
                    max_vif
                    check = True
                    if y>max_vif:
                        max_vif = y
                        remove_var = x
                        
            if remove_var:  
                print(remove_var, "VIF = ", max_vif)
                self.pass_vif = list(set(self.pass_vif) - {remove_var})

        col_passing_univar = self.pass_vif
        pass_uni_ori = pd.unique([x.split("_")[0].strip() for x in col_passing_univar ])
        self.pass_VIF = pass_uni_ori
        return self
                