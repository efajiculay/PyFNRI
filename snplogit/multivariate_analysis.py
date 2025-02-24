import matplotlib.pylab as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate, cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import norm
import seaborn as sns
import os
import sympy as sym
import pygad

class multivariate():
    
    def __init__(self,X_train,Y_train,X_train_dummies,column_map,col_dict,
                 X_test=None,Y_test=None, X_test_dummies=None,working_folder=".",multivar_folder=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_train_dummies = X_train_dummies
        self.col_dict = col_dict
        self.column_map = column_map
        self.X_teset = X_test
        self.Y_test = Y_test
        self.X_test_dummies = X_test_dummies        
        if not multivar_folder:
            multivar_folder = "multivar_folder"
        if not os.path.exists(multivar_folder):
            os.makedirs(multivar_folder)
        self.multivar_folder = multivar_folder    
        self.working_folder = working_folder

    def conf_mat_plot(self,yclabels,conf_mat):
        labels = yclabels
        fig, ax = plt.subplots(figsize=(3*4/3,2.5*4/3))
        sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax)  
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(yclabels)
        ax.yaxis.set_ticklabels(yclabels)
        plt.show()
        
    def roc_auc_plot(self,yclabels,zz):
        labels = yclabels
        plt.figure(figsize=(3,2.5))
        plt.plot(zz[0],zz[1],label=labels[1])
        plt.xlabel("1-specificity")
        plt.ylabel("sensitivity")
        plt.legend()
        plt.show()
     
    def result_summary(self,logr, X_dummies, Y, yclabels=["Positive","Normal"]):
        prob_res = logr.predict_proba(X_dummies)
        pred_res = logr.predict(X_dummies)
        acc_res = np.sum(pred_res == Y)/Y.shape[0]
        labels = yclabels
        conf_mat = confusion_matrix(Y, pred_res, labels=labels)
        self.conf_mat_plot(yclabels, conf_mat)

        overall_accuracy = (conf_mat[0,0]+conf_mat[1,1])/np.sum(conf_mat)
        sensitivity = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
        specificity = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
        PPV = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,0])
        NPV = conf_mat[1,1]/(conf_mat[0,1]+conf_mat[1,1])

        y = Y == yclabels[0]
        zz = roc_curve(y, prob_res[:,1])
        auc_res = roc_auc_score(y, prob_res[:,1])
        self.roc_auc_plot(yclabels,zz)
        return overall_accuracy, sensitivity, specificity, PPV, NPV, auc_res

    def logit_summary_stat(self,X_train, resLogit,C=0):
        predProbs = resLogit.predict_proba(X_train)
        X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        V = np.diagflat(np.product(predProbs, axis=1))
        xTVx = np.dot(np.dot(X_design.T, V), X_design)
        xTVx = xTVx + np.eye(xTVx.shape[0])*C
        covLogit = np.linalg.inv(xTVx)

        cf = [*resLogit.intercept_.flatten(), *resLogit.coef_.flatten()]
        se = np.sqrt(np.diag(covLogit))
        zs = [*(resLogit.intercept_/se[0]).flatten(), *(resLogit.coef_/se[1:]).flatten()]
        pv = norm.sf(np.abs(zs))*2
        ci1 = cf - 1.96*se
        ci2 = cf + 1.96*se
        cf, se, zs, pv, ci1, ci2 = [np.round(x,5) for x in [cf, se, zs, pv, ci1, ci2]]
        return pd.DataFrame(
            np.vstack((["const",*X_train.columns],cf,se,zs,pv,ci1,ci2)).T,
            columns=["Vars","coef","Std Err","z","P>|z|","ci1","ci2"]
        ).set_index("Vars").astype(float)
        
    def run_multivar(self, selected_var, adjust_by=[], step_wise=True, print_remove=True):
        
        Y_train = self.Y_train
        column_map = self.column_map
        
        if len(adjust_by)>0:
            selected_var = list(set(selected_var).union(set(adjust_by)))
        
        logr = LogisticRegression(
            penalty='l2',
            C=1000,  
            class_weight='balanced',
            solver='liblinear'
        )
        
        largest_pval = 1.0
        while largest_pval>0.05:
            selected_dummy = []
            for g in selected_var:
                for m in column_map[g]:
                    selected_dummy.append(m.strip())

            x_train = self.X_train_dummies[selected_dummy]
            resLogit = logr.fit(x_train, Y_train)

            res0 = self.logit_summary_stat(x_train, resLogit,C=1.0e-10)
            if step_wise:
                pvals_tups = res0["P>|z|"]
                pvals_dict = {}
                for x in pvals_tups.keys()[1:]:
                    r = x.split(":")
                    if len(r) != 2:
                        key = x.split("_")[0]
                        if key in pvals_dict:
                            pvals_dict[key].append(pvals_tups[x])
                        else:
                            pvals_dict[key] = [pvals_tups[x]]
                    else:
                        key1 = r[0].split("_")[0]
                        key2 = r[1].split("_")[0]
                        key = key1+":"+key2
                        if key in pvals_dict:
                            pvals_dict[key].append(pvals_tups[x])
                        else:
                            pvals_dict[key] = [pvals_tups[x]]           
                        
                pvals_min = {}
                for x in pvals_dict:
                    pvals_min[x] = max([np.nan_to_num(y) for y in pvals_dict[x]])

                max_cat = ""
                pvals_max = 0
                for h in pvals_min:
                    if h in set(adjust_by):
                        pass
                    else:
                        if pvals_min[h]>pvals_max:
                            pvals_max = pvals_min[h]
                            max_cat = h

                if pvals_max>0.05:
                    selected_var.remove(max_cat)
                    if print_remove:
                        print(max_cat, "removed")
                else:
                    largest_pval = pvals_max
            else:
                break
        
        self.multivar_table = res0
        self.logr = logr
        self.selected_dummy = selected_dummy
        return self
        
    def cross_validation(self,cv=None,nj=4,labels=["Positive","Normal"],C=1000):
        x_train_dummies, y_train = self.X_train_dummies[self.selected_dummy], self.Y_train 
        
        logr = LogisticRegression(
            penalty='l2',
            C=C,
            class_weight='balanced',
            solver='liblinear',
            fit_intercept=True
        )        
        
        y_train = np.array(y_train).reshape(-1,1)
        y_train = y_train.astype(str)
        y_train[y_train=="1"] = labels[0]
        y_train[y_train=="0"] = labels[1]      
        y_train = y_train.flatten().astype(str)   
        
        if not cv:
            cv = int(x_train_dummies.shape[0]*0.2)
        cv_results = cross_validate(
            logr, x_train_dummies, y_train, cv=cv,n_jobs=nj,
            scoring=('roc_auc', 'accuracy', 'precision')
        )
        return cv_results
        
    def kfold_analysis(self,nf=2,nj=4,labels=["Positive","Normal"],C=1000): 
        kf = KFold(n_splits=nf)
        x_train_dummies, y_train = self.X_train_dummies[self.selected_dummy].values, self.Y_train 
        kf.get_n_splits(x_train_dummies)
        
        logr = LogisticRegression(
            penalty='l2',
            C=C,
            class_weight='balanced',
            solver='liblinear',
            fit_intercept=True
        )        
        
        y_train = np.array(y_train).reshape(-1,1)
        y_train = y_train.astype(str)
        y_train[y_train=="1"] = labels[0]
        y_train[y_train=="0"] = labels[1]      
        y_train = y_train.flatten().astype(str)   
        
        overall_accuracy1 = []
        sensitivity1 = []
        specificity1 = []
        PPV1 = []
        NPV1 = []
        auc1 = []
        zz1 = []
        
        overall_accuracy2 = []
        sensitivity2 = []
        specificity2 = []
        PPV2 = []
        NPV2 = []
        auc2 = []
        zz2 = []
        for i, (train_index, test_index) in enumerate(kf.split(x_train_dummies)):
            x_train = x_train_dummies[train_index]
            x_valid = x_train_dummies[test_index]
            train_y = y_train[train_index]
            test_y = y_train[test_index]
       
            logr.fit(x_train, train_y)
            
            prob_res = logr.predict_proba(x_train)
            pred_res = logr.predict(x_train)
            acc_res = np.sum(pred_res == train_y)/train_y.shape[0]
            conf_mat = confusion_matrix(train_y, pred_res, labels=labels)

            v1 = (conf_mat[0,0]+conf_mat[1,1])/np.sum(conf_mat)  
            v2 = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
            v3 = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
            v4 = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,0])
            v5 = conf_mat[1,1]/(conf_mat[0,1]+conf_mat[1,1])
            
            overall_accuracy1.append(v1)
            sensitivity1.append(v2)
            specificity1.append(v3)
            PPV1.append(v4)
            NPV1.append(v5)

            y = train_y == labels[0]
            zz = roc_curve(y, prob_res[:,1])
            zz1.append(zz)
            auc_res = roc_auc_score(y, prob_res[:,1])
            auc1.append(auc_res)
            
            prob_res = logr.predict_proba(x_valid)
            pred_res = logr.predict(x_valid)
            acc_res = np.sum(pred_res == test_y)/test_y.shape[0]
            conf_mat = confusion_matrix(test_y, pred_res, labels=labels)

            v1 = (conf_mat[0,0]+conf_mat[1,1])/np.sum(conf_mat)  
            v2 = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
            v3 = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
            v4 = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,0])
            v5 = conf_mat[1,1]/(conf_mat[0,1]+conf_mat[1,1])
            
            overall_accuracy2.append(v1)
            sensitivity2.append(v2)
            specificity2.append(v3)
            PPV2.append(v4)
            NPV2.append(v5)

            y = test_y == labels[0]
            zz = roc_curve(y, prob_res[:,1])
            zz2.append(zz)
            auc_res = roc_auc_score(y, prob_res[:,1])
            auc2.append(auc_res)
   
        self.kfold_res = {   
            "train_res" : {
                "accuracy" : overall_accuracy1, "sensitivity" : sensitivity1, "specificity" : specificity1, 
                "PPV" : PPV1, "NPV" : NPV1, "AUC" : auc1, "zz" : zz1
                },
            "test_res"  : {
                "accuracy" : overall_accuracy2, "sensitivity" : sensitivity2, "specificity" : specificity2, 
                "PPV" : PPV2, "NPV" : NPV2, "AUC" : auc2, "zz" : zz2
            }   
        }
        return self
        
    def plot_kfold(self,labels=["Positive","Normal"],px=0.49,py=0.0,dy=0.1,fz=9):
        kfold_res = self.kfold_res

        trainFold = kfold_res["train_res"]
        testFold = kfold_res["test_res"]

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9,2.5))
        
        for zz in trainFold["zz"]:
            ax1.plot(zz[0],zz[1],label=labels[0])
            
        for zz in testFold["zz"]:
            ax2.plot(zz[0],zz[1],label=labels[0])
            
        ax1.set_xlabel("1-specificity")
        ax1.set_ylabel("sensitivity")
        ax1.set_title("training-set")
        j = 0
        for x in trainFold.keys():
            if x != "zz":
                v = trainFold[x]
                s = x+" = "+str(np.round(np.mean(v),2))+"(±"+str(np.round(np.sqrt(np.var(v)),2))+")"
                ax1.text(px,py+j,s,fontsize=fz)
                j = j + dy
        ax2.set_xlabel("1-specificity")
        ax2.set_ylabel("sensitivity")
        ax2.set_title("cross validation")
        j = 0
        for x in testFold.keys():
            if x != "zz":
                v = testFold[x]
                s = x+" = "+str(np.round(np.mean(v),2))+"(±"+str(np.round(np.sqrt(np.var(v)),2))+")"
                ax2.text(px,py+j,s,fontsize=fz)
                j = j + dy
        plt.show()
                     
    def performance_training_set(self,labels=["Positive","Normal"],C=1000):
        x_train_dummies, y_train = self.X_train_dummies[self.selected_dummy], self.Y_train 
        
        logr = LogisticRegression(
            penalty='l2',
            C=C,
            class_weight='balanced',
            solver='liblinear',
            fit_intercept=True
        )
        
        y_train = np.array(y_train).reshape(-1,1)
        y_train = y_train.astype(str)
        y_train[y_train=="1"] = labels[0]
        y_train[y_train=="0"] = labels[1]      
        y_train = y_train.flatten().astype(str)        
        
        resLogit = logr.fit(x_train_dummies, y_train)
        self.logr = logr
        
        overall_accuracy, sensitivity, specificity, PPV, NPV, auc_res = self.result_summary(logr, x_train_dummies, y_train, labels)
        print(
            "\n",
            "overall_accuracy = ", overall_accuracy,"\n", 
            "sensitivity = ", sensitivity,"\n", 
            "specificity = ", specificity,"\n", 
            "PPV = ", PPV,"\n", 
            "NPV = ", NPV,"\n",
            "auc =", auc_res 
        )
        
    def performance_testing_set(self,labels=["Positive","Normal"]):
        x_test_dummies, y_test = self.X_test_dummies[self.selected_dummy], self.Y_test 
        
        logr = self.logr
        
        y_test = np.array(y_test).reshape(-1,1)
        y_test = y_test.astype(str)
        y_test[y_test=="1"] = labels[0]
        y_test[y_test=="0"] = labels[1]      
        y_test = y_test.flatten().astype(str)        
                
        overall_accuracy, sensitivity, specificity, PPV, NPV, auc_res = self.result_summary(logr, x_test_dummies, y_test, labels)
        print(
            "\n",
            "overall_accuracy = ", overall_accuracy,"\n", 
            "sensitivity = ", sensitivity,"\n", 
            "specificity = ", specificity,"\n", 
            "PPV = ", PPV,"\n", 
            "NPV = ", NPV,"\n",
            "auc =", auc_res 
        )
        
    def get_polygenic_risk_score(self,normal="Normal",positive="Positive"):
        Nn = sym.Symbol(normal)
        Dd = sym.Symbol(positive)
        P = sym.Function("P")
        
        SNPs = self.selected_dummy
        resLogit = self.logr
        cf = [*resLogit.intercept_.flatten(), *resLogit.coef_.flatten()]
        params = sym.Matrix(cf)
        snpvr = sym.Matrix(sym.var(",".join(["1",*SNPs])))
        logMD = params.T*snpvr
        return sym.Eq(sym.log(P(Dd)/P(Nn)),logMD[0]) 
        
    def fitness_func(self, ga_instance, solution, solution_idx):
        try: 
            current_vars = self.current_vars_ga
            if np.sum(solution) == 0:
                return [0]
            c = []
            for i in range(len(solution)):
                if solution[i] == 1:
                    c.append(current_vars[i])
            selected_var = c
            self.run_multivar(selected_var, step_wise=True, print_remove=False)
            
            col = "".join(sorted(self.selected_dummy))
            file = self.multivar_folder+"/"+col.replace("/","").replace("|","_")+".txt"
            with open(file,"w") as f:
                f.write(str(self.multivar_table))            
               
            res = self.cross_validation()
            auc, acc = np.mean(res['test_roc_auc']), np.mean(res['test_accuracy'])
            return [1/(2-auc-acc)]
        except:
            return [0]
        
    def on_generation(self, ga_instance):
        current_vars = self.current_vars_ga
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        c = []
        for i in range(len(solution)):
            if solution[i] == 1:
                c.append(current_vars[i])
                
    def GA_SNP_multivar(self, current_vars, num_generations = 20, num_parents_mating = 20, sol_per_pop = 30):    
        self.current_vars_ga = current_vars
        self.selected_dummy = current_vars
        num_genes = len(current_vars)
        ini_pop = []
        
        for i in range(num_genes):
            #ini_pop.append(np.random.randint(2, size=num_genes))
            ini_pop.append([0]*num_genes)
            ini_pop[-1][i] = 1

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            fitness_func=self.fitness_func,
            parent_selection_type='nsga2',
            gene_space = [0,1],
            initial_population=ini_pop,
            #parallel_processing=["process", 4],
            keep_elitism=10,
            keep_parents=10,
            on_generation=self.on_generation,
            save_best_solutions=True,
            #stop_criteria="reach_1.28",
            crossover_type='uniform'
        )

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)


        c = []
        for i in range(len(solution)):
            if solution[i] == 1:
                c.append(current_vars[i])

        self.selected_dummy = c
        self.run_multivar(self.selected_dummy,step_wise=True, print_remove=False)
        return self