import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class process_dummy_col:
    
    def __init__(self, Xdata, Ydata, categorical_features, snp_categorical, snp_as_ordinal=False, working_folder="."):
        self.X_data = Xdata
        self.Y_data = Ydata
        self.categorical_features = categorical_features
        self.snp_categorical = snp_categorical
        self.working_folder = working_folder
        self.snp_as_ordinal = snp_as_ordinal
        
    def meta_data_dict(self):
        self.col_dict = {}
        if not self.snp_as_ordinal:
            cat_features = self.categorical_features + self.snp_categorical
        else:
            cat_features = self.categorical_features
        for col in self.X_data:
            feature_type = "cat" if col in set(cat_features) else "con"
            self.col_dict[col] = [feature_type,{},""]
            if feature_type == "cat":
                unique = pd.unique(self.X_data[col])
                for ux in unique:
                    self.col_dict[col][1][0] = set(unique)
            else:
                self.col_dict[col][1][0] = None
        return self
                
    def get_dummies(self, sfile=""):
        Xdata = self.X_data
        Ydata = self.Y_data

        self.Xdata_with_dummies = {}
        self.column_map = {}
        for col in Xdata.columns:
            if col in self.col_dict:
                self.column_map[col] = []
                if self.col_dict[col][0] in ["cat", "cat-ord"]:
                    major = Xdata[col].value_counts().idxmax().strip()
                    new = pd.get_dummies([str(x) for x in list(Xdata[col])],drop_first=False).drop(major,axis=1)
                    self.col_dict[col][1][0] = list(self.col_dict[col][1][0] - set(new.columns))[0]
                    g = 0
                    for c in new.columns:
                        cc = str(col)+"_"+str(c)
                        self.Xdata_with_dummies[cc] = [x for x in list(new[c])]
                        self.column_map[col].append(cc)
                        g = g + 1
                        self.col_dict[col][1][g] = c
                else:
                    self.Xdata_with_dummies[col] = [x for x in list(Xdata[col]) ]
                    self.column_map[col] = [col]

        self.Xdata_with_dummies = pd.DataFrame(self.Xdata_with_dummies).astype(float)

        self.Xdata_with_dummies['label'] = Ydata
        self.Xdata_with_dummies.index = Xdata.index
        self.Xdata_with_dummies = self.Xdata_with_dummies.drop("label",axis=1)
        self.Xdata_with_dummies.to_pickle(self.working_folder+"/Filtered_data_with_dummies_with_labels_at_last_row.pkl")
        np.save(self.working_folder+"/column_map.npy",self.column_map,allow_pickle=True)
        return self
        
    def remove_invarying_col(self):
        self.Xdata_with_dummies = self.Xdata_with_dummies.loc[:, 
        (self.Xdata_with_dummies != self.Xdata_with_dummies.iloc[0]).any()]
        self.X_data = self.X_data.loc[:, 
        (self.X_data != self.X_data.iloc[0]).any()]
        return self
        
    def get_Xdata_with_dummies(self):
        return self.Xdata_with_dummies
        
    def get_Xdata_Ydata_XdataWithDummies_colmap_coldict(self):
        return self.X_data, self.Y_data, self.Xdata_with_dummies, self.column_map, self.col_dict
        
    def save_Xdata(self):
        self.X_data.to_pickle(self.working_folder+"/X_data.pkl")
        return self
        
    def save_Ydata(self):
        self.Y_data.to_pickle(self.working_folder+"/Y_data.pkl")
        return self
        
    def save_XdataWithDummies(self):
        self.Xdata_with_dummies.to_pickle(self.working_folder+"/Y_data.pkl")
        return self
        
    def separate_training_testing(self, random_state = 42, test_size=0.2):
        Xdata = self.X_data
        Ydata = self.Y_data
        if test_size != 0:
            X_train, X_test, Y_train, Y_test = train_test_split(Xdata, Ydata, test_size=test_size, random_state=random_state)
            self.X_train_dummies = self.Xdata_with_dummies.loc[X_train.index]
            self.X_test_dummies = self.Xdata_with_dummies.loc[X_test.index]
            self.X_train = X_train
            self.Y_train = Y_train
            self.X_test = X_test
            self.Y_test = Y_test
        else:
            self.X_train, self.Y_train = Xdata, Ydata
            self.X_train_dummies = self.Xdata_with_dummies.loc[self.X_train.index]
        
        return self
        
    def get_Xtrain_Ytrain_Xtrain_dummies_colmap_coldict_Xtest_Ytest_Xtest_dummies(self):  
        return self.X_train, self.Y_train, self.X_train_dummies, self.column_map, self.col_dict, self.X_test, self.Y_test, self.X_test_dummies, 
        
    def get_Xtrain(self):
        return self.X_train
        
    def get_Ytrain(self):
        return self.Y_train

    def get_Xtest(self):
        return self.X_test
        
    def get_Ytest(self):
        return self.X_test
        
    def get_Xtrain_dummies(self):
        return self.X_train_dummies
        
    def get_Xtest_dummies(self):
        return self.X_test_dummies
        
    def save_Xtrain(self):
        self.X_train.to_pickle(self.working_folder+"/X_train.pkl")
        return self
        
    def save_Xtest(self):
        self.X_test.to_pickle(self.working_folder+"/X_test.pkl")
        return self
        
    def save_Ytrain(self):
        self.Y_train.to_pickle(self.working_folder+"/Y_train.pkl")
        return self
        
    def save_Ytest(self):
        self.Y_test.to_pickle(self.working_folder+"/Y_test.pkl")
        return self
        
    def save_Xtrain_dummies(self):
        self.X_train_dummies.to_pickle(self.working_folder+"/X_train_dummies.pkl")
        return self