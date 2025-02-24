import pandas as pd


class read_data:
    
    """
    This class is for reading data files either in excel or csv format. 
    It can also perform column renaming and row shuffling for avoiding order related systematic associations.
    """
       
    def __init__(self,file,sheet_name=None):
        """
        This function initiates the data and data_xlsx attributes
        Parameters:
        
            self - pertains to this class
            file - name of the file or the directory to the file
            sheet_name - optional parameter. only applicable to excel files
        """
        self.data = self.pd_read(file,sheet_name)
            
    def pd_read(self,file,sheet_name=None):
        """
        This function initiates the data and data_xlsx attributes by calling other 
        functions from the read_data class. The file name will be splitted with dot
        as the delimiter. The last element in the split is taken as the suffix.
        
        Parameters:
        
            self - pertains to this class
            file - name of the file or the directory to the file
            sheet_name - optional parameter. only applicable to excel files
        """
        file_suffix = file.split(".")[-1].lower()
        if file_suffix == "csv":
            data = pd.read_csv(file)
        elif file_suffix == "xlsx":
            if not sheet_name:
                data = pd.read_excel(file)
            else:
                self.data_xlsx = pd.ExcelFile(file)
                data = self.parse_xlsx_sheet(sheet_name)  
        else:
            print("File extension is not a csv and not excel")
        return data
        
            
    def shuffle_rows(self, random_state = 42):
        """
        This function shuffles the order of the rows on the dataframe. This is done to 
        ensure that there will be no systematic association just due to the order of the data
        """
        self.data = self.data.sample(frac = 1,random_state=random_state)
        return self
        
    def fix_column(self,ch1="_",ch2="|"):
        """
        This function replaces some characters in the column names that may affect the rest of the steps.
        """
        self.data.columns = map(lambda x : x.replace(ch1,ch2).strip(), self.data.columns)
        return self
        
    def get_data(self):
        """Returns the dataframe with the rows shuffled and the colunm names fixed"""
        self.shuffle_rows().fix_column()
        return self.data
        
    def parse_xlsx_sheet(self,sheet_name):
        """Function for choosing sheet if file in excel"""
        return self.data_xlsx.parse(sheet_name)
        
    def merge_data(self, file1, file2, common_col="", sheet_name1=None, sheet_name2=None):
        data1 = self.pd_read(file1,sheet_name1)
        data2 = self.pd_read(file2,sheet_name2)  
        
        data1_cols = set(data1.columns)
        sel_cols = [common_col]
        for col in data2.columns:
            if col not in data1:
                sel_cols.append(col)
                
        self.data = pd.merge(data1, data2[sel_cols], on=common_col, how="inner",suffixes=(None,))
        return self
        
    def save_to_file(self,fname):
        file_suffix = fname.split(".")[-1].lower()
        if file_suffix == "csv":
            self.data.to_csv(fname)
        elif file_suffix == "xlsx":
            self.data.to_excel(fname)
        return self



        
    
    
            
        