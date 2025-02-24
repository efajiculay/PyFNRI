import pandas as pd

class snp_frequency_calc:
    
    def __init__(self):
        pass
    
    def calculate_genotype_freq(self,data, SNPx, SNPy=None):
        if not SNPy:
            SNPy = SNPx
        gres = data.groupby(SNPx)[SNPy].count()
        if type(SNPx) != str:
            return pd.concat((gres, gres/gres.sum()))
        else:
            return pd.DataFrame({"Frequency":gres, "Percentage":gres/gres.sum(axis=0)})
    
    @classmethod
    def get_case_control_freq(self,data,sel_col,outcome):
        case = data.groupby(by=sel_col)[outcome].sum()
        alld = data.groupby(by=sel_col)[outcome].count()
        cont = data.groupby(by=sel_col)[outcome].count() - data.groupby(by=sel_col)[outcome].sum()
        return pd.DataFrame({"Total":alld, "Positive":case, "Negative" :cont, "Prevalence":case/alld})

    def calculate_allele_freq(self,Gfreq):
        allele_freq = {}
        for g in Gfreq.index:
            r = g.split("/")
            for i in range(2):
                if r[i] in allele_freq:
                    allele_freq[r[i]] += Gfreq.loc[g].Frequency
                else:
                    allele_freq[r[i]] =  Gfreq.loc[g].Frequency
        return allele_freq

    def expected_genotype_freq(self,allele_freq):
        tot = sum(allele_freq.values())
        p = {}
        for x in allele_freq:
            p[x] = allele_freq[x]/tot

        gfreq = {}
        for x in allele_freq:
            for y in allele_freq:
                key1 = x+"/"+y
                key2 = y+"/"+x     
                gfreq[key1] = p[x]*p[y]*tot/2
        return gfreq
        
    def calc_haplo_freq_obs(self,snp12slice):
        haplo_freq = {}
        for x in snp12slice.T:
            row = snp12slice.loc[x]
            loc1 = row.iloc[0]
            loc2 = row.iloc[1]
            try:
                r1 = loc1.split("/")
                r2 = loc2.split("/")
                for i in range(2):
                    for j in range(2):
                        key = r1[i]+"_"+r2[j]
                        if key in haplo_freq:
                            haplo_freq[key] += 1
                        else:
                            haplo_freq[key] = 1
            except:
                pass
        return haplo_freq