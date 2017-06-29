

print """

    ################################################################
    #                                                              #
    #        Python Script to select feature for ML modeling       #
    #                                                              #
    ################################################################
    #    Written By: Jayadev Joshi, ZebraFish lab, INMAS, DRDO.    #
    #                                                              #
    #      Before processing your data make sure that:             #
    #                                                              #
    #        1. csv file should have column headers                #
    #        2. Class lable should be at last column.              #
    #                                                              #
    ################################################################
    #  USES:                                                       #
    #  python Feature_selection.py -t test.csv -c 0.6 --o out.csv  #
    #                                                              #
    ################################################################


    """

import numpy as np
import pylab as pl
import pandas as pd
from sklearn import datasets, svm
from sklearn.feature_selection import SelectFpr, f_classif


class feature(object):
    
         
    def data_gen(self, in_file):
        
        self.in_file = in_file
        
        self.df = pd.read_csv(self.in_file)
        
        self.clm_list = []
        
        for column in self.df.columns:
            self.clm_list.append(column)

        x = self.df[self.clm_list[0:len(self.clm_list)-1]].values
        y = self.df[self.clm_list[len(self.clm_list)-1]].values
        
        x_indices = np.arange(x.shape[-1])
        
        return x, y, x_indices, self.clm_list, self.df
        

    def svm_feature_Selection(self, x_Data, y_Data):


        clf = svm.SVC(kernel='linear')
        clf.fit(x_Data, y_Data)
        svm_weights = (clf.coef_**2).sum(axis=0)
        svm_weights /= svm_weights.max()

        return  svm_weights

    def plot2(self, xval):

        conter = 0
        ylist = []
        for x in xval:
            conter = conter+1
            ylist.append(x)
        xlist  = range(0, conter)
        ax = pl.subplot(111)
        ax.bar(xlist, ylist, width=1, label='SVM weight', color='r')
        pl.axis('tight')
        pl.xlabel('Feature number')
        pl.ylabel('Weight')
        pl.legend(loc='upper right')
        pl.savefig("Selected_feature.jpg")
        pl.close()

    def feature_selection_main(self, infile, result_sum, cut_off):
        
        self.infile = infile
        self.result_sum = result_sum
        self.cut_off =  cut_off
        
        self.selected_out = []
        self.des_list = []
        
        self.data_x, self.data_y, self.x_ind, self.clm_list, self.df = feature().data_gen(self.infile)
        svm_wieghts = feature().svm_feature_Selection(self.data_x, self.data_y)
        
        sno = 0
        
        for i, svmw in enumerate(svm_wieghts):
            
            if svmw >= float(self.cut_off):
                sno = sno + 1

                self.selected_out.append([sno, self.x_ind[i+1], round(svmw, 1), self.clm_list[i]])
                self.des_list.append(self.clm_list[i])
       
        self.predf = pd.DataFrame(self.selected_out, columns=["Sn","feature index", "svm_wieght", "feature_name"])
        self.predf.to_csv(self.result_sum,index = False)

        print "Result written to the file ! ---> ", self.result_sum

        self.df.to_csv("reduced_Data.csv",columns = self.des_list+["class_label"], index = False)
        
        print "Selected features written to the file ! ---> ", "reduced.data.csv"
               
        feature().plot2(svm_wieghts)

if __name__=="__main__":
    

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--train",
                        required=True,
                        default=None,
                        help="Path to training data as csv file")
                        
    parser.add_argument("-c", "--cutoff",
                        required=False,
                        default=0.5,
                        help="cutoff value for feature selection (between 0 to 1, default is 0.5)")    
                        
    parser.add_argument("-o", "--out",
                        required=False,
                        default = "selected_feature.csv",
                        help="output file name") 
                        
    
    args = parser.parse_args()
    
    
    c = feature()

    c.feature_selection_main(args.train, args.out, args.cutoff)








