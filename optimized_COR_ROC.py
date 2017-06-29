################################################################
#                                                              #
#        Python Script for 5 Fold cross Validation.            #
#                                                              #
################################################################
#    Written By: Jayadev Joshi, ZebraFish lab, INMAS, DRDO.    #
#                                                              #
#      Before processing your data make sure that:             #
#        1. Both csv file should have equal                    #
#            number of rows.                                   #
#        2. Class lable should be at last column.              #
#                                                              #
################################################################
#  USES:                                                       #
#       python optimized_COR_ROC.py -f target.csv              #
#                                                              #
################################################################

################################################################
# Modules being Imported 
################################################################

import numpy as np
import sys
from scipy import interp
import pylab as pl
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
import matplotlib.image as mpimg

################################################################


class My_csv_class(object):



    def data_gen(self,csv_path):
        
        self.csv_path = csv_path
        
        self.df = pd.read_csv(self.csv_path)

        clm_list = []
        for column in self.df.columns:
            clm_list.append(column)

        X_data = self.df[clm_list[0:len(clm_list)-1]].values
        y_data = self.df[clm_list[len(clm_list)-1]].values
        #print X_data, y_data
        return X_data, y_data
        
    def roc_gene(self, X, y, model,nfolds,st):
        
        self.st = st
     
        self.nfolds = nfolds
        
        if self.nfolds > 10:
            print "nfolds is too high"
            sys.exit()
            
        elif self.nfolds <= 2:
            print "nfolds value is too small "
            sys.exit()

        elif self.nfolds > 5:
            
             print "################################### Warrning ##########################################" 
             print " Please make sure that your data is large Enough to Properly Handle the high nfolds,>5 "
             print "#######################################################################################"
               
        else:
            pass

        specificity_list = []
        sensitivity_list = []
        presison_list = []
        mcc_list =  []
        f1_list = []

        folds = StratifiedKFold(y, n_folds=5)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(folds):

            prob = model.fit(X[train], y[train]).predict_proba(X[test])
            predicted = model.fit(X[train], y[train]).predict(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], prob[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            CM = confusion_matrix(y[test], predicted)
            new_list = []
            
            for i in CM:
                for i in i:
                    new_list.append(i)

            TP = float(new_list[0])+0.01
            FP = float(new_list[1])+0.01
            FN = float(new_list[2])+0.01
            TN = float(new_list[3])+0.01
            
            print TP, FP, FN, TN
            """
                  3.0 3.0 2.0 2.0
                  4.0 2.0 0.0 4.0
                  4.0 2.0 1.0 3.0
                  5.0 1.0 0.0 4.0
                  3.0 2.0 0.0 5.0
                  2.0 4.0 0.0 4.0
                  1.0 5.0 0.0 4.0
                  0.0 6.0 0.0 4.0
            """
            try:
                specificity = round(float(TN / (TN + FP)),2)
            except:
                print "Error in specificity"
                pass 
            
            try:
                sensitivity = round(float(TP / (TP + FN)),2)
            except:
                print "Erro in sensitivity"
                pass 
            try:
                presison = round(float(TP /(TP + FP)),2)
            except:
                print "Error in presison"
                pass
            try:
                mcc =  round(matthews_corrcoef(y[test], predicted),2)
            except:
                print "Error in mcc"
                pass
            try:
                f1 =  round(f1_score(y[test], predicted),2)
            except:
                print "Error in F1"
                pass

            specificity_list.append(specificity)
            sensitivity_list.append(sensitivity)
            presison_list.append(presison)
            mcc_list.append(mcc)
            f1_list.append(f1)

        spe_mean = float(sum(specificity_list))/float(len(specificity_list))
        sen_mean = float(sum(sensitivity_list))/float(len(sensitivity_list))
        pre_mean = float(sum(presison_list))/float(len(presison_list))
        mcc_mean = float(sum(mcc_list))/float(len(mcc_list))
        f1_mean = float(sum(f1_list))/float(len(f1_list))

        pl.plot([0, 1], [0, 1], '--', lw=2)
        mean_tpr /= len(folds)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        pl.plot(mean_fpr, mean_tpr, '-', label='AUC = %0.2f' % mean_auc, lw=2)

        pl.xlim([-0.05, 1.05])
        pl.ylim([-0.05, 1.05])
        pl.xlabel('FP Rate',fontsize=22)
        pl.tick_params(axis='x', labelsize=22)
        pl.tick_params(axis='y', labelsize=22)
        pl.ylabel('TP Rate',fontsize=22)
        pl.legend(loc="lower right")
        pl.axis('tight')
        
        self.V_header = ("specificity","sensitivity","presison","mcc","f1")
        self.v_values = (round(spe_mean, 2),round(sen_mean, 2),round(pre_mean, 2),round(mcc_mean, 2),round(f1_mean, 2))
        mname  = ("Logistic_Regression","GaussianNB","KNeighbors","DecisionTree","SVC")


        #print "specificity" , round(spe_mean, 2)
        #print "sensitivity" , round(sen_mean, 2)
        #print "presison   " , round(pre_mean, 2)
        #print "mcc        " , round(mcc_mean, 2)
        #print "f1         " , round(f1_mean, 2)

        pl.title(mname[self.st],fontsize=22)
        #pl.set_title(mname[self.st], fontsize=22)
        pl.savefig(mname[self.st]+".jpg")
        pl.close()
        pl.show()
        
        return self.V_header, self.v_values

    def main_pro(self, Des_Set,nfold):
        
        val_list = []
        result_summery = open("result_summery.csv",'w')
        
        self.Des_Set = Des_Set
        self.nfold = nfold
        X, y = My_csv_class().data_gen(self.Des_Set)      
                
        LR = LogisticRegression()
        GNB = GaussianNB()
        KNB = KNeighborsClassifier()
        DT = DecisionTreeClassifier()
        SV = SVC(probability=True)
        
        classifiers = (LR, GNB, KNB, DT, SV)
       
        for ni, classifier in enumerate(classifiers):
           
            hdr, vlu = My_csv_class().roc_gene(X, y,classifier,self.nfold,ni)
            val_list.append(vlu)

        rdf = pd.DataFrame(val_list, index=("LR", "GNB", "KNB", "DT", "SV"), columns = hdr)
        
        rdf.to_csv("result_summary.csv")
        
        
        rdf.plot(kind='bar',label="Result Summary")

        pl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, prop={'size':12},ncol=5, mode="expand", borderaxespad=0.)
        
        pl.savefig("Result_summarty.jpg")
        pl.show()
        pl.close()
        

if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", "--filepath",
                        required=True,
                        default=None,
                        help="Path to target CSV file")
                        
    parser.add_argument("-n", "--n_folds",
                        required=None,
                        default=5,
                        help="n_folds for Cross Validation")
                        
    args = parser.parse_args()

    a = My_csv_class()
    
    a.main_pro(args.filepath,int(args.n_folds))

    

