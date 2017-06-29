import fragbuilder

from fragbuilder import Peptide
import pandas as pd
from progress.bar import Bar


from pydpi.pypro import PyPro

class DC_CLASS(object):
	


    def read_pep_file(self, pep_infile):
        
        self.pep_infile = pep_infile
        
        df = pd.read_csv(self.pep_infile)
        #list_pep_name = list(data.pep_name) #old
        list_pep_name = []
        list_class_label = []
        list_pep_name = df[df.columns[0]]
        list_class_label = df[df.columns[1]]
        return list_pep_name,list_class_label

    def Decriptor_generator(self, ps):

        protein = PyPro()
        protein.ReadProteinSequence(ps)
        moran = protein.GetPAAC(lamda=5,weight=0.5)
        DS_1 = protein.GetAPAAC(lamda=5,weight=0.5)
        DS_2 = protein.GetCTD()
        DS_3 = protein.GetDPComp()
        DS_4 = protein.GetGearyAuto()
        DS_5 = protein.GetMoranAuto()
        DS_6 = protein.GetMoreauBrotoAuto()
        DS_7 = protein.GetQSO()
        DS_8 = protein.GetSOCN()
        DS_9 = protein.GetTPComp()

        DS_ALL = {}

        for DS in (DS_1,DS_2,DS_3,DS_4,DS_5,DS_6,DS_7,DS_8,DS_9,moran):
            DS_ALL.update(DS)
            
        return DS_ALL

   

    def main_process(self,inf,outf):

        self.inf = inf
        self.outf = outf
        
        seql,class_label = DC_CLASS().read_pep_file(self.inf)
        b = len(seql)
        print "Sequence Based Descriptor Being Calculated !"
        
        bar = Bar('Processing PEPTIDES',fill='>', max=b)
        values = []
        
        for seq in seql:
            value = DC_CLASS().Decriptor_generator(seq)
            values.append(value)
            bar.next()
            
        bar.finish()
        df1 = pd.DataFrame(values)
        #df1.to_csv(self.outf, index = False)
        return df1,class_label

    

if __name__=="__main__":



    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--pep",
                        required=True,
                        default=None,
                        help="pep file")
                        
    parser.add_argument("-d", "--DesOut",
                        required=True,
                        default=None,
                        help="out put file name for str Descriptors")    
                                               
    args = parser.parse_args()
    DC_CLASS().main_process(args.pep,args.DesOut)
  

    print "Discriptor Calculation finished successfully"


