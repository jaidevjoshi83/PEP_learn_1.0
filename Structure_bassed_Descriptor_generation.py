print     """
        ################################################################
        #                                                              #
        #         Python Script peptide based Des. Gen.                #
        #                                                              #
        ################################################################
        #    Written By: Jayadev Joshi, ZebraFish lab, INMAS, DRDO.    #
        #                                                              #
        #      Before processing your data make sure that:             #
        #                                                              #
        #        1.  peptide sequence present in separate line.        #
        #                                                              #
        #                                                              #
        ################################################################
        #  USES:                                                       #
        #    Ex.  python mySript.py -p peptide.txt -d des_out.csv      #
        #                                                              #
        ################################################################


"""


import tempfile
import shutil
from progress.bar import Bar
import pandas as pd
import shutil
import sys
from pydpi.drug import *
from pydpi.pydrug import Chem
import glob
import openbabel
import fragbuilder
from pydpi.pydrug import *
from fragbuilder import Peptide
import os

class Str_DS_class(object):


    def read_pep_file(self, pep_infile):
        
        self.pep_infile = pep_infile
        df = pd.read_csv(self.pep_infile)
        list_pep_name = []
        list_class_label = []
        list_pep_name = df[df.columns[0]]
        list_class_label = df[df.columns[1]]
        return list_pep_name,list_class_label

    def structure_gen(self, pep_seq):

        if not os.path.exists("strs"):
            os.makedirs("strs")
        
        print "Structure being Generated !"

        b = len(pep_seq)

        bar = Bar('Processing PEPTIDES',fill='>', max=b)
    
        for seq in pep_seq:

            try:
                
                pep = Peptide(seq, nterm = "charged", cterm = "neutral")
                pep.regularize()
                pep.write_pdb(os.path.join("./strs", seq+".pdb"))
                bar.next()
                obConversion = openbabel.OBConversion()
                obConversion.SetInAndOutFormats("pdb", "sdf")
                mol = openbabel.OBMol()
                obConversion.ReadFile(mol, os.path.join("./strs", seq+".pdb"))  
                mol.AddHydrogens()       
                obConversion.WriteFile(mol, os.path.join("./strs",seq+".sdf"))
            
            except:

                pass

        bar.finish()

        print "Structure Generation Finished !"
        
    def new_Des_gene(self, mol):
    
        sDR1 = constitution.GetConstitutional(mol)
        sDR2 = topology.GetTopology(mol)
        sDR3 = connectivity.GetConnectivity(mol)
        sDR4 = molproperty.GetMolecularProperty(mol)
        sDR5 = kappa.GetKappa(mol)
        sDR6 = charge.GetCharge(mol)
    
        sDS_ALL = {}
    
        for sDS in (sDR1, sDR2, sDR3, sDR4, sDR5, sDR6):
        
            sDS_ALL.update(sDS)
        
        return sDS_ALL
    

    def main_process(self,str_pep_file,str_des_out):
          
        self.str_pep_file = str_pep_file
        self.str_des_out = str_des_out
        
        my_pep,list_class_label = Str_DS_class().read_pep_file(self.str_pep_file) 
        Str_DS_class().structure_gen(my_pep)
    
        sValues = []

        sdf_names = glob.glob(os.path.join("./strs",'*.sdf'))
        n = len(sdf_names)
        #print sdf_names
        print "Descriptors being calculated !"
    
        bar2 = Bar('Processing SDF file',fill='>', max=n)
    
        for sdn in my_pep:
            mols = Chem.SDMolSupplier(os.path.join("./strs", sdn+".sdf"))
            #mols = Chem.SDMolSupplier(sdn)
            for mol in mols:
                sValue = Str_DS_class().new_Des_gene(mol)
                sValues.append(sValue)
                bar2.next()
        sDF1 = pd.DataFrame(sValues)
        #sDF1.to_csv(self.str_des_out, index = False)
        bar2.finish()
      
        print "Descripor calculation has been finished "
        
        return sDF1,list_class_label

  
 

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
    Str_DS_class().main_process(args.pep, args.DesOut)
   
    print " Structure Generation and Discriptor Calculation finished successfully"
    
    
   
    
