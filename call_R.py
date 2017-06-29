import rpy2.robjects as robjects
import subprocess
import os


class Run_r(object):
    
    
    def main(self,inf,ofile):
         
        self.inf = inf
        self.ofile = ofile
        dirname, filename = os.path.split(os.path.abspath(__file__))
        os.environ["x"] = self.inf
        os.environ["y"] = self.ofile
        os.environ["Dp"] = dirname
        
        #print "running from", dirname
        #print "file is", filename
        os.system("echo $Dp/Options.r") 
        os.system("Rscript $Dp/Options.r $x $y ")  
        
        os.system("exit")      

       
if __name__=="__main__":
    
    r = Run_r()
    r.main("m.csv","jai.csv")


