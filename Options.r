  #############################################
 #                                             #
   # R script to preprocess  Descriptor Data #
 #                                             #
  #############################################


## library imported 

library(mlbench)
library(caret)
library(corrplot)

# data from the file 
options <- commandArgs(trailingOnly = TRUE)


dat = read.csv(options[1], header = TRUE)

# intial column count 

print ("intitial columns")
print (ncol(dat))

# zero std column were removed 
nd = Filter(sd, dat)

#nd = dat[,colSums(dat != 0)> 0]

#Column count after removing zero value columns 

print ("After removing zero std value columsn")
print  (ncol(nd))


# Data normalization (zero mean, unit varience)

x =  ncol(nd)
preObj <- preProcess(nd[,2:(x-1) ], method=c("center", "scale"))
normalized_Data <- predict(preObj, nd[,2:(x-1)])

#z = ncol(normalized_Data)
#print (colMdaeans(normalized_Data[2:(z-1)]))

# correlation matrix 

y = ncol(normalized_Data)
m = cor(normalized_Data[,2:(y-1)])


#corelation plot 
#corrplot(m, method = "circle")

#removes Highly correlated columns  

hc = findCorrelation(m, cutoff=0.8)
new_hc = sort(hc)
new_dat = normalized_Data[,-c(new_hc)]

#column count, Aftre removing the correlated columns . 


#z = ncol(new_dat)
#print (colMeans(new_dat[2:(z-1)]))



print ("after removing correlated columns")
print (ncol(new_dat))

#jai = ncol(new_dat)

# final data written to the csv file. 

write.csv(new_dat, file = options[2], row.names = FALSE)

