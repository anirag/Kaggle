# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
if (! ("rjson" %in% rownames(installed.packages()))) { install.packages("rjson") }
if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/rel-shannon/26/R")))
library(h2o)

RMSLE = function(pred,target){
  return(sqrt(mean((log(pred + 1) - log(target + 1))^2)))
}

RMSE = function(pred,target){
  return(sqrt(mean((pred -target)^2)))
}


#############
## Load Data 
#############

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
#install.packages("data.table", repos = "https://Rdatatable.github.io/data.table", type = "source")
# Load package
library(data.table)

library(plyr)
library(plotly)
library(dplyr)
print(paste("Load Data",Sys.time()))
## load the training file, using just the fields available for test
train<-fread("train-3.csv"
             ,select = c("Semana","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID","Demanda_uni_equil","Agencia_ID"))
train[,target:=log1p(Demanda_uni_equil)]


Sys.time()
productInfo<-train[Semana < 8,.(nProduct=.N,productMeanLog=mean(target),productmaxLog=max(target),productquantileLog=quantile(target,0.95)),Producto_ID]
Sys.time()
clientInfo<-train[Semana < 8,.(nClient=.N,clientMeanLog=mean(target),clientmaxLog=max(target),clientquantileLog=quantile(target,0.95)),Cliente_ID]
Sys.time()
agencyInfo<-train[Semana < 8,.(nAgency=.N,agencyMeanLog=mean(target),agencymaxLog=max(target),agencyquantileLog=quantile(target,0.95)),Agencia_ID]
Sys.time()
canalInfo<-train[Semana < 8,.(nCanal=.N,canalMeanLog=mean(target),canalmaxLog=max(target),canalquantileLog=quantile(target,0.95)),Canal_ID]
Sys.time()
routeInfo<-train[Semana < 8,.(nRoute=.N,routeMeanLog=mean(target),routemaxLog=max(target),routequantileLog=quantile(target,0.95)),Ruta_SAK]
Sys.time()
productClientInfo<-train[Semana < 8,.(nProductClient=.N,productClientMeanLog=mean(target),productClientmaxLog=max(target),productClientquantileLog=quantile(target,0.95)),.(Producto_ID,Cliente_ID)]
Sys.time()
productRouteInfo<-train[Semana < 8,.(nProductRoute=.N,productRouteMeanLog=mean(target),productRoutemaxLog=max(target),productRoutequantileLog=quantile(target,0.95)),.(Producto_ID,Ruta_SAK)]
Sys.time()
productAgencyInfo<-train[Semana < 8,.(nProductAgency=.N,productAgencyMeanLog=mean(target),productAgencymaxLog=max(target),productAgencyquantileLog=quantile(target,0.95)),.(Producto_ID,Agencia_ID)]
Sys.time()
productCanalInfo<-train[Semana < 8,.(nProductCanal=.N,productCanalMeanLog=mean(target),productCanalmaxLog=max(target),productCanalquantileLog=quantile(target,0.95)),.(Producto_ID,Canal_ID)]
Sys.time()
agentproductClientInfo<-train[Semana < 8,.(nAgentProductClient=.N,agentproductClientMeanLog=mean(target),agentproductClientmaxLog=max(target),agentproductClientquantileLog=quantile(target,0.95)),.(Agencia_ID,Producto_ID,Cliente_ID)]
Sys.time()
canalproductRouteInfo<-train[Semana < 8,.(nCanalProductRoute=.N,canalproductRouteMeanLog=mean(target),canalproductRoutemaxLog=max(target),canalproductRoutequantileLog=quantile(target,0.95)),.(Canal_ID,Producto_ID,Ruta_SAK)]
Sys.time()
agentrouteClientInfo<-train[Semana < 8,.(nAgentRouteClient=.N,agentrouteClientMeanLog=mean(target),agentrouteClientmaxLog=max(target),agentrouteClientquantileLog=quantile(target,0.95)),.(Agencia_ID,Ruta_SAK,Cliente_ID)]
Sys.time()


## remove all weeks used for creating the averages, and create a modeling set with the remaining data
train<-train[Semana >= 8,]
## now add the features we have created from weeks 3-7 to weeks 8 and 9
train<-merge(train,productInfo,by="Producto_ID",all.x=TRUE)
Sys.time()
train<-merge(train,clientInfo,by="Cliente_ID",all.x=TRUE)
Sys.time()
train<-merge(train,agencyInfo,by="Agencia_ID",all.x=TRUE)
Sys.time()
train<-merge(train,canalInfo,by="Canal_ID",all.x=TRUE)
Sys.time()
train<-merge(train,routeInfo,by="Ruta_SAK",all.x=TRUE)
Sys.time()
train<-merge(train,productClientInfo,by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
Sys.time()
train<-merge(train,productRouteInfo,by=c("Ruta_SAK","Producto_ID"),all.x=TRUE)
Sys.time()
train<-merge(train,productAgencyInfo,by=c("Agencia_ID","Producto_ID"),all.x=TRUE)
Sys.time()
train<-merge(train,productCanalInfo,by=c("Canal_ID","Producto_ID"),all.x=TRUE)
Sys.time()
train<-merge(train,agentproductClientInfo,by=c("Agencia_ID","Cliente_ID","Producto_ID"),all.x=TRUE)
Sys.time()
train<-merge(train,canalproductRouteInfo,by=c("Canal_ID","Producto_ID","Ruta_SAK"),all.x=TRUE)
Sys.time()
train<-merge(train,agentrouteClientInfo,by=c("Agencia_ID","Cliente_ID","Ruta_SAK"),all.x=TRUE)
Sys.time()


client_table = fread("cliente_tabla.csv",header = T,sep = ",")
client_table = client_table %>% distinct(Cliente_ID)
product_table = fread("example_cluster.csv",header = T,sep = ",")
town_state = fread("town_state.csv",header = T,sep = ",")

train = merge(train,product_table,by = "Producto_ID")
train[is.na(train)] = 0 

#clean colnames
colnames(train) <- iconv(colnames(train), to='ASCII', sub='')

test<-fread("test.csv")
test[1:2,] ## take a look at a few rows of the test data

## merge in the offset column, just as with val and final
## now add the features we have created from weeks 3-7 to weeks 8 and 9
test<-merge(test,productInfo,by="Producto_ID",all.x=TRUE)
Sys.time()
test<-merge(test,clientInfo,by="Cliente_ID",all.x=TRUE)
Sys.time()
test<-merge(test,agencyInfo,by="Agencia_ID",all.x=TRUE)
Sys.time()
test<-merge(test,canalInfo,by="Canal_ID",all.x=TRUE)
Sys.time()
test<-merge(test,routeInfo,by="Ruta_SAK",all.x=TRUE)
Sys.time()
test<-merge(test,productClientInfo,by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
Sys.time()
test<-merge(test,productRouteInfo,by=c("Ruta_SAK","Producto_ID"),all.x=TRUE)
Sys.time()
test<-merge(test,productAgencyInfo,by=c("Agencia_ID","Producto_ID"),all.x=TRUE)
Sys.time()
test<-merge(test,productCanalInfo,by=c("Canal_ID","Producto_ID"),all.x=TRUE)
Sys.time()
test<-merge(test,agentproductClientInfo,by=c("Agencia_ID","Cliente_ID","Producto_ID"),all.x=TRUE)
Sys.time()
test<-merge(test,canalproductRouteInfo,by=c("Canal_ID","Producto_ID","Ruta_SAK"),all.x=TRUE)
Sys.time()
test<-merge(test,agentrouteClientInfo,by=c("Agencia_ID","Cliente_ID","Ruta_SAK"),all.x=TRUE)
Sys.time()

test = merge(test,product_table,by = "Producto_ID")
colnames(test) <- iconv(colnames(test), to='ASCII', sub='')
test[is.na(test)] = 0

fwrite(train,"train_final.csv")
fwrite(train[Semana==8,],"train_val.csv")
fwrite(train[Semana==9,],"test_val.csv")
fwrite(test,"test_final.csv")
#################
## Set up Cluster (H2O is a Java ML Platform, with R/Python/Web/Java/Spark/Hadoop APIs)
#################
h2o.init(nthreads=-1,max_mem_size = '350G')

dev<-h2o.importFile(path = trainpath, destination_frame = "dev.hex")
val<-h2o.importFile(path = valpath, destination_frame = "val.hex")

testHex<-h2o.importFile(path = testpath, destination_frame = "test.hex")

dev[1:2,]

##############################
## Model: Product Groups & GBM
##############################
print(paste("Model: Product Groups & GBM",Sys.time()))
## train a GBM; 
## this model is fit on Semana 7 and evaluated on Semana 8.
predictors<-colnames(dev)[!colnames(dev) %in% c("target","Semana")]
g<-h2o.gbm(
  training_frame = dev,
  validation_frame = val,## H2O frame holding the training data ## extra holdout piece for three layer modeling
  x=predictors,                 ## this can be names or column numbers
  y="target",                   ## target: using the logged variable created earlier
  model_id="gbm1",              ## internal H2O name for model
  ntrees = 400,                  ## use fewer trees than default (50) to speed up training
  learn_rate = 0.09,             ## lower learn_rate is better, but use high rate to offset few trees     ## score every 3 trees
  sample_rate = 0.55,            ## use half the rows each scoring round
  col_sample_rate = 0.7
  ## an offset terms allows the model to have 
)

## look at model diagnostics
summary(g)
## specifically look at validation RMSE (sqrt of MSE)
(h2o.mse(g,valid=T))^0.5
rm(dev)
rm(val)

test = fread("test_with_cnames.csv",select = c("id"))

#####################
## Create Predictions
#####################
print(paste("Create Predictions",Sys.time()))
cols = c("target")
testHex = h2o.removeVecs(testHex, cols)
p<-as.data.frame(h2o.predict(g,testHex))[,1]
p<-pmax(0,exp(p)-1)
summary(p)

####################
## Create Submission
####################
print(paste("Create Submission",Sys.time()))
submissionFrame<-cbind.data.frame(test$id,p)
colnames(submissionFrame)<-c("id","Demanda_uni_equil")
write.csv(submissionFrame,"h2o_gbm_train_8_9_30rounds_lag.csv",row.names=F)  ## export submission
str(submissionFrame)