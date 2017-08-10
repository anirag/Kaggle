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


#############
## Load Data 
#############

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(data.table)
library(plyr)
library(plotly)
library(dplyr)
print(paste("Load Data",Sys.time()))
## load the training file, using just the fields available for test
train<-fread("train-3.csv"
             ,select = c("Semana","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID","Demanda_uni_equil","Agencia_ID"))
train[,target:=log1p(Demanda_uni_equil)]

train2 <- fread("train.csv")
train<-fread("train_6.csv",drop = c("Demanda_uni_equil","nProductCanal", "nProductRoute"    ,            
                                    "canalMeanLog"         ,         "productAgencymaxLog" ,                    "agentproductClientmaxLog" ,    
                                    "canalmaxLog"   ,                "routequantileLog"   ,           "routeMeanLog"  ,                "canalquantileLog"  ,           
                                    "productRoutemaxLog"   ,         "agencyMeanLog"    ,             "nProduct"     ,                 "nCanal" ,                      
                                    "agentproductClientquantileLog", "nAgency"      ,                              "agencymaxLog" , "routemaxLog"   ,"nRoute", "nCanalProductRoute"))
train2 = train2[Semana >= 6,]
df = merge(train,train2,by=c("Semana","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID","Agencia_ID","target"),all.x=TRUE)
rm(train2)
rm(train)
df[is.na(df)] = 0
test = fread("test.csv")
week_3_avg = train[Semana == 3,.(avgPC3=median(target)),.(Producto_ID,Cliente_ID)]
Sys.time()
week_4_avg = train[Semana == 4,.(avgPC4=median(target)),.(Producto_ID,Cliente_ID)]
Sys.time()
week_5_avg = train[Semana == 5,.(avgPC5=median(target)),.(Producto_ID,Cliente_ID)]
Sys.time()
week_6_avg = train[Semana == 6,.(avgPC6=median(target)),.(Producto_ID,Cliente_ID)]
Sys.time()
week_7_avg = train[Semana == 7,.(avgPC7=median(target)),.(Producto_ID,Cliente_ID)]
Sys.time()
week_8_avg = train[Semana == 8,.(avgPC8=median(target)),.(Producto_ID,Cliente_ID)]
Sys.time()
week_9_avg = train[Semana == 9,.(avgPC9=median(target)),.(Producto_ID,Cliente_ID)]
Sys.time()
week = vector("list",2)
j=1
for(i in 10:11){
  a = test[Semana == i,]
  print(nrow(a))
  a = merge(a,eval(parse(text = paste0("week_",(i-3),"_avg"))),by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
  print(nrow(a))
  a = merge(a,eval(parse(text = paste0("week_",(i-2),"_avg"))),by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
  print(nrow(a))
  setnames(a,paste0("avgPC",(i-3)),"Lag3")
  setnames(a,paste0("avgPC",(i-2)),"Lag2")
  a$Lag1 = a$Lag2
  week[[j]] = a
  j=j+1
}
test = merge(test,extra,by=c("id","Semana","Agencia_ID","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID"),all.x=T)
week = vector("list",2)
j=1
for(i in 8:9){
  a = test[Semana == i,]
  print(nrow(a))
  a = merge(a,eval(parse(text = paste0("week_",(i-3),"_avg"))),by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
  print(nrow(a))
  a = merge(a,eval(parse(text = paste0("week_",(i-2),"_avg"))),by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
  print(nrow(a))
  a = merge(a,eval(parse(text = paste0("week_",(i-1),"_avg"))),by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
  print(nrow(a))
  setnames(a,paste0("avgPC",(i-3)),"Lag3")
  setnames(a,paste0("avgPC",(i-2)),"Lag2")
  setnames(a,paste0("avgPC",(i-1)),"Lag1")
  week[[j]] = a
  j=j+1
}


extra = do.call("rbind",week)
train = train[Semana >= 8,]
train$nLag3 = 0
train$avgLag3 = 0
train$nLag2 = 0
train$avgLag2 = 0
train$nLag1 = 0
train$avgLag1 = 0
train2 = rbind(train,extra)
fwrite(train2,"train.csv")
Sys.time()
productInfo<-train[Semana < 8,.(productMeanLog=mean(target),productmaxLog=max(target),productquantileLog=quantile(target,0.95)),Producto_ID]
Sys.time()
clientInfo<-train[Semana < 8,.(nClient=.N,clientMeanLog=mean(target),clientmaxLog=max(target),clientquantileLog=quantile(target,0.95)),Cliente_ID]
Sys.time()
agencyInfo<-train[Semana < 8,.(agencyquantileLog=quantile(target,0.95)),Agencia_ID]
Sys.time()
productClientInfo<-train[Semana < 8,.(nProductClient=.N,productClientMeanLog=mean(target),productClientmaxLog=max(target),productClientquantileLog=quantile(target,0.95)),.(Producto_ID,Cliente_ID)]
Sys.time()
productRouteInfo<-train[Semana < 8,.(nProductRoute=.N,productRouteMeanLog=mean(target),productRoutequantileLog=quantile(target,0.95)),.(Producto_ID,Ruta_SAK)]
Sys.time()
productAgencyInfo<-train[Semana < 8,.(nProductAgency=.N,productAgencyMeanLog=mean(target),productAgencyquantileLog=quantile(target,0.95)),.(Producto_ID,Agencia_ID)]
Sys.time()
productCanalInfo<-train[Semana < 8,.(nProductCanal=.N,productCanalMeanLog=mean(target),productCanalmaxLog=max(target),productCanalquantileLog=quantile(target,0.95)),.(Producto_ID,Canal_ID)]
Sys.time()
agentproductClientInfo<-train[Semana < 8,.(nAgentProductClient=.N,agentproductClientMeanLog=mean(target)),.(Agencia_ID,Producto_ID,Cliente_ID)]
Sys.time()
canalproductRouteInfo<-train[Semana < 6,.(canalproductRouteMeanLog=mean(target),canalproductRoutemaxLog=max(target),canalproductRoutequantileLog=quantile(target,0.95)),.(Canal_ID,Producto_ID,Ruta_SAK)]
Sys.time()
agentrouteClientInfo<-train[Semana < 6,.(nAgentRouteClient=.N,agentrouteClientMeanLog=mean(target),agentrouteClientmaxLog=max(target),agentrouteClientquantileLog=quantile(target,0.95)),.(Agencia_ID,Ruta_SAK,Cliente_ID)]
Sys.time()


## remove all weeks used for creating the averages, and create a modeling set with the remaining data
train<-train[Semana >= 6,]
## now add the features we have created from weeks 3-7 to weeks 8 and 9
train<-merge(train,productInfo,by="Producto_ID",all.x=TRUE)
Sys.time()
train<-merge(train,clientInfo,by="Cliente_ID",all.x=TRUE)
Sys.time()
train<-merge(train,agencyInfo,by="Agencia_ID",all.x=TRUE)
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
train<-merge(train,train2,by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
Sys.time()

train[is.na(train)] = 0
head(train)


client_table = fread("cliente_tabla.csv",header = T,sep = ",")
client_table = client_table %>% distinct(Cliente_ID)
product_table = fread("example_cluster.csv",header = T,sep = ",")

train = fread("train_final.csv")
train = merge(train,product_table,by = "Producto_ID")
train = train[,product_name:=NULL]
train = train[,has_choco:=NULL]
train = train[,product_shortname:=NULL]
train = train[,has_multigrain:=NULL]
train = train[,has_vanilla:=NULL]

train_set = train[Semana <=8 ,]
val_set = train[Semana == 9, ]
rm(train)
train_set[is.na(train_set)] = 0
val_set[is.na(val_set)] = 0

#clean colnames
colnames(train) <- iconv(colnames(train), to='ASCII', sub='')
fwrite(train[Semana==8,],"train_val.csv")
fwrite(train[Semana==9,],"test_val.csv")


#################
## Set up Cluster (H2O is a Java ML Platform, with R/Python/Web/Java/Spark/Hadoop APIs)
#################
h2o.init(nthreads=-1,max_mem_size = '350G')
dev<-as.h2o(train[Semana==8,],destination_frame = "dev.hex")
val<-as.h2o(train[Semana==9,],destination_frame = "val.hex")
rm(dev1)


##############################
## Model: Product Groups & GBM
##############################
print(paste("Model: Product Groups & GBM",Sys.time()))
## train a GBM; use aggressive parameters to keep overall runtime within 20 minutes
## this model is fit on Semana 7 and evaluated on Semana 8.
predictors<-colnames(dev)[!colnames(dev) %in% c("Demanda_uni_equil.x","target.x","Demanda_uni_equil.y","target.y","Demanda_uni_equil","target","Semana","has_choco","has_vanilla","has_multigrain","nProductCanal", "nProductRoute"    ,            
                                                "canalMeanLog"         ,         "productAgencymaxLog" ,          "Producto_ID"    ,               "agentproductClientmaxLog" ,    
                                                "canalmaxLog"   ,                "routequantileLog"   ,           "routeMeanLog"  ,                "canalquantileLog"  ,           
                                                "productRoutemaxLog"   ,         "agencyMeanLog"    ,             "nProduct"     ,                 "nCanal" ,                      
                                                "agentproductClientquantileLog", "nAgency"      ,                 "Cliente_ID"  ,                  "agencymaxLog"   ,              
                                                "Agencia_ID"            ,        "routemaxLog"   ,                "nRoute"      ,                  "nCanalProductRoute","product_name","product_shortname","Canal_ID","Ruta_SAK")]
g<-h2o.gbm(
  training_frame = dev,      ## H2O frame holding the training data
  validation_frame = val,  ## extra holdout piece for three layer modeling
  x=predictors,                 ## this can be names or column numbers
  y="target",                   ## target: using the logged variable created earlier
  model_id="gbm1",              ## internal H2O name for model
  ntrees = 45,                  ## use fewer trees than default (50) to speed up training
  learn_rate = 0.12,             ## lower learn_rate is better, but use high rate to offset few trees
  score_tree_interval = 3,      ## score every 3 trees
  sample_rate = 0.5,            ## use half the rows each scoring round
  col_sample_rate = 0.8,        ## use 4/5 the columns to decide each split decision
  keep_cross_validation_predictions = T,
  offset_column = "productClientMeanLog"
  ## an offset terms allows the model to have 
)

## look at model diagnostics
summary(g)
## specifically look at validation RMSE (sqrt of MSE)
(h2o.mse(g,valid=T))^0.5
View(h2o.varimp(g))
h2o.shutdown()
