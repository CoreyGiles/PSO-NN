## 3 input 3 output NN
## data structure row = samples, col = var
In.Data<-matrix(0,nrow=dim(iris),ncol=4)
In.Data[,1]<-iris$Sepal.Length
In.Data[,2]<-iris$Sepal.Width
In.Data[,3]<-iris$Petal.Length
In.Data[,4]<-iris$Petal.Width
Out.Data<-matrix(0,nrow=dim(iris),ncol=3)
Out.Data[which(iris$Species=="setosa"),1]<-1
Out.Data[which(iris$Species=="versicolor"),2]<-1
Out.Data[which(iris$Species=="virginica"),3]<-1
In.Data<-scale(In.Data)

#data<-matrix(0, nrow=nSamples, ncol=nInput)
Train.split<-sample(1:nrow(In.Data),round(0.8*nrow(In.Data)),replace=F)
Test<-setdiff(1:nrow(In.Data),Train.split)
data<-In.Data[Train.split,]
data.out<-Out.Data[Train.split,]

### Find the NN with the lowest MSE
GetBest<-function(net) {
  n<-net$number
  best.MSE<-net$Best$MSE
  best.i<-net$Best$i
  for(i in 1:n) {
    if(eval(parse(text=paste("net$Net",i,"$MSE",sep="")))<best.MSE) {
      best.MSE<-eval(parse(text=paste("net$Net",i,"$MSE",sep="")))
      best.i<-i
    }
  }
  print(c(best.MSE,best.i))
}

## NN structure
##   first column is the weights for each input for a given neuron
##   second column is the weights for each input for the next neuron
##   The last row is the bias weight. It gets multiplied by the appended (1)


## append a numeric (1) column for bias
data.append<-cbind(data,rep(1,nrow(data)))
nInput<-ncol(data)
nOutputs<-ncol(data.out)
nNeuronsL1<-6
nNeuronsL2<-nOutputs

FullNet<-list()
FullNet$number<-200
FullNet$Best$MSE<-1E20
FullNet$Best$i<-0
FullNet$Best$NNL1<-matrix(0,nrow=ncol(data.append),ncol=nNeuronsL1+1)
FullNet$Best$NNL2<-matrix(0,nrow=ncol(FullNet$Best$NNL1),ncol=nOutputs)
for(i in 1:FullNet$number) {
  NNL1<-matrix(3*rnorm(ncol(data.append)*(nNeuronsL1+1)),nrow=ncol(data.append),ncol=nNeuronsL1+1)
  NNL1[,ncol(NNL1)]<-c(rep(0,nrow(NNL1)-1),10)
  NNL2<-matrix(3*rnorm(ncol(NNL1)*nNeuronsL2),nrow=ncol(NNL1),ncol=nNeuronsL2)
  
  vNNL1<-matrix(0,nrow=ncol(data.append),ncol=nNeuronsL1+1)
  vNNL2<-matrix(0,nrow=ncol(NNL1),ncol=nNeuronsL2)
  
  NNL1.val<-data.append%*%NNL1
  NNL1.val<-tanh(NNL1.val)
  
  NNL2.val<-NNL1.val%*%NNL2
  
  output<-1/(1+exp(NNL2.val))
  
  MSE<-sum((output-data.out)^2)
  
  eval(parse(text=paste("FullNet$Net",i,"$NNL1<-NNL1",sep="")))
  eval(parse(text=paste("FullNet$Net",i,"$NNL2<-NNL2",sep="")))
  eval(parse(text=paste("FullNet$Net",i,"$MSE<-MSE",sep="")))
  
  eval(parse(text=paste("FullNet$Net",i,"$pNNL1<-NNL1",sep="")))
  eval(parse(text=paste("FullNet$Net",i,"$pNNL2<-NNL2",sep="")))
  eval(parse(text=paste("FullNet$Net",i,"$pMSE<-MSE",sep="")))
  
  eval(parse(text=paste("FullNet$Net",i,"$vNNL1<-vNNL1",sep="")))
  eval(parse(text=paste("FullNet$Net",i,"$vNNL2<-vNNL2",sep="")))
}

PrimeNet<-function(net) {
  n<-net$number
  for(i in 1:n) {
    NNL1<-matrix(3*rnorm(ncol(data.append)*(nNeuronsL1+1)),nrow=ncol(data.append),ncol=nNeuronsL1+1)
    NNL1[,ncol(NNL1)]<-c(rep(0,nrow(NNL1)-1),10)
    NNL2<-matrix(3*rnorm(ncol(NNL1)*nNeuronsL2),nrow=ncol(NNL1),ncol=nNeuronsL2)
    
    NNL1.val<-data.append%*%NNL1
    NNL1.val<-tanh(NNL1.val)
    
    NNL2.val<-NNL1.val%*%NNL2
    
    output<-1/(1+exp(NNL2.val))
    
    MSE<-sum((output-data.out)^2)
    
    eval(parse(text=paste("net$Net",i,"$NNL1<-NNL1",sep="")))
    eval(parse(text=paste("net$Net",i,"$NNL2<-NNL2",sep="")))
    eval(parse(text=paste("net$Net",i,"$MSE<-MSE",sep="")))
    
    if(MSE < eval(parse(text=paste("net$Net",i,"$pMSE",sep="")))) {
      eval(parse(text=paste("net$Net",i,"$pMSE<-MSE",sep="")))
      eval(parse(text=paste("net$Net",i,"$pNNL1<-NNL1",sep="")))
      eval(parse(text=paste("net$Net",i,"$pNNL2<-NNL2",sep="")))
    }
    
    if(MSE < net$Best$MSE) {
      net$Best$MSE<-MSE
      net$Best$NNL1<-NNL1
      net$Best$NNL2<-NNL2
      net$Best$i<-i
    }
  }
  return(net)
}

LearnNet<-function(net) {
  n<-net$number
  bNNL1<-net$Best$NNL1
  bNNL2<-net$Best$NNL2
  bMSE<-net$Best$MSE
  for(i in 1:n) {
    eval(parse(text=paste("vNNL1<-net$Net",i,"$vNNL1",sep="")))
    eval(parse(text=paste("vNNL2<-net$Net",i,"$vNNL2",sep="")))
    eval(parse(text=paste("pNNL1<-net$Net",i,"$pNNL1",sep="")))
    eval(parse(text=paste("pNNL2<-net$Net",i,"$pNNL2",sep="")))
    eval(parse(text=paste("NNL1<-net$Net",i,"$NNL1",sep="")))
    eval(parse(text=paste("NNL2<-net$Net",i,"$NNL2",sep="")))
    vNNL1<-0.7*vNNL1 + runif(1)*1.4*(bNNL1-NNL1) + runif(1)*1.4*(pNNL1-pNNL1)
    vNNL2<-0.7*vNNL2 + runif(1)*1.4*(bNNL2-NNL2) + runif(1)*1.4*(pNNL2-pNNL2)
    NNL1<-NNL1+vNNL1
    NNL2<-NNL2+vNNL2
    eval(parse(text=paste("net$Net",i,"$vNNL1<-vNNL1",sep="")))
    eval(parse(text=paste("net$Net",i,"$vNNL2<-vNNL2",sep="")))
    eval(parse(text=paste("net$Net",i,"$NNL1<-NNL1",sep="")))
    eval(parse(text=paste("net$Net",i,"$NNL2<-NNL2",sep="")))

    MSE<-EvalNet(eval(parse(text=paste("net$Net",i,sep=""))))
    eval(parse(text=paste("net$Net",i,"$MSE<-MSE",sep="")))
    
    if(MSE<eval(parse(text=paste("net$Net",i,"$pMSE",sep="")))) {
      eval(parse(text=paste("net$Net",i,"$pNNL1<-NNL1",sep="")))
      eval(parse(text=paste("net$Net",i,"$pNNL2<-NNL2",sep="")))
      eval(parse(text=paste("net$Net",i,"$pNNL2<-MSE",sep="")))  
    }
    if(MSE<bMSE) {
      net$Best$NNL1<-NNL1
      net$Best$NNL2<-NNL2
      net$Best$MSE<-MSE
    }
  }
  return(net)
}

EvalNet<-function(net) {
  NNL1.val<-data.append%*%net$NNL1
  NNL1.val<-tanh(NNL1.val)

  NNL2.val<-NNL1.val%*%net$NNL2

  output<-1/(1+exp(NNL2.val))

  MSE<-sum((output-data.out)^2)
  return(MSE)
}

NetPerf<-function(net,selected=NULL) {
  if(is.null(selected)) {
  n<-net$number
  MSE.all<-numeric(n)
  for(i in 1:n) {
    MSE.all[i]<-eval(parse(text=paste("net$Net",i,"$MSE",sep="")))
  }
  return(MSE.all)
  } else if(is.numeric(selected)) {
    return(eval(parse(text=paste("net$Net",selected,"$MSE",sep=""))))
  }
}

NetPredict<-function(net,data.in,data.out) {
  data.append<-cbind(data.in,rep(1,nrow(data.in)))
  NNL1.val<-data.append%*%net$NNL1
  NNL1.val<-tanh(NNL1.val)
  
  NNL2.val<-NNL1.val%*%net$NNL2
  
  output<-1/(1+exp(NNL2.val))
  
  return(output)
}

GetBest(FullNet)
for(i in 1:50) {
  FullNet<-PrimeNet(FullNet)
}

for(i in 1:100){
  FullNet<-LearnNet(FullNet)
}
GetBest(FullNet)
hist(NetPerf(FullNet),breaks=50)

Predicted<-NetPredict(FullNet$Net151,In.Data[Test,],Out.Data[Test,])
round(Predicted,1)-Out.Data[Test,]


GetBest(FullNet)
for(n in 1:5) {
  FullNet<-PrimeNet(FullNet)
  for(i in 1:30){
    FullNet<-LearnNet(FullNet)
  }
}
GetBest(FullNet)
hist(NetPerf(FullNet),breaks=50)

