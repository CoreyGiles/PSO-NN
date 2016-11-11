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
      net$Best$i<-i
    }
  }
  print(c(best.MSE,best.i))
}

BuildNet<-function(Input,Output,nNeurons,nParticles) {
  data.append<-cbind(Input,rep(1,nrow(Input)))
  nInput<-ncol(Input)
  nOutputs<-ncol(Output)
  nNeuronsL1<-nNeurons
  nNeuronsL2<-nOutputs
  
  FullNet<-list()
  FullNet$data<-data.append
  FullNet$data.out<-Output
  FullNet$number<-nParticles
  FullNet$Best$MSE<-1E20
  FullNet$Best$i<-0
  FullNet$Best$NNL1<-matrix(0,nrow=ncol(data.append),ncol=nNeuronsL1+1)
  FullNet$Best$NNL2<-matrix(0,nrow=ncol(FullNet$Best$NNL1),ncol=nOutputs)
  for(i in 1:FullNet$number) {
    NNL1<-matrix(3*rnorm(ncol(data.append)*(nNeuronsL1+1)),nrow=ncol(data.append),ncol=nNeuronsL1+1)
    NNL1[,ncol(NNL1)]<-c(rep(0,nrow(NNL1)-1),10)
    NNL2<-matrix(3*rnorm(ncol(NNL1)*nNeuronsL2),nrow=ncol(NNL1),ncol=nNeuronsL2)
    
    vNNL1<-0.1*NNL1
    vNNL1[nrow(vNNL1),ncol(vNNL1)]<-0
    vNNL2<-0.1*NNL2
    
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
  return(FullNet)
}

PrimeNet<-function(net) {
  n<-net$number
  for(i in 1:n) {
    NNL1<-matrix(3*rnorm(nrow(net$Best$NNL1)*ncol(net$Best$NNL1)),nrow=nrow(net$Best$NNL1),ncol=ncol(net$Best$NNL1))
    NNL1[,ncol(NNL1)]<-c(rep(0,nrow(NNL1)-1),10)
    NNL2<-matrix(3*rnorm(nrow(net$Best$NNL2)*ncol(net$Best$NNL2)),nrow=nrow(net$Best$NNL2),ncol=ncol(net$Best$NNL2))
    
    NNL1.val<-net$data%*%NNL1
    NNL1.val<-tanh(NNL1.val)
    
    NNL2.val<-NNL1.val%*%NNL2
    
    output<-1/(1+exp(NNL2.val))
    
    MSE<-sum((output-net$data.out)^2)
    
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
    vNNL1[vNNL1>2]<-2
    vNNL1[vNNL1<(-2)]<-(-2)
    vNNL2[vNNL2>10]<-10
    vNNL2[vNNL2<(-10)]<-(-10)
    NNL1<-NNL1+vNNL1
    NNL2<-NNL2+vNNL2
    eval(parse(text=paste("net$Net",i,"$vNNL1<-vNNL1",sep="")))
    eval(parse(text=paste("net$Net",i,"$vNNL2<-vNNL2",sep="")))
    eval(parse(text=paste("net$Net",i,"$NNL1<-NNL1",sep="")))
    eval(parse(text=paste("net$Net",i,"$NNL2<-NNL2",sep="")))

    MSE<-EvalNet(net,i)
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

EvalNet<-function(net,selected) {
  NNL1.val<-net$data%*%eval(parse(text=paste("net$Net",selected,"$NNL1",sep="")))
  NNL1.val<-tanh(NNL1.val)

  NNL2.val<-NNL1.val%*%eval(parse(text=paste("net$Net",selected,"$NNL2",sep="")))

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

NetPredict<-function(net,data.in) {
  data.append<-cbind(data.in,rep(1,nrow(data.in)))
  NNL1.val<-data.append%*%net$NNL1
  NNL1.val<-tanh(NNL1.val)
  
  NNL2.val<-NNL1.val%*%net$NNL2
  
  output<-1/(1+exp(NNL2.val))
  
  return(output)
}



##############################################################################
FullNet<-BuildNet(data,data.out,4,100)
GetBest(FullNet)
hist(NetPerf(FullNet),breaks=50)
for(i in 1:50) {
  FullNet<-LearnNet(FullNet)
}
GetBest(FullNet)
hist(NetPerf(FullNet),breaks=50)

Predicted<-NetPredict(FullNet$Net50,In.Data[Test,])
round(Predicted,1)-Out.Data[Test,]

for(i in 1:50) {
  FullNet<-PrimeNet(FullNet)
}

for(i in 1:10){
  FullNet<-LearnNet(FullNet)
}
GetBest(FullNet)
hist(NetPerf(FullNet),breaks=50)

Predicted<-NetPredict(FullNet$Net154,In.Data[Test,])
round(Predicted,1)-Out.Data[Test,]


GetBest(FullNet)
for(n in 1:3) {
  FullNet<-PrimeNet(FullNet)
  for(i in 1:50){
    FullNet<-LearnNet(FullNet)
  }
}
GetBest(FullNet)
hist(NetPerf(FullNet),breaks=50)

