rm(list = ls())
# Load R-packages
library(e1071) # svm
library(rpart)
library(cvTools) # CV


# ------------------------------------------
#                 Exercise 1             
# ------------------------------------------

# Pen and paper exercise

# ------------------------------------------
#                 Exercise 2             
# ------------------------------------------

# ------------------a)----------------------

# Load data
Tb=read.csv('Data/Synthetic2DNoOverlapp.csv')
colnames(Tb)=c("X1","X2","Y")
X= as.matrix(Tb[,c(1:2)])
Tb$Y=as.factor(Tb$Y)
Y=Tb$Y

#Compute optimal seperating hyperplanes using the svm() function
svm.model = svm.model <- svm(Y ~ .,
                             data=Tb,
                             type="C-classification",
                             degree=# parameter needed if the kernel is polynomial (default: 3),
                             scale=TRUE,
                             kernel=#Try different kernels,
                             gamma=#parameter needed for all types of kernels except linear,
                             cost=10000)

# It is possible to use the in-built plot for svm's "plot(svm.model,data=Tb)",
# however this plot is not that clear. 
# Instead we create the plot ourselves. 
# The red area corresponds to "1", the black area to "0". 

# Create a mesh grid
n=50
grange = apply(X, 2, range)
x1 = seq(from = grange[1, 1], to = grange[2, 1], length = n)
x2 = seq(from = grange[1, 2], to = grange[2, 2], length = n)
grid=expand.grid(X1 = x1, X2 = x2)
Ygridd = predict(svm.model, grid)
#find decision buondary
dec=predict(svm.model, grid,decision.values = T)
ZZ=as.vector(attributes(dec)$decision)
# Plot
{
  plot(grid, col = c( "black","red")[as.numeric(Ygridd)], pch = 20, cex = 0.2)
  points(X, col = Y , pch = 19)
  points(X[svm.model$index, ], pch = 5, cex = 2)#supportvectors
  contour(x1,x2,matrix(ZZ,length(x1),length(x2)),level=0,lwd=2,drawlabels=F,add=T)#decision boundary
}

# ------------------b)----------------------

# Load data
Data1=read.csv('Data/Synthetic2DOverlap.csv')
d=data.frame(x1=Data1[,1],x2=Data1[,2],y=as.factor(Data1[,3]))
X = d[, 1:2]
Y = as.factor(d$y)

#Fit svm
svm.model <- svm(y ~ .,data=d,
                 type="C-classification",
                 kernel='radial',
                 gamma=# choose kernel parameter,
                 scale = TRUE,
                 cost=10000) 

# Plot result, create mesh grid
n=50
grange = apply(X, 2, range)
x1 = seq(from = grange[1, 1], to = grange[2, 1], length = n)
x2 = seq(from = grange[1, 2], to = grange[2, 2], length = n)
grid=expand.grid(x1=x1,x2=x2)
Ygridd = predict(svm.model, grid)
#find decision buondary
dec=predict(svm.model, grid,decision.values = T)
ZZ=as.vector(attributes(dec)$decision)
# Plot
{
  plot(grid, col = c( "black","red")[as.numeric(Ygridd)], pch = 20, cex = 0.2)
  points(X, col = Y , pch = 19)
  points(X[svm.model$index, ], pch = 5, cex = 2)#supportvectors
  contour(x1,x2,matrix(ZZ,length(x1),length(x2)),level=0,lwd=2,drawlabels=F,add=T)#decision boundary
}
  
# ------------------------------------------
#                 Exercise 3             
# ------------------------------------------  
  
# Load data
Tb = read.csv('Data/Ex3Data.csv',header=F);
colnames(Tb)=c("X1","X2","Y")
X=data.matrix(Tb[,1:2])
Tb[,3]=as.factor(Tb[,3])
Y=Tb[,3]

# Apply the support vector machine with the svm function:
svm.model=svm(Y~.,
              data =Tb,
              type="C-classification",
              kernel=# Kernel,
              gamma=# parameter needed for all kernels except linear,
              cost=# cost of constraints violation (default: 1),
              scale = TRUE,
              degree=# used when 'polynomial')

# Plot
n=50
grange = apply(X, 2, range)
x1 = seq(from = grange[1, 1], to = grange[2, 1], length = n)
x2 = seq(from = grange[1, 2], to = grange[2, 2], length = n)
grid=expand.grid(X1 = x1, X2 = x2)
Ygridd = predict(svm.model, grid)
#find decision buondary
dec=predict(svm.model, grid,decision.values = T)
ZZ=as.vector(attributes(dec)$decision)
# Plot
plot(grid, col = c( "black","red")[as.numeric(Ygridd)], pch = 20, cex = 0.2)
points(X, col = Y , pch = 19)
points(X[svm.model$index, ], pch = 5, cex = 2)#supportvectors
contour(x1,x2,matrix(ZZ,length(x1),length(x2)),level=0,lwd=2,drawlabels=F,add=T)#decision boundary
contour(x1,x2,matrix(ZZ,length(x1),length(x2)),level=1,lty=2,lwd=1.5,drawlabels=F,add=T)#supportvectors boundary
contour(x1,x2,matrix(ZZ,length(x1),length(x2)),level=-1,lty=2,lwd=1.5,drawlabels=F,add=T)#supportvectors boundary

# ------------------------------------------
#                 Exercise 4             
# ------------------------------------------ 

# Pen and paper exercise

# ------------------------------------------
#                 Exercise 5             
# ------------------------------------------

# Load data 
Tb = read.csv('Data/ACS.csv');
Tb$Gender=as.factor(Tb$Gender)

# Split data in train and test.
Itrain = Tb$Train==1;
Train=Tb[Itrain,-dim(Tb)[2]]
Test=Tb[!Itrain,-dim(Tb)[2]]
Y_train = Tb$Y[Itrain];
Y_test  = Tb$Y[!Itrain];
X_train = Tb[Itrain,1:(dim(Tb)[2]-2)];
X_test  = Tb[!Itrain,1:(dim(Tb)[2]-2)];


K = 10 # K-fold cross-validation
scale_vec = c() #Use a relatively small number of values to reduce the computational cost
cost_vec = c() #Use a relatively small number of values to reduce the computational cost

accuracy_array = array(NA,dim=c(length(cost_vec),length(scale_vec),K)); # prepare array for all the accuracy values
folds <- # Use cvFolds to create folds 
for (j in 1:K){ # K fold cross-validation
  tr =  # training data
  tst = # test data
  
  for (sc in 1:length(scale_vec)) {
    for (ct in 1:length(cost_vec)) {
      
      svm.model = svm(formula = Y ~ .,
                      data = tr,
                      type = 'C-classification',
                      kernel = 'radial',
                      gamma=scale_vec[sc],
                      cost=cost_vec[ct],
                      scale=TRUE)
      
      y_pred = predict() # Use the predict function
      accuracy_array[ct,sc,j] = # Compute accuracy
    }
  }
}
accuracy_array_mean = # calculate mean accuracy
idxmax = # find index with highest accuracy

# Fit svm with the optimal parameters
svm.finalmodel=svm(Y~.,data = Train,
                     type="C-classification",
                     kernel = "radial",
                     scale = TRUE,
                     gamma = scale_vec[idxmax[1,2]],
                     cost = cost_vec[idxmax[1,1]])
yhat2=predict() # Predict for the test set
accuracysvm = # Calculate accuracy
print(paste('Accuracy by SVM =',round(accuracysvm,4)))

#Logistic regression
B=# use the function glm() to fit a logistic regression

yhat= # use predict to get predictions
accuracylogreg = # compute the accuracy
print(paste('Accuracy by logistic regression =',round(accuracylogreg,4)))



