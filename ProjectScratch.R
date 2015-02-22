library(caret)
library(plyr)
library(RANN)
raw_train    <- read.csv('pml-training.csv')
validation  <- read.csv('pml-testing.csv')

set.seed(78910)
train_idx <- createDataPartition(y=raw_train$classe, list=FALSE, p=0.6)
train     <- raw_train[train_idx, ]
test      <- raw_train[-train_idx, ]

# We have 160 variables, including *classe*, so we can afford to be fairly brutal in our initial pruning.
# A model fit will be attempted on the items we have remaining, and if unsuccessful we will return to this step.

numerics <- which(lapply(train,class) %in% c('numeric'))
train_class <- train[ ,160]
test_class  <- test[ ,160]
validation_probid <- validation[ ,160]

train <- train[, numerics]
test  <- test[, numerics]
validation <- validation[, numerics]

# remove near zero variance indicators
nzv <- nearZeroVar(train)
train <- train[-nzv]
test  <- test[-nzv]
validation <- validation[-nzv]

imputer <- preProcess(train, method=c('knnImpute','pca'))

train <- predict(imputer, train)
test  <- predict(imputer, test)
validation <- predict(imputer, validation)

#model <- train(train, train_class, method='rf', ntree=50)
model <- randomForest(train, train_class, ntree=500)

train_pred <- predict(model, train)
confusionMatrix(train_pred, train_class)

test_pred <- predict(model, test)
confusionMatrix(test_pred, test_class)

answers <- predict(model, validation)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
