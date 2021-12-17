
# Credit scoring algorithms


# This is the report of Brice Akouvi and Antoine Chatelain made after the ML Hackaton.
# The goal of this competition was to build a model that borrowers can use to 
# help make the best financial decisions to know if somebody will experience 
# financial distress in the next two years.

# In these datasets, it is represented by the **SeriousDlqin2yrs** variable which 
# is a binary variable y in {0,1}. So if y = 1, the person will be more likely to 
# have a financial distress. We are on a classification problem which requires 
# supervised learning.

# In order to get the best prediction, we've started to clean, 
# explore and analyse the train dataset and then to try to compared multiple ML models.

rm(list=ls())


### Libraries required 

library(rpart)
library(caTools)
library(corrplot)
library(MASS)
library(factoextra)
library(FactoMineR)
library(e1071)
require(xgboost)
library(randomForest)
library(gbm)
library(MLmetrics)




## Data Gathering

dirpath <- "./data/"
train <- read.csv(paste(dirpath, "data.csv", sep= ""))
test <- read.csv(paste(dirpath, "data.csv", sep= ""))

dim(train)
dim(test)

summary(train)


# Data cleaning


# Check omitted values

empty.col <- colSums(is.na(train))
empty.row <- rowSums(is.na(train))
empty.col[empty.col > 0]
which(empty.row > 0)

# We notice at first that there is multiple Na values in all columns 
# which means that there was probably multiple individuals to omit  

train <- na.omit(train)
dim(train)

# After removing all rows that contains Na value, 
# we have our training dataset of 53000 x 11 dimension


#### PCA section : Removing the inneficient variable 

#train[, 1] <- as.factor(train[, 1])
pca <- PCA(train, scale.unit = T, quanti.sup = 1, ncp = 2)
summary(pca)
fviz_eig(pca, addlabels=T)

# We quikly find out that it would be useless to try to run our models with reducted data

fviz_pca_var(pca, col.var = "cos2")

# With this plot, we can't pretend anything about variable correlation 
# to the predicator beacause most of them aren't well represented 
# (espacially the predicator **SeriousDlqin2yrs**)

fviz_contrib(pca, choice = "var", axes = 1, bottom = 5)


## Data Visualisation

y <- as.factor(train$SeriousDlqin2yrs)
tab <- table(y)
names(tab) <- c('No', 'Yes')
barplot(tab, 
        main="Number of person who will probably experience a financial distress",
        col=c('darkblue', 'darkred'),
        ylab='Number of person', xlab='Will they experience it ?')

# We noticed that we have an unbalanced dataset, with too much people in the first class 


# Analysis of train dataset 

# By analysing the correlation of variables, we can have a preview of useful 
# and useless  variable. Perhaps by eliminating those variables, 
# we could get a better reponse emitted by the model we will test  

corrplot(cor(train), method="square",tl.srt = 45, tl.col = 'black', type = 'upper')

# We can immediatly see there is a huge correlation between **NumberOfTime60_89DaysPastDueNotWorse**, **NumberOfTime30_59DaysPastDueNotWorse** and **NumberOfTimes90DaysLate** variables

pairs(train, lower.panel=NULL, pch=16, col=c('darkblue', 'darkred')[y])



# Model selection


# Validation data creation

# In order to **compare the different models** we will lately test, 
# the split of train data is required 


set.seed(128)
s <- sample.split(train$SeriousDlqin2yrs, SplitRatio = 0.8)
train_data <- subset(train, s == T)
val_data <- subset(train, s == F)
dim(train_data)
dim(val_data)


# Discriminant Analysis : LDA and QDA 

# we choose to start by doing LDA & QDA because they imply gaussian density functions for both classes, those two models were pretty good but LDA was better and it's over all the model we tried the best.

cl.lda <- lda(SeriousDlqin2yrs ~ ., data=train_data)
cl.lda.test <- lda(SeriousDlqin2yrs ~ ., data=train) # Final model
cl.lda


# we make cl.lda.test to fit the model for the test set given 
# for the hackaton so by training our model with more value we should make it
# more accurate even if there is high risks of overfitting 
# (submitted twice to test which is the best)


cl.qda <- qda(SeriousDlqin2yrs ~ ., data=train_data)
cl.qda

# Logistic Regression 

# Then, a logistic regression, logic and unavoidable

log.reg <- glm(SeriousDlqin2yrs ~ ., data=train_data, family = binomial)
log.reg.test <- glm(SeriousDlqin2yrs ~ ., data=train, family = binomial)
log.reg


# Random Forest & Boosting and Extreme Gradient Boosting

# Random forest

rd.forest <- randomForest(SeriousDlqin2yrs ~ ., data=train_data, 
                          mtry=3, ntrees = 100, sampsize = 700)
rd.forest

# eXtreme Gradient Boosting

cl.xgb <- xgboost(data = as.matrix(train_data[, -1]), label = train_data[, 1], 
                  max.depth = 3, eta = 1, nthread = 2, 
                  nrounds = 2, objective = "binary:logistic")


# SVM

# At the end, we saw the possibility of using a SVM Classifier 
# with a linear kernel (because LDA with linear boudaries decisions
# was more efficient than QDA for instance) but there was little time 
# remaining so we couldn't test it 

# To train this model, be patient (~10 min)

cl.svm <- svm(SeriousDlqin2yrs ~ ., type="C-classification",
              data = train, kernel = "linear")
cl.svm


# Metrics : accuracy

# The metric required to maximaze was the F1 score 
# So let's use it to compare our different model on validation dataset


### Testing 

y.pred.lda <- predict(cl.lda, val_data)
y.pred.qda <- predict(cl.qda, val_data)
y.pred.svm <- predict(cl.svm, val_data)
y.pred.xgb <- predict(cl.xgb, as.matrix(val_data[, -1]))
y.pred.xgb <- ifelse(y.pred.xgb > 0.5, 1, 0)
y.pred.log <- predict(log.reg, val_data, type="response")
y.pred.log <- ifelse(y.pred.log > 0.5, 1, 0)
y.pred.forest <- predict(rd.forest,val_data)
y.pred.forest <- ifelse(y.pred.log > 0.5, 1, 0)

cat("\n\nLDA result")
table(y.pred.lda$class, val_data$SeriousDlqin2yrs)
F1_Score(y.pred.lda$class, val_data$SeriousDlqin2yrs)
Accuracy(y.pred.lda$class, val_data$SeriousDlqin2yrs)

cat("\n\nQDA result")
table(y.pred.qda$class, val_data$SeriousDlqin2yrs)
F1_Score(y.pred.qda$class, val_data$SeriousDlqin2yrs)
Accuracy(y.pred.qda$class, val_data$SeriousDlqin2yrs)

cat("\n\nLogistic Regression result")
table(y.pred.log, val_data$SeriousDlqin2yrs)
F1_Score(y.pred.log, val_data$SeriousDlqin2yrs)
Accuracy(y.pred.log, val_data$SeriousDlqin2yrs)

cat("\n\nRandom Forest result")
table(y.pred.forest, val_data$SeriousDlqin2yrs)
F1_Score(y.pred.forest, val_data$SeriousDlqin2yrs)
Accuracy(y.pred.forest, val_data$SeriousDlqin2yrs)

cat("\n\nXGB result")
table(y.pred.xgb, val_data$SeriousDlqin2yrs)
F1_Score(y.pred.xgb, val_data$SeriousDlqin2yrs)
Accuracy(y.pred.xgb, val_data$SeriousDlqin2yrs)


# as seen below lda is better than qda for this dataset.


# LDA With less features

# We will use the results obtained at PCA section to improve our model

useless_col <- names(train) %in% c('age','RevolvingUtilizationOfUnsecuredLines')

train_ <- train[,which(!useless_col)]
test <- test[,which(!useless_col)[-1]-1]

s <- sample.split(train_$SeriousDlqin2yrs, SplitRatio = 0.8)
train_data <- subset(train_, s == T)
val_data <- subset(train_, s == F)
cl.lda.less <- lda(SeriousDlqin2yrs ~ ., data=train_data)
cl.lda.less

y.pred.lda.less <- predict(cl.lda.less, val_data)
table(y.pred.lda.less$class, val_data$SeriousDlqin2yrs)
F1_Score(y.pred.lda.less$class, val_data$SeriousDlqin2yrs)
Accuracy(y.pred.lda.less$class, val_data$SeriousDlqin2yrs)

# Finally, we have our best model

# Publish

y.test <- predict(cl.lda.less, test)

to_be_submitted = data.frame(id=rownames(test), SeriousDlqin2yrs=y.test$class)
write.csv(to_be_submitted , file = "to_be_submitted.csv", row.names = F)


### Conclusion

# As you can see, we explored multiple machine learning model for this clasification 
# problem but the fact is that it was a competition and unfortunately, we only had 
# a few hours to do this notebook from the beginning till the end. We would have processed 
# differently to compare and get the best model. 

# For instance, we'd have trained those models on different way, digging each ML 
# algorithm in order to find and select their best hyperparameters and variables, 
# espacially with a cross-validation.
 
# Even, before that, we'd have explored our train dataset and cleaned the outliers 
# that could come disturb the models' accuracies and analyse deeper the dependancy 
# between the variables.

# And finally, in search of best model, use deep learning methods... who knows ? 


