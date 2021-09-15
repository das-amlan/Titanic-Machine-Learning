library(psych)
library(ggplot2)
library(dplyr)
library(Amelia)
library(caTools)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)

train <- read.csv("train.csv", header = TRUE, sep = ",")
test <- read.csv("test.csv",header = TRUE, sep = ",")

## Exploring the Data

describe(train)
describe(test)

colSums(is.na(train))
# PassengerId    Survived      Pclass        Name         Sex         Age       SibSp       Parch 
# 0           0           0           0           0         177           0           0 
# Ticket        Fare       Cabin    Embarked 
# 0           0           0           0 
colSums(is.na(test))
# PassengerId      Pclass        Name         Sex         Age       SibSp       Parch      Ticket 
# 0           0           0           0          86           0           0           0 
# Fare       Cabin    Embarked 
# 1           0           0


missmap(train, col = c('yellow', 'black'), main = "Missing Value in training data", legend = FALSE)
missmap(test, col = c('yellow', 'black'), main = "Missing Value in test data", legend = FALSE)

# merging train and test data set

train$IsTrain <- TRUE
test$IsTrain <- FALSE

test$Survived <- NA

mergedt <- rbind(train, test) 
head(mergedt)
colSums(is.na(mergedt))

# Visualizing Data

ggplot(mergedt, aes(Survived)) + geom_bar()
ggplot(mergedt,aes(Age)) + geom_histogram(bins = 20, alpha = 0.5 , fill = "blue")
ggplot(mergedt,aes(Sex)) + geom_bar(aes(fill = factor(Sex))) + theme_bw()

ggplot(mergedt, aes(x = Survived, fill=Sex)) + geom_bar(position = position_dodge())+
  geom_text(stat= "count", aes(label=stat(count)),position = position_dodge(width=1), vjust=-0.5)+
  theme_bw()
ggplot(mergedt,aes(Pclass,Age)) + 
  geom_boxplot(aes(group = Pclass,fill = factor(Pclass)),alpha= 0.4) +
  scale_y_continuous(breaks = seq(min(0),max(80),by = 2))+
  theme_bw()

# Imputing the missing values

age_imputation <- function(age,class){
  out <- age
  for (i in 1:length(age)) {
    if(is.na(age[i])){
      if(class[i] == 1){
        out[i] <- 37
      }else if(class[i] == 2){
        out[i] <- 29
      } else{
        out[i] <- 24
      }
    } else{
      out[i] <- age[i]
    }
  }
  return(out)
}

imputedAge <- age_imputation(mergedt$Age, mergedt$Pclass)
mergedt$Age <- imputedAge

mergedt$Fare <- as.numeric(as.character(mergedt$Fare))
mergedt[is.na(mergedt$Fare), "Fare"] <- median(mergedt$Fare, na.rm = TRUE)

colSums(is.na(mergedt))
# PassengerId    Survived      Pclass        Name         Sex         Age       SibSp       Parch 
# 0         418           0           0           0           0           0           0 
# Ticket        Fare       Cabin    Embarked     IsTrain 
# 0           0           0           0           0 

mergedt$Pclass <- as.factor(mergedt$Pclass)
mergedt$Sex <- as.factor(mergedt$Sex)
mergedt$Embarked <- as.factor(mergedt$Embarked)
mergedt$Survived <- as.factor(mergedt$Survived)

# Splitting back to Test, Training 

nwtrain <- mergedt[mergedt$IsTrain==TRUE,]
nwtest <- mergedt[mergedt$IsTrain==FALSE,]


dfTrain <- select(nwtrain, -PassengerId, -Name, -Ticket, -Cabin, -IsTrain)
dfTest <- select(nwtest, -PassengerId, -Name, -Ticket, -Cabin, -IsTrain)
###########################
# Creating traing and validation set from the training data

set.seed(101)
split <- sample.split(dfTrain$Survived,SplitRatio = 0.7)
finalTrain <- subset(dfTrain, split == TRUE)
valTest <- subset(dfTrain, split == FALSE)

#####################
## Building a model ##
# Decision Tree 

dtModel <- rpart(Survived~., data = finalTrain, method = "class")
rpart.plot(dtModel, extra = "auto" , main = "Decision Tree")

dtPredicted = predict(dtModel, newdata = valTest , type = "class")

confusionMatrix(valTest$Survived, dtPredicted)
# Accuracy : 0.8134

## Logit

glmModel <- glm(formula = Survived~. , family = binomial(link = "logit"),data = finalTrain)
print(summary(glmModel))

prediction <- predict(glmModel,newdata = valTest,type = "response")
fittedResult <- factor(ifelse(prediction > 0.5,1,0)) 

confusionMatrix(valTest$Survived, fittedResult)
# Accuracy : 0.7836 


## Naïve Bayes
nbModel = naiveBayes(Survived ~., data=finalTrain)
nbpredict = predict(nbModel, valTest)

confusionMatrix(valTest$Survived, nbpredict)
# Accuracy : 0.7687 


## Random Forest
rfModel = randomForest(formula = Survived~. , data = finalTrain, ntree = 500, mtry = 3, nodesize = 0.01 * nrow(finalTrain))

rfPred = predict(rfModel, valTest)
confusionMatrix(valTest$Survived, rfPred)
# Accuracy : 0.8396 

# random forest model has the highest accuracy
# making the final prediction

Survived <- predict(rfModel, newdata = dfTest)

# Creating Output file

PassengerId <- test$PassengerId
df_new <- as.data.frame(PassengerId)
df_new$Survived <- Survived

head(df_new)

write.csv(df_new, file = "titanic_soln.csv", row.names = FALSE)

# Score of the model 0.77751 