# titanic_decision_tree
kaggle_titanic
```
# 导入数据集
set.seed(123)
PATH <- './data/titanic_csv.csv'
titanic <- read.csv(PATH)
head(titanic)
tail(titanic)

shuffle.index <- sample(1:nrow(titanic))
head(shuffle.index)

titanic <- titanic[shuffle.index,]
head(titanic)
class(titanic)

# 清洗数据集
# 1 删除变量集
# 2 变量类型转换
# 3 删除NA
library(dplyr)
clean.titanic <- titanic %>% 
  dplyr::select(-c(home.dest, cabin, name, X, ticket)) %>% 
  dplyr::mutate(pclass = factor(pclass, levels = c(1, 2, 3), labels = c('Upper', 'Middle', 'Lower')),
         survived = factor(survived, levels = c(0, 1), labels = c('No', 'Yes'))) %>% 
  na.omit()
dplyr::glimpse(clean.titanic)

# 数据集划分
# 训练集和测试集
library(caret)
train.index <- createDataPartition(clean.titanic$survived, p = 0.8, list = FALSE)
train.data <- clean.titanic[train.index, ]
test.data <- clean.titanic[-train.index, ]

dim(train.data)
dim(test.data)

# 目标变量的分布
prop.table(table(train.data$survived))

# 生成决策树
library(rpart)
library(rpart.plot)
tree.fit <- rpart(
  survived ~ ., 
  data = train.data,
  method = 'class'
)
par(mfrow=c(1,1))
rpart.plot(tree.fit, extra = 100)


# 模型的预测
predict.unseen <- predict(tree.fit, test.data, type = 'class')

# 模型评价-混淆矩阵
table.mat <- table(test.data$survived, predict.unseen)
table.mat

# 模型新能评价
# 模型准确度
accuracy.test <- sum(diag(table.mat)) / sum(table.mat)
print(paste('Accuracy for test', accuracy.test))

# 模型调参优化
AccuracyTune <- function(fit){
  predict.unseen <- predict(fit, test.data, type='class')
  table.mat <- table(test.data$survived, predict.unseen)
  accuracy.test <- sum(diag(table.mat)) / sum(table.mat)
  accuracy.test
}
control <- rpart.control(
  minsplit = 4,
  minbucket = round(5/3),
  maxdepth = 3,
  cp = 0
)

tune.fit <- rpart(survived ~ ., data = train.data, method = 'class', control = control)
AccuracyTune(tune.fit)
```
