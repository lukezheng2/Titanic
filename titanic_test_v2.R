library(tidyverse)
library(magrittr)
library(data.table)
library(MLmetrics)
library(stringr)
library(glmnet)

# tratando a base -------------------------------------------------------------------

train <- fread("../train.csv") 
test <- fread("../test.csv")
test <- test %>% mutate(Survived = 0, Embarked = factor(Embarked, levels = c("", "C", "Q", "S")))

#Verificando quantidades de Na em cada variável
#train %>% sapply(function(x) sum(is.na(x)))

new_base <- function(base){
  base %>% 
    mutate(#removendo os Na's
      FLAG_Age = ifelse(is.na(Age),"0","1"),
      Age = ifelse(is.na(Age),0,Age),
      #tratando os nomes
      Name_level = ifelse(str_detect(Name, "Mr."), "Mr", "Others"),
      Name_level = ifelse(str_detect(Name, "Mrs."), "Mrs", Name_level),
      Name_level = ifelse(str_detect(Name, "Miss."), "Miss", Name_level),
      #FACTOR PCLASS
      Pclass = as.factor(Pclass))
}

new_train <- new_base(train)

# Modelo de regressão logística -------------------------------------------------------------------

glm <- glm(Survived ~ . , family = binomial, data = new_train %>% select(-PassengerId, -Name,- Ticket, -Cabin, -Fare))
summary(glm)
step <- step(glm, direction = "backward", trace = F)
summary(step)

pre <- predict(glm, newdata = new_train, type = "response") %>% as.data.frame %>% 
  mutate(Survived = ifelse(.>.4,"S","N"))

desempenho <- function(pre, train=fread("../train.csv")){
  KS = KS_Stat(pre$Survived, train$Survived)
  A = table(pre$Survived, train$Survived)
  ACC = table(pre$Survived, train$Survived) %>% as.data.frame() %>% {.[c(1,4),"Freq"] %>% sum}/nrow(train)
  return(list(KS = KS, ACC = ACC, TABLE = A))
}

#desempenho(pre)

resp <- new_base(test) %>% 
  bind_cols(Prob = predict(glm, newdata = new_base(test), type = "response")) %>% 
  mutate(Survived = ifelse(Prob>.4,1,0)) %>% 
  select(PassengerId, Survived)

resp %>% write.csv(file = "../resp_logistic_regression.csv", row.names = F)
#ACC = 0.76076


# Lasso & Ridge-------------------------------------------------------------------

elasticnet_model <- function(base, alpha = 1){ #Falta arrumar
  base <- base %>% select(-PassengerId, -Name,- Ticket, -Cabin, -Fare)
  #treino e teste
  set.seed(123456)
  n <- sample(1:nrow(base), size = trunc(0.9*nrow(base)), replace = F)
  matriz = model.matrix(Survived ~. , base)[,-1] %>% apply(2, scale)
  matriz_treino = matriz[n,]
  matriz_teste = matriz[-n,]
  
  fit = cv.glmnet(x = matriz_treino, y = base[n,]$Survived, alpha = alpha, family = "binomial")
  
  acc=array()
  for(i in 1:length(fit$lambda)){
    #cat(i, "\n")
    acc[[i]] = base[-n,] %>% 
      mutate(Prob = fit %>% predict(s=fit$lambda[i], newx=matriz_teste, type= "response") %>% c()) %>% 
      mutate(Survived = ifelse(Prob>.59,1,0)) %>% 
      select(Survived) %>% 
      desempenho(train = base[-n,]) %>% {.$KS}
  }
  lambda_acc = fit$lambda[which.max(acc)]
  
  warning("Alpha = ",alpha, "\n Alpha = 1 -> Lasso \n Alpha = 0 -> Ridge")
  return(list(fit = fit, lambda_acc = lambda_acc, acc=acc))
  
}

lasso_model = elasticnet_model(new_train)
ridge_model = elasticnet_model(new_train, alpha = 0)
coef(lasso_model$fit, s=lasso_model$lambda_acc)
coef(ridge_model$fit, s=ridge_model$lambda_acc)

predict_elasticnet <- function(base,fit){
  base2 <- base %>% select(-PassengerId, -Name,- Ticket, -Cabin, -Fare)
  #base %>% sapply(function(x) unique(x)) %>% View
  matriz = model.matrix(Survived ~. , base2)[,-1] %>% apply(2, scale) 
  P = base %>% 
    bind_cols(Prob = fit$fit %>% predict(newx = matriz, s=fit$lambda_acc, type = "response") %>% c()) %>% 
    mutate(Survived = ifelse(Prob>.59,1,0)) %>% 
    select(PassengerId, Survived)
  return(P)
}

predict_elasticnet(new_train, lasso_model) %>% desempenho()
predict_elasticnet(new_train, ridge_model) %>% desempenho()

predict_elasticnet(new_base(test), lasso_model) %>% 
  write.csv(file = "../resp_logistic_regression_lasso3.csv", row.names = F)
#ACC -.5 = 0.78468
#ACC -.59 = 0.79425

predict_elasticnet(new_base(test), ridge_model) %>% 
  write.csv(file = "../resp_logistic_regression_ridge.csv", row.names = F)
#ACC -.59 = 0.79425



# Decison Tree and Random Forest ------------------------------------------

make_tree <- function(base){
  base <- base %>% 
    dplyr::select(-PassengerId, -Name, -Ticket, -Cabin, -Fare) %>% 
    mutate_if(is.character, funs(as.factor)) %>% 
    mutate(Survived = factor(Survived))
  
  #set.seed(1234)
  #n <- sample(1:nrow(base), size = trunc(0.9*nrow(base)), replace = F)
  tree <- party::ctree(Survived ~. , data = base)
  plot(tree)
  return(tree)
}

tree <- make_tree(new_train)

predict_tree <- function(base, model){
  base2 <- base %>% 
    dplyr::select(-PassengerId, -Name, -Ticket, -Cabin, -Fare) %>% 
    mutate_if(is.character, funs(as.factor)) %>% 
    mutate(Survived = factor(Survived))
  #predict(model, base2) %>% c()
  
  P = base %>% 
    dplyr::select(PassengerId) %>% 
    bind_cols(Survived = predict(model, base2) %>% c()) %>% 
    mutate(Survived = Survived-1)
    
  return(P)
}

predict_tree(new_train, tree) %>% desempenho()

predict_tree(new_base(test), tree) %>% 
  write.csv(file = "../resp_tree", row.names = F)

make_forest <- function(base){
  base <- base %>% 
    dplyr::select(-PassengerId, -Name, -Ticket, -Cabin, -Fare) %>% 
    mutate_if(is.character, funs(as.factor)) %>% 
    mutate(Survived = factor(Survived))
  
  titanic.rf=randomForest::randomForest(Survived ~ . , data = base , subset = c(1:nrow(base)))
  plot(titanic.rf)
  return(titanic.rf)
}

random_forest <- make_forest(new_train)
predict_tree(new_train, random_forest) %>% desempenho()
predict_tree(new_base(test), random_forest) %>% 
  write.csv(file = "../resp_rf", row.names = F)
