data <- read.csv("Heart Attack Risk.csv")

# Set seed for reproducibility
set.seed(123)

# Convert categorical variables to factors
data <- data %>% mutate(across(where(is.character), as.factor))

str(data)

# Split the dataset into training and testing sets
train_index <- createDataPartition(data$Heart_Attack_Risk, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


########################### Train a Decision Tree model ################################

decision_tree_model <- rpart(Heart_Attack_Risk ~ ., data = train_data, method = "class", control = rpart.control(maxdepth = 30))

# Visualize the Decision Tree
rpart.plot(decision_tree_model, box.palette = "Blues", shadow.col = "gray", nn = TRUE)

# Evaluate models on the test set
# Decision Tree predictions
dt_predictions <- predict(decision_tree_model, test_data, type = "class")
dt_confusion_matrix <- confusionMatrix(dt_predictions, test_data$Heart_Attack_Risk)
print("Decision Tree Performance:")
print(dt_confusion_matrix)


# For Decision Tree Checking the Test VS Train Data
dt_predictions_train <- predict(decision_tree_model, train_data, type = "class")
dt_predictions_test <- predict(decision_tree_model, test_data, type = "class")

# Calculate accuracy for both training and test sets for Decision Tree
dt_accuracy_train <- sum(dt_predictions_train == train_data$Heart_Attack_Risk) / nrow(train_data)
dt_accuracy_test <- sum(dt_predictions_test == test_data$Heart_Attack_Risk) / nrow(test_data)

# Print the accuracies Decision Tree
cat("Decision Tree Accuracy (Training): ", dt_accuracy_train * 100, "%\n")
cat("Decision Tree Accuracy (Test): ", dt_accuracy_test * 100, "%\n")


# Checking if the model is Overfitting / Underfitting using "cross-validation" Method

dt_train_control <- trainControl(method = "cv", number = 10)
dt_model_cv <- train(Heart_Attack_Risk ~ ., data = train_data, method = "rpart", trControl = dt_train_control)
print(dt_model_cv)


######################### Train a Random Forest model ####################################

random_forest_model <- randomForest(Heart_Attack_Risk ~ ., data = train_data, ntree = 100, mtry = 5)

print(random_forest_model)

# Evaluate models on the test set
# Random Forest predictions
rf_predictions <- predict(random_forest_model, test_data)
rf_confusion_matrix <- confusionMatrix(rf_predictions, test_data$Heart_Attack_Risk)
print("Random Forest Performance:")
print(rf_confusion_matrix)

# For Random Forest Checking the Test VS Train Data
rf_predictions_train <- predict(random_forest_model, train_data)
rf_predictions_test <- predict(random_forest_model, test_data)

# Calculate accuracy for both training and test sets for Random Forest
rf_accuracy_train <- sum(rf_predictions_train == train_data$Heart_Attack_Risk) / nrow(train_data)
rf_accuracy_test <- sum(rf_predictions_test == test_data$Heart_Attack_Risk) / nrow(test_data)

# Print the accuracies Random Forest
cat("Random Forest Accuracy (Training): ", rf_accuracy_train * 100, "%\n")
cat("Random Forest Accuracy (Test): ", rf_accuracy_test * 100, "%\n")

# Checking if the model is Overfitting / Underfitting using "cross-validation" Method

rf_train_control <- trainControl(method = "cv", number = 10)
rf_model_cv <- train(Heart_Attack_Risk ~ ., data = train_data, method = "rf", trControl = rf_train_control)
print(rf_model_cv)


############### Training Model on Mutinominal Logistic Regression ############################

# Train the multinomial logistic regression model
lrmodel <- multinom(Heart_Attack_Risk ~ ., data = train_data)

# Print model summary
summary(lrmodel)

# Predict the Heart Risk level for the test data
lrpredict <- predict(lrmodel, newdata = test_data)

# Add predicted and actual values to the test data
test_data$Predicted_Heart_Attack_Risk <- lrpredict

# Print the complete test data with predictions
print(test_data)

# Calculate model accuracy
confMatrix <- confusionMatrix(lrpredict, test_data$Heart_Attack_Risk)
print(confMatrix)

# To check the "accuracy" of Model 
accuracy <- confMatrix$overall["Accuracy"]
print(paste("Model Accuracy: ", accuracy))



############################## XGBoost Moodel for Prediction ############################

data_xgb <- data %>% mutate(across(where(is.character), ~ as.integer(as.factor(.)) - 1))
data_xgb <- data_xgb %>% mutate(across(where(is.factor), as.numeric))

str(data_xgb)

train_index_xgb <- createDataPartition(data_xgb$Heart_Attack_Risk, p = 0.8, list = FALSE)
train_data_xgb <- data_xgb[train_index, ]
test_data_xgb <- data_xgb[-train_index, ]

# Manually adjust the labels to start from 0
train_data_xgb$Heart_Attack_Risk <- train_data_xgb$Heart_Attack_Risk - 1
test_data_xgb$Heart_Attack_Risk <- test_data_xgb$Heart_Attack_Risk - 1

# Ensure that the target variable labels are in the range [0, num_class-1]
# Check the unique values of the target variable
train_labels <- unique(train_data_xgb$Heart_Attack_Risk)
test_labels <- unique(test_data_xgb$Heart_Attack_Risk)

cat("Unique labels in training set: ", train_labels, "\n")
cat("Unique labels in testing set: ", test_labels, "\n")


# Prepare the training and testing data for XGBoost
# Exclude the target variable (Heart_Attack_Risk) from the features

train_matrix <- xgb.DMatrix(data = as.matrix(train_data_xgb[, -which(names(train_data_xgb) == "Heart_Attack_Risk")]), 
                            label = train_data_xgb$Heart_Attack_Risk)

test_matrix <- xgb.DMatrix(data = as.matrix(test_data_xgb[, -which(names(test_data_xgb) == "Heart_Attack_Risk")]), 
                           label = test_data_xgb$Heart_Attack_Risk)

# XGBoost parameters for multi-class classification

params <- list(
  objective = "multi:softmax", 
  num_class = 3,         # Number of classes in the target variable (Heart_Attack_Risk)
  eval_metric = "merror",
  max_depth = 6,         # Maximum depth of trees
  eta = 0.3,             # Learning rate
  subsample = 0.8,       # Fraction of data used for each tree
  colsample_bytree = 0.8 # Fraction of features used for each tree
)

# Train the XGBoost model
xgboost_model <- xgb.train(
  params = params, 
  data = train_matrix, 
  nrounds = 100
)

# Make predictions with XGBoost
xgboost_predictions <- predict(xgboost_model, test_matrix)

# Convert predictions to factor with proper labels (Low, Moderate, High)
xgboost_predictions <- factor(xgboost_predictions, levels = c(0, 1, 2), labels = c("Low", "Moderate", "High"))

# Ensure the actual labels in the test set are factors with the same levels
test_data_xgb$Heart_Attack_Risk <- factor(test_data_xgb$Heart_Attack_Risk, levels = c(0, 1, 2), labels = c("Low", "Moderate", "High"))

# Evaluate XGBoost model performance
xgboost_conf_matrix <- confusionMatrix(xgboost_predictions, test_data_xgb$Heart_Attack_Risk)
print("XGBoost Performance:")
print(xgboost_conf_matrix)


############################## Predict risk for a new patient ############################

new_patient <- data.frame(
  Age = 55,
  Gender = factor("Male", levels = levels(train_data$Gender)),
  Smoking = 1,
  Alcohol_Consumption = 1,
  Physical_Activity_Level = factor("Low", levels = levels(train_data$Physical_Activity_Level)),
  BMI = 32,
  Diabetes = 1,
  Hypertension = 1,
  Cholesterol_Level = 250,
  Resting_BP = 150,
  Heart_Rate = 95,
  Family_History = 1,
  Stress_Level = factor("High", levels = levels(train_data$Stress_Level)),
  Chest_Pain_Type = factor("Typical", levels = levels(train_data$Chest_Pain_Type)),
  Thalassemia = factor("Fixed defect", levels = levels(train_data$Thalassemia)),
  Fasting_Blood_Sugar = 1,
  ECG_Results = factor("ST-T abnormality", levels = levels(train_data$ECG_Results)),
  Exercise_Induced_Angina = 1,
  Max_Heart_Rate_Achieved = 140
)

str(new_patient)
str(train_data)


# Predict using Decision Tree and Random Forest
new_patient_dt_prediction <- predict(decision_tree_model, new_patient, type = "class")
new_patient_rf_prediction <- predict(random_forest_model, new_patient)

cat("Predicted Heart Attack Risk using Decision Tree:", new_patient_dt_prediction, "\n")
cat("Predicted Heart Attack Risk using Random Forest:", new_patient_rf_prediction, "\n")

#################################### End #####################################################



############################### Comparing all the models #####################################

# Get predictions from both models for the test set
dt_predictions_test <- predict(decision_tree_model, test_data, type = "class")
rf_predictions_test <- predict(random_forest_model, test_data)
lr_predictions_test <- predict(lrmodel, newdata = test_data)
xgboost_predictions <- predict(xgboost_model, test_matrix)

# Convert XGBoost predictions to factors with correct labels
xgboost_predictions <- factor(xgboost_predictions, levels = c(0, 1, 2), labels = c("Low", "Moderate", "High"))


# Combine all predictions into a data frame
comparison_df <- data.frame(
  Actual = test_data$Heart_Attack_Risk,
  Decision_Tree = dt_predictions_test,
  Random_Forest = rf_predictions_test,
  Logistic_Regression = lr_predictions_test,
  XGBoost = xgboost_predictions
)

print(head(comparison_df))

write_xlsx(comparison_df,"Actual VS predicted.xlsx")