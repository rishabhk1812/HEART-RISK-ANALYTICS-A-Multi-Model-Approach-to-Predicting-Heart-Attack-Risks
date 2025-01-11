# HEART-RISK-ANALYTICS-A-Multi-Model-Approach-to-Predicting-Heart-Attack-Risks
This project, Heart-Risk Analytics, leverages machine learning models—Decision Tree, Random Forest, Logistic Regression, and XGBoost—to predict heart attack risks. By preprocessing health data, training models, and evaluating performance, it identifies risk levels with precision, aiding proactive healthcare decisions.

**Key Objectives:**
The primary objective of the Heart-Risk Analytics project is to predict heart attack risks by utilizing various machine learning models. This project aims to provide accurate risk categorization (Low, Moderate, High) to support healthcare professionals in early intervention and prevention strategies.

**Data Overview:**
The dataset contains health-related features such as age, gender, cholesterol levels, blood pressure, diabetes, physical activity, and family history, among others. The target variable is the Heart Attack Risk, categorized into Low, Moderate, and High risk levels. Categorical variables were converted to factors, and the dataset was split into training (80%) and testing (20%) subsets for model development and evaluation.

**Approach:**

1. Preprocessing: The data was cleaned and transformed, ensuring categorical variables were encoded appropriately for model compatibility.
2. Models Used:
   i. Decision Tree: Constructed using the rpart library with a maximum depth of 30 and visualized using rpart.plot.
   ii. Random Forest: Built with 100 decision trees using the randomForest library for robust predictions.
   iii. Logistic Regression: Multinomial logistic regression was applied using the nnet package for multiclass classification.
   iv. XGBoost: The advanced boosting model was trained using the xgboost library with hyperparameters optimized for multi-class classification.
3. Evaluation: Each model was evaluated on the test dataset, and metrics like accuracy and confusion matrices were analyzed to compare performance.
4. Prediction for New Data: A hypothetical patient profile was used to demonstrate real-world applicability of the trained models.

**Expected Outcomes:**

The models aim to identify heart attack risk levels accurately and provide actionable insights into health risk categorization. Cross-validation techniques were applied to ensure robustness and to check for overfitting/underfitting issues in all models.

**Conclusion:**
The performance of each model is summarized as follows:

1. Logistic Regression:
The model achieved an accuracy of 50.05%, which is close to baseline and indicates no significant predictive power. Sensitivity was poor for the 'High' and 'Moderate' classes but perfect for 'Low'. However, the specificity was high for 'High' and 'Moderate' but zero for 'Low', reflecting an imbalance in predictions. The Kappa score of 0 confirms that the model's performance was no better than random guessing.

2. Decision Tree:
Similar to Logistic Regression, the Decision Tree achieved an accuracy of 50.05% and demonstrated the same sensitivity and specificity patterns. It also had a Kappa score of 0, indicating no added value beyond chance-level predictions.

3. Random Forest:
With an accuracy of 48.79%, the Random Forest underperformed compared to the Logistic Regression and Decision Tree. Sensitivity was extremely low for the 'High' (0.25%) and 'Moderate' (4.63%) classes, though relatively high for 'Low' (94.6%). Specificity was high for 'High' (99.72%) and 'Moderate' (94.79%), but very low for 'Low' (5.25%). The Kappa score of -0.0031 reveals performance worse than random guessing.

4. XGBoost:
XGBoost achieved an accuracy of 47.32%, which is lower than the baseline (50.05%). Sensitivity was poor for 'Low' (1.49%) and 'High' (8.05%), but decent for 'Moderate' (89.15%). Specificity was high for 'Low' (98.27%) and 'High' (91.04%) but very poor for 'Moderate' (9.89%). The Kappa score of -0.009 suggests that the model performed worse than random guessing.
