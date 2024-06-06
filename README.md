# Credit-Card-Fraud-Detection

# Summary:

# 1. Data Collection: 
The initial data is collected from the Kaggle. (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
The dataset is stored in a CSV file named "creditcard.csv" and it consists of the credit card transactions and information about the fraudulent transactions.

The dataset contains 31 numerical features (including V1 to V28), resulting from the PCA transformation due to privacy reasons. It also includes the features such as 'Time' and 'Amount'.

The features V1 to V28 in the credit card fraud dataset represent transformed or anonymized features derived from the original credit card transaction data.

The 'Time' feature represents the seconds elapsed between each transaction and the first transaction in the dataset.

The 'Amount' feature denotes the transaction amount.

The target variable 'Class' indicates whether a transaction is fraudulent (1) or not (0).

One of the key characteristics of this dataset is its severe class imbalance.

The majority of transactions are non-fraudulent, while fraudulent transactions are rare occurrences.

Class imbalance can significantly affect the performance of machine learning models, particularly in fraud detection tasks.

# 2. Data Visualization: 
I have performed data visualization to provide insights into the credit card transaction dataset.
Exploratory Data Analysis (EDA):

a. Histograms are used to visualize the distribution of the 'V1' feature.

b. Scatter plot is created to visualize the relationship between 'Time' and 'Amount'.

c. A correlation heatmap is generated to visualize the correlation between numerical features in the dataset.

# 3. Data Preprocessing/Preparation:
a. Handling Missing Values: The dataset identifies missing values using the isnull().sum() method, which counts the number of missing values for each feature.

The SimpleImputer class from scikit-learn is used to fill missing values with the mean strategy, ensuring that no missing values are present in the dataset before proceeding with model training.

b. Scaling Features: The 'Amount' feature in the dataset is scaled using the StandardScaler from scikit-learn. Scaling is important for ensuring that all features have the same scale and range, which helps improve the performance of certain machine learning algorithms, such as logistic regression and KMeans clustering.

c. Dealing with Imbalanced Data using SMOTE: Synthetic Minority Over-sampling Technique (SMOTE) is applied to handle the class imbalance in the dataset. SMOTE generates synthetic samples for the minority class (fraudulent transactions) to balance the class distribution. The SMOTE class from the imbalanced-learn library is used to perform SMOTE resampling on the training data.

d. Dimensionality Reduction with PCA: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset. PCA helps in reducing the number of features while preserving as much variance as possible in the data. The PCA class from scikit-learn is used to perform PCA transformation, reducing the number of features to 10 components.

In summary, data preparation involves handling missing values, scaling features, reducing dimensionality with PCA, and addressing class imbalance using SMOTE. These steps ensure that the dataset is well-prepared and suitable for training machine learning models, ultimately improving the models' performance and effectiveness in detecting fraudulent transactions.

# 4. Model Classification: 
I have used 2 classification techniques.
a. Logistic Regression Model: A logistic regression model is created using a pipeline that includes feature scaling (StandardScaler) and logistic regression classifier (LogisticRegression).

The model is trained on the resampled and PCA-transformed training data (X_train_pca and y_train_resampled).

Predictions are made on the PCA-transformed test set (X_test_pca) using the trained model. Performance metrics such as accuracy, confusion matrix, and classification report are calculated and printed to evaluate the model's performance.

Confusion matrix, ROC curve, and Precision-Recall curve are visualized to further assess the model's performance.

Feature importances of the logistic regression model are plotted to understand the importance of different features in the model's predictions.

b. Random Forest Classifier: A random forest classifier is instantiated with specified parameters such as the number of estimators (n_estimators=100), maximum depth of the trees (max_depth=10), and utilizing all available processors (n_jobs=-1).

The model is trained on the resampled and PCA-transformed training data (X_train_pca and y_train_resampled).

Predictions are made on the PCA-transformed test set (X_test_pca) using the trained model. Performance metrics such as accuracy, confusion matrix, and classification report are calculated and printed to evaluate the model's performance.

Confusion matrix, ROC curve, and Precision-Recall curve for the random forest classifier are visualized to further assess the model's performance.

Feature importances of the random forest classifier are plotted to understand the importance of different features in the model's predictions.

In summary, model classification involves training logistic regression and random forest classifier models, evaluating their performance using various metrics and visualizations, and understanding the importance of different features in predicting fraudulent transactions. These steps help in selecting the most suitable model for fraud detection based on its performance and interpretability.

# 5. Predictions:
Both model are highly accurate but the random forest had the slightest advantage over logistic regression. So, I have used the random forest model to perform the predictions.

The trained Random Forest classifier is used to predict whether a new credit card transaction is fraudulent or not.

Synthetic data representing a new credit card transaction is created and transformed using the same PCA transformation applied to the training data (new_entry_processed).

The predict() method is then applied to the transformed data to obtain the predicted label.

Based on the predicted label, a message is printed indicating whether the model predicts the new transaction to be fraudulent or non-fraudulent.

# 6. Conclusion:
In conclusion, my model showcases a comprehensive approach to credit card fraud detection using machine learning techniques. Through data exploration, preprocessing, model training, and evaluation, the code effectively addresses the challenges associated with class imbalance, missing data, and high dimensionality.

By selecting and fine-tuning the random forest classifier, the model demonstrates strong performance in accurately identifying fraudulent transactions.

The use cases for this solution extend beyond credit card fraud detection. Similar methodologies can be applied in various industries, including healthcare, insurance, and e-commerce, to detect anomalies, identify fraudulent activities, and enhance security measures.

Furthermore, the insights gained from the data visualization and feature importance analysis provide valuable information for decision-making processes, risk assessment, and fraud prevention strategies.

Overall, the implemented solution serves as a versatile tool for data-driven organizations seeking to mitigate risks and safeguard their operations against fraudulent activities.
