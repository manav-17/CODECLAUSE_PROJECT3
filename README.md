# CODECLAUSE_PROJECT3

Heart Disease Risk Assessment

Aim - Build a UI allowing users to input health metrics. Develop a machine learning model to predict the risk of heart disease.

Description - Create a user-friendly interface for inputting health data and implement a model (e.g., Random Forest) for risk assessment.

Technologies-

a.Pandas: A powerful library for data manipulation and analysis in Python. It is used to read the CSV file and handle the dataset throughout the code.

b.Streamlit: A framework for building interactive web applications in Python. It is used here to create a user interface for the heart disease risk assessment tool, allowing users to input their features and see predictions.

c.Scikit-learn (sklearn): A widely used machine learning library in Python. The following components from scikit-learn are utilized in your code:

1.RandomForestClassifier: A machine learning algorithm used for classification tasks. It is trained to predict the risk of heart disease based on user input.

2.train_test_split: A function to split the dataset into training and testing sets for model evaluation.

3.classification_report and accuracy_score: Functions to evaluate the performance of the model on the test dataset.

d.Machine Learning: The code demonstrates a machine learning workflow, including data loading, preprocessing, model training, and evaluation, specifically using a Random Forest classifier for predicting heart disease risk.

e.Data Preprocessing: The code prepares the input features for prediction and concatenates them with the original dataset for prediction purposes.

f.Model Evaluation: The code evaluates the trained model on the test data, providing metrics like accuracy and a classification report to assess performance.

g.User Input Handling: The function user_input_features() captures user input through various Streamlit widgets (like number inputs and select boxes), allowing for a user-friendly interface.

h.CSS Styling: The load_css function attempts to load a CSS file for custom styling of the Streamlit application, enhancing the user interface.

