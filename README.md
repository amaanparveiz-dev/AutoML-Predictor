# AutoML Predictor

A desktop-based **AutoML Predictor and Machine Learning Studio** built with Python, CustomTkinter, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

The application allows users to load a CSV dataset, select whether the problem is a classification or regression task, choose input and output columns, automatically clean and encode the dataset, evaluate multiple machine learning algorithms with different train/test splits and parameters, select the best-performing model, and make predictions through a graphical user interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Supported Machine Learning Models](#supported-machine-learning-models)
- [Data Preprocessing](#data-preprocessing)
- [Prediction System](#prediction-system)
- [Technology Stack](#technology-stack)
- [Application Interface](#application-interface)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [How to Use](#how-to-use)
- [Classification](#classification)
- [Regression](#regression)
- [Model Selection](#model-selection)
- [Dataset Requirements](#dataset-requirements)
- [Example Workflow](#example-workflow)
- [Important Implementation Details](#important-implementation-details)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Learning Objectives](#learning-objectives)
- [Dependencies](#dependencies)
- [License](#license)
- [Author](#author)

---

# Overview

**AutoML Predictor** is a GUI-based machine learning application designed to simplify the process of preparing datasets, training machine learning models, comparing their performance, and generating predictions.

Instead of manually writing separate training code for every dataset, the application provides a step-by-step interface where the user can:

1. Load a CSV dataset.
2. Select Classification or Regression.
3. Select the target/output column.
4. Select input/features columns.
5. Automatically clean and preprocess the data.
6. Train multiple machine learning models.
7. Test different train/test split ratios.
8. Compare model scores.
9. Automatically select the best-performing configuration.
10. Enter new feature values.
11. Generate a prediction.

The application is designed as an educational AutoML-style tool that demonstrates how different stages of a machine learning workflow can be combined into a single desktop application.

---

# Key Features

## Dataset Loading

Users can load CSV datasets directly through a graphical file picker.

The application reads the selected dataset using Pandas.

```python
df = pd.read_csv(path)
```

## Classification and Regression Modes

The application supports two major machine learning tasks:

### Classification

Used when the target variable represents categories or classes.

Examples:

- Pass / Fail
- Yes / No
- Disease / No Disease
- Low / Medium / High

### Regression

Used when the target variable represents a continuous numerical value.

Examples:

- House Price
- Salary
- Temperature
- Sales
- Student Marks

The application provides a GUI toggle for selecting the desired mode.

## Machine Learning Workflow

The application follows a structured workflow:

```
┌─────────────────────┐
│     Load CSV        │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Select Model Type   │
│ Classification /    │
│ Regression          │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Select Output       │
│ Target Column       │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Select Input        │
│ Feature Columns     │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Data Cleaning       │
│ Missing Values      │
│ Encoding            │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Train Multiple      │
│ ML Models           │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Compare Scores      │
│ & Parameters        │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Select Best Model   │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Enter New Data      │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Generate Prediction │
└─────────────────────┘
```

## Supported Machine Learning Models

The application supports multiple algorithms for both classification and regression.

### Classification Models

**1. Logistic Regression**

```python
LogisticRegression(max_iter=1000)
```

Used for classification problems where the target consists of discrete classes.

**2. Decision Tree Classifier**

```python
DecisionTreeClassifier()
```

A tree-based classification algorithm that creates decision rules from the dataset.

**3. Random Forest Classifier**

```python
RandomForestClassifier(
    n_estimators=i*10,
    random_state=42
)
```

The application tests different numbers of trees to find a strong configuration.

**4. Ridge Classifier**

```python
RidgeClassifier(alpha=i*0.2)
```

A linear classification model that applies L2 regularization.

**5. K-Nearest Neighbors**

```python
KNeighborsClassifier(n_neighbors=i)
```

The application tests different values of n_neighbors.

**6. Gradient Boosting Classifier**

```python
GradientBoostingClassifier(
    n_estimators=i*10,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

A boosting-based classification algorithm that builds an ensemble of weak learners.

### Regression Models

**1. Linear Regression**

```python
LinearRegression()
```

A basic linear regression algorithm used to model relationships between input features and a continuous target.

**2. Decision Tree Regressor**

```python
DecisionTreeRegressor()
```

A tree-based algorithm for predicting continuous numerical values.

**3. Random Forest Regressor**

```python
RandomForestRegressor(
    n_estimators=i*10,
    random_state=42
)
```

An ensemble of decision trees used for regression.

**4. Ridge Regression**

```python
Ridge(alpha=i*0.2)
```

Linear regression with L2 regularization.

**5. Lasso Regression**

```python
Lasso(alpha=i*0.2)
```

Linear regression with L1 regularization.

**6. Gradient Boosting Regressor**

```python
GradientBoostingRegressor(
    n_estimators=i*10,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

A boosting-based regression algorithm.

## Model Comparison

One of the main features of the project is automatic comparison between different algorithms and configurations.

The application does not simply train one model.

It evaluates multiple models using different train/test split ratios.

The split ratio is tested from:

```
5%
10%
15%
20%
...
90%
```

For example:

```python
train_test_split(
    x,
    y,
    test_size=i * 0.05,
    random_state=42
)
```

The model score is then calculated using:

```python
model.score(x_test, y_test)
```

Whenever a better score is found, the application stores information about the best configuration.

The stored information includes:

- Best Model
- Best Score
- Best Train/Test Split
- Best Number of Estimators
- Best Alpha
- Best Number of Neighbors

depending on the selected algorithm.

## Data Preprocessing

Before training, the application performs several preprocessing operations.

### Column Selection

The application first reduces the dataset to the selected input and output columns.

```python
selected_columns = selected_input_columns + [
    selected_output_column
]

df = df[selected_columns]
```

### Categorical Value Filtering

For object-type columns, the application calculates the percentage of each unique value.
Values occurring at or below the 2.5% threshold are filtered out.

```python
value_percents = df[col].value_counts(
    normalize=True
) * 100

valid_values = value_percents[
    value_percents > 2.5
].index
```

This helps remove very rare categorical values from the training data.

### Removing Single-Value Columns

If a categorical column contains only one unique value, it is removed because it does not provide useful variation for the model.

```python
if df[col].nunique() == 1:
    df.drop(col, axis=1, inplace=True)
```

### Missing Numerical Values

Numerical missing values are initially filled using the column mean.

```python
df[col] = df[col].fillna(
    df[col].mean()
)
```

### Label Encoding

Categorical columns are converted into numerical values using Scikit-learn's LabelEncoder.

```python
le = LabelEncoder()

df[col] = le.fit_transform(df[col])

encoders[col] = le
```

The encoders are stored so that categorical values can later be converted when making predictions.

### Missing Values After Encoding

The application also checks numerical columns for remaining missing values and fills them using the mode.

```python
df[col] = df[col].fillna(
    df[col].mode()[0]
)
```

## Prediction System

After training, the application provides a separate prediction window.

The prediction interface dynamically generates input fields based on the selected feature columns.

For example:

- Age
- Gender
- Salary
- Experience

The application automatically determines whether a feature is categorical or numerical.

### Categorical Prediction Inputs

For encoded categorical columns, the application creates a dropdown menu.
The user can select from the original categories found in the dataset.

Example:

```
Gender

[ Male ▼ ]
```

The selected category is then transformed using the previously fitted LabelEncoder.

### Numerical Prediction Inputs

For numerical columns, the application creates an input field.

Example:

```
Age

[ 25 ]
```

The entered value is converted into a floating-point number before prediction.

### Prediction Process

When the user clicks:

```
Run Prediction
```

the application:

1. Reads the input values.
2. Converts categorical values using stored encoders.
3. Converts numerical values to floats.
4. Creates a Pandas DataFrame.
5. Reconstructs the best-performing model configuration.
6. Splits the original dataset using the selected split.
7. Trains the selected model.
8. Generates the prediction.
9. Converts encoded classification output back to the original class name when applicable.
10. Displays the result.

The output is displayed as:

```
Predicted Output: <result>
```

### Classification Prediction

For classification tasks, if the output column was categorical and encoded using LabelEncoder, the predicted numerical class is converted back to the original category.

Conceptually:

```
Original Class
     ↓
Label Encoding
     ↓
0, 1, 2...
     ↓
Machine Learning Model
     ↓
Predicted Class
     ↓
Inverse Transform
     ↓
Original Class Name
```

This allows the user to see the original class rather than the encoded numerical value.

## Graphical User Interface

The application uses CustomTkinter to create a modern desktop interface.

The interface includes:

- Dark theme
- Sidebar navigation
- Progress indicator
- CSV file selection
- Classification/Regression toggle
- Dynamic column selection
- Dataset configuration
- Training status
- Prediction window
- Reset functionality

The application uses a custom dark UI design with:

- Background
- Surface
- Borders
- Accent Colors
- Status Colors

The interface is designed to provide a guided machine learning workflow rather than requiring users to interact with Python code directly.

### Application Steps

The sidebar contains the following workflow:

```
1. Load Data
2. Select Model Type
3. Select Output
4. Select Inputs
5. Train Data
6. Predict Data
```

The progress bar updates as the user moves through the workflow.

## Project Structure

A simple project structure can be used:

```
AutoML-Predictor/
│
├── main.py
├── README.md
├── requirements.txt
└── datasets/
    └── example.csv
```

If the project is kept as a single Python file, the main application can be stored as:

```
main.py
```

## Installation

### Prerequisites

Make sure Python is installed on your system.
Python 3.9 or newer is recommended.

Check your Python version:

```bash
python --version
```

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd AutoML-Predictor
```

### Install Dependencies

Install the required libraries:

```bash
pip install pandas numpy customtkinter matplotlib seaborn scikit-learn
```

Alternatively, create a requirements.txt file containing:

```
pandas
numpy
customtkinter
matplotlib
seaborn
scikit-learn
```

Then run:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Python file:

```bash
python main.py
```

The AutoML Predictor window should open.

## How to Use

### Step 1: Load Dataset

Click:

```
Open CSV File
```

Select a CSV dataset from your computer.
The application loads the dataset using Pandas.

### Step 2: Select Model Type

Choose:

```
Classifier Mode
```

for classification.
Leave it unchecked for:

```
Regression
```

### Step 3: Select Output Column

The application displays the columns from the dataset.
Select exactly one output column.

For example:

- Age
- Salary
- Experience
- Department

If predicting salary:

```
Output = Salary
```

### Step 4: Select Input Columns

Select the columns that should be used as features.
For example:

- Age
- Experience
- Education
- Department

These become the model's input variables.

### Step 5: Train Model

Click:

```
Train Model
```

The application will:

```
Clean Data
      ↓
Encode Categories
      ↓
Prepare X and Y
      ↓
Train Models
      ↓
Test Different Splits
      ↓
Compare Scores
      ↓
Select Best Model
```

### Step 6: Make Prediction

After training, click:

```
Predict Data
```

A new prediction window opens.
Enter values for each selected input feature.

Then click:

```
Run Prediction
```

The predicted output will be displayed.

## Example Classification Workflow

Suppose the dataset contains:

- Age
- Gender
- StudyHours
- Attendance
- Result

You could configure the application as:

```
Model Type:
Classification

Output:
Result

Inputs:
Age
Gender
StudyHours
Attendance
```

The application will encode categorical values, train classification models, compare their scores, and select the best-performing configuration.

## Example Regression Workflow

Suppose the dataset contains:

- Area
- Bedrooms
- Bathrooms
- Location
- Price

You could configure:

```
Model Type:
Regression

Output:
Price

Inputs:
Area
Bedrooms
Bathrooms
Location
```

The application will preprocess the data and compare regression algorithms.

## Model Selection Logic

The application keeps track of the best score found during training.

Initially:

```python
best_score = -1
```

Whenever a model produces a better score:

```python
if model.score(x_test, y_test) > best_score:
```

the application updates the best configuration.

The selected model is identified internally using a model number.

```
1  Logistic Regression
2  Decision Tree Classifier
3  Random Forest Classifier
4  Ridge Classifier
5  KNN Classifier
6  Gradient Boosting Classifier

7  Linear Regression
8  Decision Tree Regressor
9  Random Forest Regressor
10 Ridge Regression
11 Lasso Regression
12 Gradient Boosting Regression
```

## Technologies Used

### Python

The primary programming language used to develop the application.

### Pandas

Used for:

- Reading CSV files
- Data manipulation
- Column selection
- Missing-value handling
- DataFrame creation

### NumPy

Used as part of the numerical/data processing environment.

### Scikit-learn

The main machine learning library.

Used for:

- Label encoding
- Train/test splitting
- Classification
- Regression
- Model training
- Model evaluation

### CustomTkinter

Used to build the desktop GUI.

The application uses CustomTkinter for:

- Buttons
- Frames
- Labels
- Checkboxes
- Input fields
- Dropdowns
- Scrollable frames
- Progress bars
- Windows

### Matplotlib

Imported for data visualization support.

### Seaborn

Imported for statistical visualization support.

## Learning Objectives

This project was developed to gain practical experience with:

- Python
- Pandas
- NumPy
- Data preprocessing
- Missing-value handling
- Categorical encoding
- Feature selection
- Target selection
- Train/test splitting
- Classification
- Regression
- Model comparison
- Hyperparameter experimentation
- Machine learning prediction
- GUI development
- CustomTkinter
- Scikit-learn
- Dynamic user interfaces
- Building an AutoML-style workflow

## Concepts Implemented

### Data Science

- CSV Data Loading
- Data Filtering
- Data Cleaning
- Missing Value Handling
- Categorical Encoding
- Feature Selection
- Target Selection

### Machine Learning

- Classification
- Regression
- Train/Test Split
- Model Training
- Model Evaluation
- Model Comparison
- Parameter Testing
- Prediction

### Software Development

- GUI Development
- Event-Driven Programming
- State Management
- Dynamic Widgets
- Error Handling
- Modular Functions

## Error Handling

The application includes error handling for several operations.
For example, errors during CSV loading are displayed through the application's status area.

Prediction errors are also caught and displayed to the user.

```python
except Exception as e:
    result_label.configure(
        text=f"Error: {str(e)}"
    )
```

## Reset Functionality

The application includes a Reset button that returns the application to its initial state.

Resetting clears:

- Current dataset
- Selected output
- Selected input columns
- Checkboxes
- Progress
- Application state

The user can then start a new machine learning workflow.

## Important Implementation Details

### Random State

The application uses:

```python
random_state=42
```

for supported models and train/test splitting.
This helps produce reproducible results for the same dataset and configuration.

### Training Split Range

The application evaluates test sizes from:

```
5% to 90%
```

in increments of 5%.
This is implemented using:

```python
test_size=i * 0.05
```

### Random Forest Estimators

For Random Forest models, the number of estimators is tested using:

```python
i * 10
```

This allows the application to evaluate different forest sizes.

### Ridge and Lasso Alpha

For Ridge and Lasso models, different alpha values are tested:

```python
i * 0.2
```

### KNN Neighbors

For KNN classification, the application tests:

```
1 to 10 neighbors
```

## Current Limitations

This project is an educational AutoML-style application and has several limitations.

### Evaluation Metric

The application primarily uses:

```python
model.score()
```

for model comparison.

For many classification models, this represents accuracy.

For regression models, it represents the model's R² score.

A production AutoML system should evaluate additional metrics such as:

```
Classification:
Accuracy
Precision
Recall
F1 Score
ROC AUC

Regression:
MAE
MSE
RMSE
R²
```

### No Cross Validation

The current implementation uses a single train/test split for each tested configuration.
It does not currently implement techniques such as:

- K-Fold Cross Validation
- Stratified Cross Validation
- Repeated Cross Validation

### Potential Overfitting

Because multiple train/test split ratios and model parameters are tested against the same evaluation approach, the selected configuration can potentially overfit to the chosen test set.

A more robust AutoML system should use cross-validation and a separate final holdout dataset.

### Label Encoding

Categorical variables are encoded using LabelEncoder.
For some machine learning algorithms, one-hot encoding or other categorical encoding methods may be more appropriate.

### Data Validation

The application expects a reasonably structured CSV dataset.

Advanced validation for:

- Duplicate records
- Extreme outliers
- Incorrect data types
- Invalid values
- Highly correlated features
- Data leakage

is not currently implemented.

### Feature Scaling

The current implementation does not automatically standardize or normalize numerical features.

This can affect algorithms such as:

- Logistic Regression
- Ridge
- Lasso
- KNN

depending on the dataset.

## Future Improvements

Possible improvements include:

### Advanced Data Cleaning

- Duplicate detection
- Advanced outlier detection
- Automated datatype detection
- Advanced missing-value strategies
- Data validation

### Better Model Evaluation

Add:

- Cross-validation
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- MAE
- MSE
- RMSE
- R²

### More Algorithms

Add models such as:

- Support Vector Machine
- XGBoost
- LightGBM
- Naive Bayes
- Extra Trees
- Elastic Net
- Neural Networks

### Feature Engineering

Implement:

- Feature scaling
- One-hot encoding
- Feature selection
- Feature importance
- Polynomial features
- Automated feature engineering

### Visualization

The imported Matplotlib and Seaborn libraries could be extended to provide:

- Correlation heatmaps
- Distribution plots
- Scatter plots
- Feature importance charts
- Confusion matrices
- Prediction vs actual plots
- Model comparison charts

### Model Export

Add the ability to save trained models using:

- Joblib
- Pickle
- ONNX

and load them later without retraining.

### Dataset Reports

Generate automated reports containing:

- Dataset Summary
- Missing Values
- Feature Types
- Model Scores
- Best Model
- Predictions

### Improved AutoML

A future version could automatically determine:

- Problem Type
- Feature Types
- Preprocessing Strategy
- Best Algorithm
- Best Hyperparameters
- Best Evaluation Metric

### Example Future Architecture

A more advanced version could follow:

```
CSV Dataset
     |
     v
Data Profiler
     |
     v
Data Cleaning
     |
     v
Feature Engineering
     |
     v
Preprocessing Pipeline
     |
     v
Cross Validation
     |
     v
Model Training
     |
     v
Hyperparameter Optimization
     |
     v
Model Evaluation
     |
     v
Best Model
     |
     v
Model Export
     |
     v
Prediction
```

## Project Goals

The main goal of this project was to combine concepts from:

```
Data Science
        +
Machine Learning
        +
Python Programming
        +
GUI Development
```

into a single practical application.

Instead of interacting with machine learning algorithms only through Python scripts, the project provides a graphical workflow that allows users to experiment with datasets and models without manually writing the complete training pipeline.

## Why I Built This

This project was created to strengthen my understanding of the complete machine learning workflow.

Rather than focusing only on model training, I wanted to understand how a practical machine learning application handles:

```
Dataset
   ↓
Cleaning
   ↓
Preprocessing
   ↓
Feature Selection
   ↓
Model Selection
   ↓
Training
   ↓
Evaluation
   ↓
Prediction
```

Building the application also helped me combine my Python programming and GUI development skills with my growing interest in AI, Machine Learning, and Data Science.

## License

This project is intended primarily for educational and academic purposes.
You may add an open-source license such as the MIT License if you intend to distribute the project under specific open-source terms.

## Author

Developed as a Python Machine Learning and Data Science project.

Project: AutoML Predictor

Language: Python

GUI: CustomTkinter

Machine Learning: Scikit-learn

Data Processing: Pandas and NumPy

Visualization: Matplotlib and Seaborn

## Conclusion

AutoML Predictor combines data preprocessing, machine learning model training, model comparison, and prediction into a single desktop application.

The project demonstrates how Python and Scikit-learn can be integrated with a graphical interface to create an accessible machine learning workflow.

It serves as a foundation for developing a more advanced AutoML platform with cross-validation, automated feature engineering, advanced visualization, hyperparameter optimization, model persistence, and comprehensive evaluation metrics.
