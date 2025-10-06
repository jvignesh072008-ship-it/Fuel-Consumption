# Fuel-Consumption

AIM:

The aim of this program is to analyze the relationship between vehicle specifications—such as the number of cylinders, engine size, and combined fuel consumption—and their impact on carbon dioxide (CO2) emissions. The program will generate visual scatter plots comparing these variables with CO2 emissions, train regression models to predict CO2 emissions based on selected independent variables (cylinders and fuel consumption), and evaluate model performance across different training and testing data splits. This analysis enables understanding of how vehicle design factors influence emissions and helps in developing predictive models for environmental impact assessment.

PROJECT DESCRIPTION:

This project aims to explore, visualize, and model the relationships between vehicle characteristics—such as number of cylinders, engine size, and combined fuel consumption—and their CO2 emissions. Using the FuelConsumption.csv dataset, scatter plots will be created to reveal patterns and correlations among these variables. Predictive models will be trained with key independent variables like cylinders and fuel consumption to estimate CO2 emissions. The project will experiment with different train-test splits to evaluate model accuracy and robustness. Ultimately, this project will provide comprehensive insights and reliable predictive tools for estimating vehicle emissions based on practical vehicle parameters, contributing to better environmental impact awareness and optimization.

The project encompasses data visualization, feature selection, machine learning model training, evaluation, and reporting, linking all prior steps into a cohesive data science pipeline focused on fuel consumption and emissions analysis.

EQUIPMENTS REQUIRED:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

ALGORITHM:

-Load the FuelConsumption.csv dataset and select relevant features such as CYLINDERS, ENGINESIZE, FUELCONSUMPTION_COMB, and CO2EMISSIONS for analysis.

-Create scatter plot visualizations for (a) CYLINDERS vs CO2EMISSIONS, (b) CYLINDERS and ENGINESIZE vs CO2EMISSIONS with different colors, and (c) CYLINDERS, ENGINESIZE, and FUELCONSUMPTION_COMB vs CO2EMISSIONS with distinct colors to understand variable relationships.

-Prepare independent and dependent variables for machine learning by selecting CYLINDERS and later FUELCONSUMPTION_COMB as predictors and CO2EMISSIONS as the target variable.

-Split the dataset into training and testing sets with configurable ratios to train and evaluate models.

-Train regression models (e.g., linear regression or support vector regression) using the specified independent variables and training data.

-Predict CO2 emissions on testing data and measure model performance using appropriate accuracy or error metrics.

-Iterate training with multiple train-test splits, record the performance results, and compile program code and outputs including screenshots and upload or share the Colab notebook with the author’s name and registration number.

-This covers all parts of the tasks from visualization through modeling to evaluation and reporting.

EXPLANATION:

-Loading and cleaning datasets,

-Extracting and selecting relevant features,

-Visualizing key relationships in the data via plots,

-Splitting data into training and test sets for model building and evaluation,

-Training machine learning models (such as SVM or regression) to predict target variables,

-Assessing model performance quantitatively using accuracy and other metrics,

-Iteratively refining and validating models to inform insights and decisions.

PROGRAM:
```
/*
Program to implement the Predictive Modeling and Analysis of Vehicle CO2 Emissions Based on Fuel Consumption and Engine Specifications
Developed by: VIGNESH J 
RegisterNumber: 25014705
*/

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('FuelConsumption.csv')

# Q1: Scatter plot - Cylinder vs CO2 Emission (green)
plt.figure(figsize=(8,5))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', alpha=0.6)
plt.title('Cylinder vs CO2 Emission')
plt.xlabel('Cylinder')
plt.ylabel('CO2 Emission')
plt.show()

# Q2: Scatter plot - Cylinder vs CO2 Emission and Engine Size vs CO2 Emission (different colors)
plt.figure(figsize=(8,5))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinder vs CO2')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size vs CO2')
plt.title('Cylinder & Engine Size vs CO2 Emission')
plt.xlabel('Cylinder / Engine Size')
plt.ylabel('CO2 Emission')
plt.legend()
plt.show()

# Q3: Scatter plot - Cylinder vs CO2, Engine Size vs CO2, and Fuel Consumption Comb vs CO2 (different colors)
plt.figure(figsize=(8,5))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinder vs CO2')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size vs CO2')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='red', label='Fuel Consumption vs CO2')
plt.title('Cylinder, Engine Size & Fuel Consumption vs CO2 Emission')
plt.xlabel('Cylinder / Engine Size / Fuel Consumption (L/100 km)')
plt.ylabel('CO2 Emission')
plt.legend()
plt.show()

# Function to train & evaluate linear regression model given independent and dependent variables and train_test_ratio
def train_evaluate_model(X, y, train_ratio=0.8):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
    model = LinearRegression()
    model.fit(X_train.values.reshape(-1,1), y_train)
    
    y_train_pred = model.predict(X_train.values.reshape(-1,1))
    y_test_pred = model.predict(X_test.values.reshape(-1,1))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return train_r2, test_r2

# Q4: Train model with CYLINDERS as independent variable predicting CO2EMISSIONS
X_cyl = df['CYLINDERS']
y_co2 = df['CO2EMISSIONS']
train_r2_cyl, test_r2_cyl = train_evaluate_model(X_cyl, y_co2)
print(f"Model with Cylinders - Train R2: {train_r2_cyl:.4f}, Test R2: {test_r2_cyl:.4f}")

# Q5: Train model with FUELCONSUMPTION_COMB as independent variable predicting CO2EMISSIONS
X_fuel = df['FUELCONSUMPTION_COMB']
train_r2_fuel, test_r2_fuel = train_evaluate_model(X_fuel, y_co2)
print(f"Model with Fuel Consumption - Train R2: {train_r2_fuel:.4f}, Test R2: {test_r2_fuel:.4f}")

# Q6: Train models on different train-test ratios and record accuracies for CYLINDERS
ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
results = []

print("Train Ratio | Train R2 (Cylinders) | Test R2 (Cylinders)")
for ratio in ratios:
    tr, ts = train_evaluate_model(X_cyl, y_co2, train_ratio=ratio)
    results.append((ratio, tr, ts))
    print(f"{ratio:<11} | {tr:.4f}            | {ts:.4f}")

print("\nProgram execution completed.")

```

OUTPUT:
<img width="1293" height="586" alt="Screenshot 2025-10-06 213748" src="https://github.com/user-attachments/assets/8aa8cb47-82eb-4396-92cd-9f7757fa8e52" />
<img width="1161" height="595" alt="Screenshot 2025-10-06 213757" src="https://github.com/user-attachments/assets/0ff1e727-9ece-446d-b6b1-b04083ee92e7" />
<img width="1182" height="587" alt="Screenshot 2025-10-06 213805" src="https://github.com/user-attachments/assets/37fb0fc2-c951-4ee8-b6ff-e7c9a55784a1" />
<img width="725" height="227" alt="Screenshot 2025-10-06 213812" src="https://github.com/user-attachments/assets/6c7ed1dd-bd2d-4db0-814f-881c8b52ac38" />



