# 🌍 Disaster Response Classification Project

This project focuses on building a machine learning pipeline that can **classify emergency messages** into multiple categories of disaster response. The dataset used in this project is provided by **Figure Eight**, and includes messages that are mapped to **36 different types of disaster-related responses**.

The primary goal is to create a reliable system that helps emergency responders by accurately classifying messages during real-life disaster scenarios.

## 🧰 Built With

This project is developed using the following tools and technologies:

- Python (Jupyter Notebook)
- Pandas & NumPy – Data manipulation
- scikit-learn – Machine Learning
- Pickle – Model serialization

## 🎯 Business Requirements

The system is designed to meet the following objectives:

- **Train a model** that can classify emergency messages into multiple categories
- **Optimize model performance** using hyperparameter tuning
- Ensure that the model **minimizes prediction errors** and is robust enough for real-world deployment

## 🧹 Data Preparation

The original dataset contained noise and inconsistencies. An **ETL pipeline (Extract, Transform, Load)** was developed to:

- Clean and normalize text messages
- Merge message and category datasets
- Store the cleaned data in a **SQLite database** for further processing

## 🤖 Machine Learning Methodology

A **machine learning pipeline** was built to:

- Tokenize and vectorize the messages
- Train using **Random Forest Classifier**
- Tune hyperparameters with **GridSearchCV**
- Evaluate model performance using a **classification report**

Although both **Random Forest** and **AdaBoost** classifiers were evaluated, the **Random Forest classifier** outperformed AdaBoost and was selected as the final model. It has been saved using `pickle` for future use.

## 🛠️ Setup Instructions

Follow these commands from the project root to set up and run the application:

### 1. Run the ETL Pipeline
This step loads and cleans the data, and saves it into a SQLite database:
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
### 2. Run the ML Pipeline
This step trains the model and exports the classifier as a pickle file:
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
### 3. Launch the Web Application
```bash
python run.py
```

## Contact
Waqar Ahmed - waqar.nu@gmail.com

Project Link: [https://github.com/waqar-ahmed91/Disaster-Response-ML](https://github.com/waqar-ahmed91/Disaster-Response-ML)
