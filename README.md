# Disaster Response Pipeline Project
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Business Requirements](#business-requirements)
* [Data Preparation](#data-preparation)
* [Machine Learning Method](#machine-learning-method)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project
This project is about training the messages of dataset that is provided by Figure Eight to predict the disaster responses which are based on 36 different types of disasters.  
<!-- Built With -->
### Built With
This project is built with Jupyter Notebook using Pandas, Numpy, SQL, Sci-kit learn, Pickle and Plotly.
<!-- Business Requirements -->
## Business Requirements
The requirements of this project is to train the dataset to predict different disaster responses accurately with the best model possible as well as tuning the hyper parameters to make the model somehow perfect to predcit the responses with less error rate.
<!-- Data Preparation -->
## Data Preparation
The data was messy and ETL (Extract, Transform and Load) method has been used to clean the data as much as possible to refine the text messages.
<!-- Machine Learning Method -->
## Machine Learning Method
After cleaning the data Machine Learning Pipeline is used to train the dataset with different features and Random Forest classifier as well as hyper parameter tuning is used to refine the model to predict the responses with accuracy and less error rate. Another method AdaBoost classifier is also used to train the data but Random Forest classifier classification report is turned out better than AdaBoost classifier so the Random Forest classifier model has been saved in a pickle file for future use.
<!-- CONTACT -->
## Contact
Waqar Ahmed - waqar.nu@gmail.com
Project Link: [https://github.com/waqar-ahmed91/Disaster-Response-ML](https://github.com/waqar-ahmed91/Disaster-Response-ML)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
