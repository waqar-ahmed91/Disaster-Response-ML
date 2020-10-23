# Disaster Response Pipeline Project
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Business Requirements](#business-requirements)
* [Data Preparation](#data-preparation)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project
This project is about training the messages of dataset that is provided by Figure Eight to predict the disaster responses which are based on 36 different types of disasters.  
<!-- Built With -->
### Built With
This project is built with Jupyter Notebook using Pandas, Numpy, SQL, Sci-kit learn and Plotly.
<!-- Business Requirements -->
## Business Requirements
The requirements of this project is to train the dataset to predict different disaster responses accurately with the best model possible as well as tuning the hyper parameters to make the model somehow perfect to predcit the responses with less error rate.
<!-- Data Preparation -->
## Data Preparation
The data was messy and ETL (Extract, Transform and Load) method has been used to clean the data as much as possible to refine the text messages.

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
