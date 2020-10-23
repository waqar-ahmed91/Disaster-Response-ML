# Disaster Response Pipeline Project
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Business Requirements](#business-requirements)
* [Data Preparation](#data-preparation)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- Built With -->
### Built With
This project is built with Jupyter Notebook using Pandas, Numpy, Matplotlib, Seaborn.
<!-- Business Requirements -->
## Business Requirements

<!-- Data Preparation -->
## Data Preparation


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
