# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project motivation:
In this project, I did some initial study on the  natural language processing, I applied some machine learning skills to analyze message data that people sent during disasters to build a model for an API that classifies disaster messages. These messages could potentially be sent to appropriate disaster relief agencies.

### Results:
An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database. A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset. A Flask app was created to show data visualization and classify the message that user enters on the web page.
