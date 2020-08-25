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

### File description:

    data
 
      disaster_categories.csv: dataset including all the categories
      disaster_messages.csv: dataset including all the messages
      process_data.py: ETL pipeline scripts to read, clean, and save data into a database
      DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
      
   models
      train_classifier.py: machine learning pipeline scripts to train and export a classifier
      classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer

   app
      run.py: Flask file to run the web application
      templates contains html file for the web applicatin
