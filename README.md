# Disaster Response Pipeline Project

### Table of Contents 
1. [Description](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline#Description)
2. [Installation](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline#Installation)
3. [File Descriptions](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline#File-Descriptions)
4. [Instructions](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline#Instructions)
5. [Results](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline#Results)
6. [Licensing, Authors, and Acknowledgements](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline#Licensing)


### Description 

This project is a collaboration project about disaster response pipeline between Udacity who provided the structure and Figure Eight who have provided the data. The datasets contained the tweets and queries about disasters that happened in real life which later will be classified into categories and visualized by using Natural Language Processing and visualization tools. 

There are three segments in this project: 
1. Performing ETL (Extract, Transform and Load) into the datasets  
2. Training the classification model 
3. Running the web app which shows the classification and visualization results 


### Installation
This project was created and run using Python version 3.0.

Plugins and imports used were: 
1. Pandas, MatplotLib, SKlearn.
2. Natural Language Process (NLTK). 
3. SQLlite Database (SQLalchemy). 
4. Flask, Plotly. 


### File Descriptions 
1. Image folder contained the images for the README 
2. App folder contained the run.py in order to run the web 
3. Data folder contained the csvs and database as well as py file
4. Models folder contained pkl and py file for the model training  
5. README document


### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Results 

1. Entering the example message at the query bar. 

![classification query](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline/blob/master/Image/classification%20query.png)



2. Classifications result 

![classification result](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline/blob/master/Image/classification%20result.png)



3. Visualizations of overall datasets 

![Distribution of Message Genres](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline/blob/master/Image/Distribution%20of%20Message%20Genres.png)

![Top 10 Categories](https://github.com/audichandra/Udacity_Disaster_Respone_Pipeline/blob/master/Image/Top%2010%20Categories.png)


### Licensing, Authors, and Acknowledgements

- Authors: [Audi Chandra](https://github.com/audichandra)
- License: [MIT](https://opensource.org/licenses/MIT)
- Acknowledgements 1: [Udacity](https://www.udacity.com/) and its communities in providing and supporting the completion of this project for Data Science Nanodegree 
- Acknowledgements 2: [Figure Eight](https://www.figure-eight.com/)in providing the daatsets and explanation 

