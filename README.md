# Deploying-on-AWS-a-dockerized-API-to-time-series-prediction-problem

# 1. Objetives

<p align=justify> 1.1. Build and host a predictive model on AWS with Python in 36 hours

<p align=justify> 1.2. Using a dataset of news articles, train a model to predict the average sentiment of the next day.

<p align=justify> 1.3.	Host the model within a free-tier instance on AWS.

<p align=justify> 1.4.	Build an endpoint that accepts parameters from a user and returns a time series of average sentiment values with the final value as a prediction from your model.
  
# 2. Solution

<p align=justify> 2.1. Postgresql stores the sentiment of the articles, summaries, the articles itslef and categorization by several topics. 
 
<p align=justify> 2.2 The endpoint accepts parameters from the user in a request like the following.Those parameters are the input for an ARIMA model

{"hold out samples": 20,
"lag observations": 3,
"degree of differencing": 0,
"moving average window": 0}

<p align=justify> The user can change any of these parameters but be aware that some combinations are computationally expensive, it is just a free-tier ec2 instance. 

<p align=justify> 2.3. After accept the inputs, the API script allow to perform a rolling forecast to re-create the ARIMA model after each new observation is received. Therefore, the model able to adapt to new data easily. 
  
<p align=justify> 2.4.This walk-forward validation is performed in the hold out samples and then finally predict the average sentiment of the articles for the next day.

![image](https://github.com/robeespi/Deploying-on-AWS-a-dockerized-API-to-time-series-prediction-problem/blob/main/solution_diagram.jpeg)

# 3. Results and Pipeline

You can test by using postman at:

http://ec2-54-79-143-239.ap-southeast-2.compute.amazonaws.com/API/PREDICT_AVG_SENTIMENT

<p align=justify>The endpoint accepts parameters from the user in a request like the following.

{"hold out samples": 20,
"lag observations": 3,
"degree of differencing": 0,
"moving average window": 0}

# 3.1. Why arima? 

<p align=justify> All models were tested with a hold out samples (33% of the dataset). 

<p align=justify>Even tough Regularized Regression such as Ridge performed slightly better than ARIMA models, I picked ARIMA model because it can be adapted easily to new data by incorporating each new observation into the model (Autoregressive models have worked better (>1,0,0))

Model | RMSE |  
--- | --- | 
Persistence(Baseline) | 0.124 | 
Autoregressive (X,0,0) | 0.098 | 
ARIMA(X,X,0)| 0.11 | 
Linear Regression | 8*10> | 
Lasso Regression| 0.094 | 
Ridge Regression| 0.087 | 
Decision Tree Regression | 0.11 | 
XGB Regressor | 0.11 | 
Univariate LSTM | 0.092 | 

# 3.2.Pipeline
  
<p align=justify> Trial1 notebook has all the details about connection to the database, EDA, basic feature engineering and performance and experiment of these models

<p align=justify>Some dataframes were inspected by profiling pandas library.There are two html outpus for this purpose. The bigger one couldn´t be uploaded here, but you can pull the images from DockerHub to access it

https://hub.docker.com/repository/docker/robeespi/roblast27


Some EDA activities and basic feature engineering techniques explored:

* Pandas profiling ( They are in the docker container as output2.html and output3.html, output2 is the EDA about the sql query and output3 is the dataframe by grouping the timestamp by day and incorporating category and sector as dummy variables)<p align=justify>

* Lag plots

* Autocorrelation plots

* Plotting Distribution response variable vs variables in the dataset

* Correlations

* Category and Sector as a dummy variable to run regressions

* Feature Importance performance but not conclusive at all

* There are three timestamps on the data, but I picked the timestamp with more distinct observations and longer period of time.


# 4. Future Work

<p align=justify> LSTM univariate approach shows good performance, but showing overfitting. Still room for find suitable hyperparameters. AutoML/DL and/or Multivariate approach by ussing attention mechanism will be explored

