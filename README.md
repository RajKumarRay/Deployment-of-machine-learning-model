# Deployment-of-machine-learning-model

https://rajmlcarprice-api.herokuapp.com/

ABSTRACT

The car industries collect huge amounts of data that contain some hidden information, which is useful for making effective decisions. For providing appropriate results and making effective decisions on data, some machine learning techniques are used. In this study, an Effective Car Price Prediction System is developed using different machine learning algorithm for predicting the price of the car. The system uses 9 parameters such as Car_name, Year, Selling price, presnt_price, kms_driven, Fuel_type, seller_type, Transmission, Owner for prediction. This model predicts the price of car. We have employed the Python programming for implementing the ML Algorithms. The obtained results have illustrated that the designed model system can effectively predict the car price.

Keywords: Car Price Prediction using Linear Regression and Deployment of the Model.





INTRODUCTION
Machine Learning is a field of technology developing with immense abilities and applications in automating tasks, where neither human intervention is needed nor explicit programming. The power of ML is such great that we can see its applications trending almost everywhere in our day-to-day lives. ML has solved many problems that existed earlier and have made businesses in the world progress to a great extend. Today, we’ll go through one such practical problem and build a solution (model) on our own using ML.
What’s exciting about this? Well, we will deploy our built model using Flask and Heroku applications. And in the end, we will have fully working web applications in our hands.

1.1 PROBLEM DEFINITION
Imagine a situation where you have an old car and want to sell it. You may of course approach an agent for this and find the market price, but later may have to pay pocket money for his service in selling your car. But what if you can know your car selling price without the intervention of an agent. Or if you are an agent, definitely this will make your work easier. Yes, this system has already learned about previous selling prices over years of various cars.
So, to be clear, this deployed web application will provide you will the approximate selling price for your car based on the fuel type, years of service, showroom price, the number of previous owners, kilometers driven, if dealer/individual, and finally if the transmission type is manual/automatic. And that’s a brownie point. Any kind of modifications can also be later inbuilt in this application. It is only possible to later make a facility to find out buyers. This a good idea for a great project you can try out. You can deploy this as an app like OLA or any e-commerce app. The applications of Machine Learning don’t end here. Similarly, there are infinite possibilities that you can explore. But for the time being, let me help you with building the model for Car Price Prediction and its deployment process.






1.2 MOTIVATION FOR THE WORK
The main motivation of making this model is that to ease the work of us. During such a tight schedule, it is not possible for most of people to find an agent and then find the price of the car based on the current standard. So this model will help all the people who want to know the price of their old car. This model is deployed on online so it can be used by the people.  

Although these are commonly used machine learning algorithms, the car price prediction is a vital task involving highest possible accuracy. Hence, the three algorithms are evaluated at numerous levels and types of evaluation strategies. This will provide researchers and practitioners to establish a better.

Learning algorithms has motivated this work. This paper contains a brief literature survey. An efficient Car prediction has been made by using various algorithms some of them include Linear Regression, XGboost Regressor , Random Forest Regression Etc. It can be seen in Results that each algorithm has its strength to register the defined objectives






CHAPTER 2:

SYSTEM ANALYSIS AND DESIGN

2.2 PROPOSED SYSTEM

The objective of the proposed system technique is to use ensemble techniques to improve the performance of predicting heart disease. Figure1describes the architecture of the proposed system. It is structured into six stages, including data collection, data preprocessing, and feature selection, data splitting, training models, and evaluating models.
The data mining techniques-based systems could have a crucial impact on the employees’ lifestyle to predict heart diseases. There are many scientific papers, which use the techniques of data mining to predict heart diseases. However, limited scientific papers have addressed the four cross-validation techniques of splitting the data set that plays an important role in selecting the best technique for predicting heart disease.
It is important to choose the optimal combination between the cross-validation techniques and the data mining, classification techniques that can enhance the performance of the prediction models. This paper aims to apply the four-cross-validation techniques (holdout, k-fold cross- validation, stratified k fold cross-validation, and repeated random) with the eight data mining, classification techniques to improve the accuracy of heart disease prediction and select the best prediction models.


2.2.1 DATA COLLECTION

In this Dataset I got from the Kaggle. As well as here I mentioned some of the things about the dataset like features. The goal of this project is to create a regression model that is able to accurately estimate the price of the car given the features.
![image](https://user-images.githubusercontent.com/90521823/169537321-3b928fa3-e584-4f93-a882-c4d2b1300c6d.png)


METHODS AND ALGORITHMS USED

The main purpose of designing this system is to predict the Car price. We have used Linear Regression, XGboost and Random Forest Regressor as a machine-learning algorithm to train our system.

3.1 LINEAR REGRESSION

Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables they are considering, and the number of independent variables getting used.

Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y (output). Hence, the name is Linear Regression. In the figure above, X (input) is the work experience and Y (output) is the salary of a person. The regression line is the best fit line for our model.

While training the model we are given: 
x: input training data (univariate – one input variable (parameter))
y: labels to data (supervised learning)
When training the model – it fits the best line to predict the value of y for a given value of x. The model gets the best regression fit line by finding the best θ1 and θ2 values.
θ1: intercept θ2: coefficient of x
Once we find the best θ1 and θ2 values, we get the best fit line. So when we are finally using our model for prediction, it will predict the value of y for the input value of x.
 

 
Fig 3.1 Linear Regression Algorithm

HOW TO UPDATE Θ1 AND Θ2 VALUES TO GET THE BEST FIT LINE?
COST FUNCTION:
By achieving the best-fit regression line, the model aims to predict y value such that the error difference between predicted value and true value is minimum. So, it is very important to update the θ1 and θ2 values, to reach the best value that minimize the error between predicted y value (pred) and true y value (y).
GRADIENT DESCENT: 
To update θ1 and θ2 values in order to reduce Cost function (minimizing RMSE value) and achieving the best fit line the model uses Gradient Descent. The idea is to start with random θ1 and θ2 values and then iteratively updating the values, reaching minimum cost.

3.2	RANDOM FOREST
Random Forest is a supervised learning algorithm. It is an extension of machine learning classifiers which include the bagging to improve the performance of Decision Tree.
It can be used for both classification and regression problems. As we know that a forest is made up of trees and more trees means more robust forest. Similarly, random forest algorithm creates decision trees on data samples and then gets the prediction from each of them and finally selects the best solution by means of voting. It is an ensemble method which is better than a single decision tree because it reduces the over-fitting by averaging the result. It combines tree predictors, and trees are dependent on a random vector which is independently sampled. The distribution of all trees is the same. Random Forests splits nodes using the best among of a predictor subset that are randomly chosen from the node itself, instead of splitting nodes based on the variables.

The time complexity of the worst case of learning with Random Forests is O (M (dnlogn)), where M is the number of growing trees, n is the number of instances, and d is the data dimension. An instance is classified by starting at the root node of the tree, testing the attribute specified by this node, and then moving down the tree branch corresponding to the value of the attribute as shown in the above figure. This process is then repeated for the sub tree rooted at the new node.
  
Fig 3.2 Random forest Regressor


ALGORITHM RANDOM FOREST:

Step 1 − First, start with the selection of random samples from a given dataset.
Step 2 − Next, this algorithm will construct a decision tree for every sample. 
Step 3 − in this step, voting will be performed for every predicted result.
Step 4 − At last, select the most voted prediction result as the final prediction result. 




The following diagram will illustrate its working:

           Fig 3.3 Working of Random Forest Algorithm
 

3.3FLASK FRAMEWORK:
What we need is a web application containing a form to take the input from the user, and return the predictions from the model. So we’ll develop a simple web app for this. The front end is made using simple HTML and CSS. I advise you to go through the basics of web development to understand the meaning of code written for the front end. It would be also great if you have an understanding of the flask framework.
Let me explain to you, in brief, what I have coded using FLASK.
So let’s start the code by importing all the required libraries used here.
From flask import Flask, render template, request
Import pickle
Import requests
Import numpy as np


DEPLOYING USING HEROKU:
All you need to do is connect your Github repository containing all necessary files for the project with Heroku. For all those who don’t know what Heroku is, Heroku is a platform that enables developers to build, run, and operate applications in the cloud.
This is the link to the web app that I have created using the Heroku platform. So, we have seen the building and deployment process of a machine learning model. You can also do it, learn more and never hesitate to try out new things and develop.

