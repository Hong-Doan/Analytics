# Using machine learning to predict Kickstarter success
### Business Problems
Kickstarter, founded in 2009, is one particularly well-known and popular crowdfunding platform. It has an all-or-nothing funding model, whereby a project is only funded if it meets its goal amount; otherwise no money is given by backers to a project.
1. Current business situations:
-  Creators and their projects are autonomous
-  If a project fails to meet its goal, no money is exchanged
-  We take only 5% of total money pledge from successful projects 
-  To date, only 37.8% of projects have succeeded
2. COVID-19 impact:
-  Negative trend active projects trend: April 2020: 25% fewer active projects compared to previous year; August 2020: 7% fewer active projects compared to previous year 
-  Pandemic delays project launches and product shipping: In May 2020, the company was forced to lay off almost 40% of its workforce
### Approach: How can we increase the number of successful projects
A huge variety of factors contribute to the success or failure of a project — in general, and also on Kickstarter. Some of these are able to be quantified or categorized, which allows for the construction of a model to attempt to predict whether a project will succeed or not.

The model will be used to predict the likelihood a kickstarter project will succeed or fail before its actual deadline. It will also search for any features that influence the success or failure of Kickstarter projects. That is why we are going to focus only on data whose state is 'successful' or 'failed'.

The notebook is divided into 3 parts. In Part 1, we are going to clean data, visualize some patters, and perform data analysis. In part 2, we are going to prepare data for before modelling. In Part 3, we will create a simple model to predict, if a project is going to be successful or not based on chosen features.

In conclusion, we provide our findings and appilcation.
### Methodology
* Dataset: 	Kickstart project from 2009 to 2017
* Data preparation and exploration insights: Which effects on projects that were successful? 
* Prediction: Machine learning models
<img width="835" alt="Untitled5" src="https://user-images.githubusercontent.com/70985552/106018737-ded4a600-608f-11eb-95b7-245a08493632.png">

* Algorithm Tuning Model by Parameter optimization using GridSearchCV and New predictions with 0.4 threshold
* Evaluate the selected Model
<img width="864" alt="Untitled3" src="https://user-images.githubusercontent.com/70985552/106018249-50602480-608f-11eb-9704-4e913b0a8cba.png">

### Recommendation
<img width="762" alt="recomend" src="https://user-images.githubusercontent.com/70985552/106019521-bdc08500-6090-11eb-9771-e2e21ff9e2ae.png">
