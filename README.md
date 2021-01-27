# Using machine learning to predict Kickstarter success
### I. Business Problems
Kickstarter, founded in 2009, is one particularly well-known and popular crowdfunding platform. It has an all-or-nothing funding model, whereby a project is only funded if it meets its goal amount; otherwise no money is given by backers to a project.
##### Insights – Overall from 2009 to 2017
<img width="1047" alt="Untitled7" src="https://user-images.githubusercontent.com/70985552/106020165-75559700-6091-11eb-949e-09c046034953.png">

1. Current business situations:
-  Creators and their projects are autonomous
-  If a project fails to meet its goal, no money is exchanged
-  We take only 5% of total money pledge from successful projects 
-  To date, only 37.8% of projects have succeeded
2. COVID-19 impact:
-  Negative trend active projects trend: April 2020: 25% fewer active projects compared to previous year; August 2020: 7% fewer active projects compared to previous year 
-  Pandemic delays project launches and product shipping: In May 2020, the company was forced to lay off almost 40% of its workforce
### II. Approach: How can we increase the number of successful projects
A huge variety of factors contribute to the success or failure of a project — in general, and also on Kickstarter. Some of these are able to be quantified or categorized, which allows for the construction of a model to attempt to predict whether a project will succeed or not.

The model will be used to predict the likelihood a kickstarter project will succeed or fail before its actual deadline. It will also search for any features that influence the success or failure of Kickstarter projects. That is why we are going to focus only on data whose state is 'successful' or 'failed'.

The notebook is divided into 3 parts. In Part 1, we are going to clean data, visualize some patters, and perform data analysis. In part 2, we are going to prepare data for before modelling. In Part 3, we will create a simple model to predict, if a project is going to be successful or not based on chosen features.

In conclusion, we provide our findings and appilcation.
### III. Methodology
1. Dataset: 	Kickstart project from 2009 to 2017
2. Data preparation  

Some features were initially retained for exploratory data analysis (EDA) purposes, but were then dropped in order to use machine learning models. These included features that are related to outcomes (e.g. the amount pledged) rather than related to the properties of the project itself (e.g. category, goal, length of campaign).Country can be explained by Currency i.e. Euro is used by European countries, Pounds for Great Britain, Dollar in USA, etc.

Based on the log statistics, we can observe that goal is highly skewed i.e. difference between 75% percentile and max value is very high. Distribution of the goal variable. We can observe goal is highly skeweed to the right. Log transformation can solve the outlier problem.
```
kick2['usdGoalRealLog'] = np.log1p(kick2.usdGoalReal)
# View the original usdGoalReal and the log-transformed one
dims = (10, 4)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=dims)
sns.distplot(kick2.usdGoalReal, ax=ax1)
sns.distplot(kick2.usdGoalRealLog, ax=ax2);
```
<img width="734" alt="Untitled8" src="https://user-images.githubusercontent.com/70985552/106023154-89e75e80-6094-11eb-8c4d-aa92328ef4a1.png">

3. Data Exploration: Which effects on projects that were successful?
4. Prediction: Machine learning models

* Model Selections
<img width="671" alt="Untitled12" src="https://user-images.githubusercontent.com/70985552/106026000-4fcb8c00-6097-11eb-9551-c00fdf923821.png">

* Algorithm Tuning Model: by Parameter optimization using GridSearchCV and New predictions with 0.4 threshold
* Evaluate the Selected Model
<img width="864" alt="Untitled3" src="https://user-images.githubusercontent.com/70985552/106018249-50602480-608f-11eb-9704-4e913b0a8cba.png">

### IV. Recommendation
1. Adapt business model: Charge 1% ‘incentive fee’
2. Summer highlights campaign
- June: Technology 
- July: Journalism
- August: Food
3. Focus on the factors that are the most important
- [x] Backer is the most important feature 
- [ ] Larger goals negatively impact successful projects
- [x] Projects in the categories of Dance, Theater, Music, F&V are more likely to be successful
- [ ] Projects in Game category are more likely to fail

### V. Conclusions
The final and best model is Logistic Regression, it is best at predicting the odds of a project either failing or succeeding.

Since we are using 'backers' as a variable, this model would ideally be run continously throughout a project's duration. When used before a project launch, with backers = 0, the USD Goal variable will pull the odds of a project to 'failing' until it gains enough backers. This can still be useful as we can plug in possible values for backers to find the ideal number to swing the project's odds more to succeeding.

Further, we can run the model on a periodic schedule for live projects and provide odds for success. At these checkpoint dates, we can identify projects that are likely to fail early and allocate resources to promote the project.

#### Application
Instead of outright classification, this problem was best tackled with probabilities due to the nature of the question: What makes a project successful? Intuitively, no project on kickstarter is a guaranteed success, but there are ones that are more likely to succeed before project launch: those with moderate goals and are performance arts-focused.

As a project launches and attracts backers, a threshold is reached due to the number of backers and the probability of success becomes greater than failure. The model also puts more weight on backers, so a one unit change in backers (log), holding every else constant, increases the odds of success by a greater margin than the negative impact of USD goal (log). Projects that are not performance focused, couple with excessively high goals, are starting the race farther from the finish line. Running this model before a project launch can tell us how much disadvantage a project has incurred and project the number of backers it will take to overcome those disadvantages.

Further, we can run the model on a periodic schedule for live projects and provide odds for success. At these checkpoint dates, we can identify projects that are likely to fail early and allocate resources to promote the project, either through adjustment of the marketing strategy or a re-evaluation of project goals.

#### END


