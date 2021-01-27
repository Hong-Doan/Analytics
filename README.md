# Using Machine Learning to Predict Kickstarter Success
## I. Business Problems
Kickstarter, founded in 2009, is one particularly well-known and popular crowdfunding platform. It has an all-or-nothing funding model, whereby a project is only funded if it meets its goal amount; otherwise no money is given by backers to a project.
#### Insights – Overall from 2009 to 2017
<img width="922" alt="Untitled12" src="https://user-images.githubusercontent.com/70985552/106045456-efe0df80-60ae-11eb-9f06-fa36ededaf64.png">

#### Current business situations:
-  Creators and their projects are autonomous
-  If a project fails to meet its goal, no money is exchanged
-  We take only 5% of total money pledge from successful projects 
-  To date, only 37.8% of projects have succeeded
#### COVID-19 impact:
-  Negative trend active projects trend: April 2020: 25% fewer active projects compared to previous year; August 2020: 7% fewer active projects compared to previous year 
-  Pandemic delays project launches and product shipping: In May 2020, the company was forced to lay off almost 40% of its workforce
## II. Approach: How can we increase the number of successful projects
A huge variety of factors contribute to the success or failure of a project — in general, and also on Kickstarter. Some of these are able to be quantified or categorized, which allows for the construction of a model to attempt to predict whether a project will succeed or not.

The model will be used to predict the likelihood a kickstarter project will succeed or fail before its actual deadline. It will also search for any features that influence the success or failure of Kickstarter projects. That is why we are going to focus only on data whose state is 'successful' or 'failed'.

The notebook is divided into 3 parts. In Part 1, we are going to clean data, visualize some patters, and perform data analysis. In part 2, we are going to prepare data for before modelling. In Part 3, we will create a simple model to predict, if a project is going to be successful or not based on chosen features.

In conclusion, we provide our findings and appilcation.
## III. Methodology
### 1. Dataset
Source: Kickstart project from 2009 to 2017

I remove features that are leaking our label so that our classifier will be able to predict success of project right after it's launch. We end up with following features:
kick1: The columns which were kept or calculated were:

**DataFrame**

Int64Index: 331462 entries, Data columns (total 11 columns):

| #  | Column | Non-Null Count | Dtype |    
| ---- |----- | ---- | ------ |
| 0 | mainCategory | 331462 non-null | object |      
| 1 | currency | 331462 non-null  | object |     
| 2 | launched | 331462 non-null | datetime64 | 
| 3 | state |  331462 non-null | object |       
| 4 | backers |  331462 non-null | int64 |      
| 5 | country |  331462 non-null | object |       
| 6 | usdPledgedReal | 331462 non-null | float64 |      
| 7 | usdGoalReal | 331462 non-null | float64 |      
| 8 | duration | 331462 non-null | int64 |       
| 9 | yearlaunched | 331462 non-null | int64 |        
| 10 | monthlaunched | 331462 non-null | int64 |       

dtypes: datetime64 (1), float64(2), int64(4), object(4), memory usage: 30.3+ MB

### 2. Data preparation  

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

### 3. Data Exploration: Which effects on projects that were successful?

The graphs below show how various features differ between failed and successful projects.
- Unsurprisingly, successful projects tend to have smaller (and therefore more realistic) goals - the median amount sought by successful projects is about half that of failed projects
- The differences in the median amount pledged per project are more surprising. The median amount pledged per successful project is considerably higher than the median amount requested, suggesting that projects that meet their goal tend to go on to gain even more funding, and become 'over-funded'
- Longer projects are not necessarily better. Successful projects have slightly shorter durations. Actually, Kickstarter also noticed it and at some point limited their projects durations to up to 60 days. Project duration of 32-35 days is optimal - on the bottom right graph we can see the percent of projects funded as a function of their duration

<img width="835" alt="Untitled14" src="https://user-images.githubusercontent.com/70985552/106050341-1ace3200-60b5-11eb-8ea6-91bf6ce5d908.png">

### 4. Prediction: Machine Learning Model
#### (1) Model Selections

**Machine learning models considered**

* Logistic Regression (LR)
* Support Vector Machine (SVC)
* K-Nearest Neighbors (KNN)
* Random Forest  (RF)

**Analyze the results**

| Model Name|	LR | SVM	| KNN	| RF |
| ---- |----- | ---- | ------ | ------- |
| Accuracy | 0.9170	| 0.9070 |	0.9160 |	0.9070 |
| Precision	| 0.9211	| 0.9231	| 0.9050	| 0.9052 |
| Recall	| 0.8851	| 0.8828	| 0.8759	| 0.8782 |
| F1 | Score	| 0.9027	| 0.9025	| 0.8759	| 0.8915 | 
| ROC and AUC	| 0.9133	| 0.9131	| 0.9025	| 0.9037 |

**How to evaluate model performances and select the best model?**
It is important to evaluate both the precision and recall of a model. Thus, it makes sense to combine the precision and recall metrics; the common approach for combining these metrics is known as the f-score (weighted average F1 score): (1) Dataset is slightly imbalanced and (2) Balance trade-off between precision and recall

The F1 score calculates the harmonic mean between precision and recall and is a suitable measure because there is no preference for false positives or false negatives in this case (both are equally bad). The weighted average will be used because the classes are of slightly different sizes, and we want to be able to predict both successes and failures.

**The winning model**
`Logistic Regression`

#### (2) Algorithm Tuning Model
* Parameter optimization using GridSearchCV 
* New predictions with 0.4 threshold

#### (3) Evaluate the Selected Model
<img width="864" alt="Untitled3" src="https://user-images.githubusercontent.com/70985552/106018249-50602480-608f-11eb-9704-4e913b0a8cba.png">

## IV. Recommendation
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

## V. Conclusions
The final and best model is Logistic Regression, it is best at predicting the odds of a project either failing or succeeding.

Since we are using 'backers' as a variable, this model would ideally be run continously throughout a project's duration. When used before a project launch, with backers = 0, the USD Goal variable will pull the odds of a project to 'failing' until it gains enough backers. This can still be useful as we can plug in possible values for backers to find the ideal number to swing the project's odds more to succeeding.

Further, we can run the model on a periodic schedule for live projects and provide odds for success. At these checkpoint dates, we can identify projects that are likely to fail early and allocate resources to promote the project.

#### Application
Instead of outright classification, this problem was best tackled with probabilities due to the nature of the question: What makes a project successful? Intuitively, no project on kickstarter is a guaranteed success, but there are ones that are more likely to succeed before project launch: those with moderate goals and are performance arts-focused.

As a project launches and attracts backers, a threshold is reached due to the number of backers and the probability of success becomes greater than failure. The model also puts more weight on backers, so a one unit change in backers (log), holding every else constant, increases the odds of success by a greater margin than the negative impact of USD goal (log). Projects that are not performance focused, couple with excessively high goals, are starting the race farther from the finish line. Running this model before a project launch can tell us how much disadvantage a project has incurred and project the number of backers it will take to overcome those disadvantages.

Further, we can run the model on a periodic schedule for live projects and provide odds for success. At these checkpoint dates, we can identify projects that are likely to fail early and allocate resources to promote the project, either through adjustment of the marketing strategy or a re-evaluation of project goals.

## VI. End
Contribution
| | Note	 | 
| ---- |----- | 
| Date | 12/10/2020	|
| Author	| Hong Doan	| 
| Teammate | Rina Calderon, Beth Castaneda, Harley Hayden | 


