# ML-NBA-Wins-Loses-Season-Predications
In this program, we will try and predict NBA team's next games wins or losses using differing Machine Learning classification types including
Ridge Regression, Random Forest Classification, and Logistic Regression. 

The Dataset we will be using has been taken from https://www.basketball-reference.com/ on games between 2016-2022. This data set has roughly 150 features and 20,000 games so we will have to use a sequential feature selector to pick out the features that provide the best accuracy. We will use 20 features for the testing output to reduce processing time but each feature the sequential feature selector adds will improve accuracy so, in theory, we should use as many features as possible. 

This program will also only use rolling averages to predict future games. This program will use the first three seasons to backtest (2016,2017,2018) and then use rolling averages of the past 10 games to predict future games. Therefore no predictions will be made on those seasons or any of the first ten games of each season. To get a significantly more accurate score, we can use a higher rolling average. Using the first half of the season as a rolling average, we get an improvement in accuracy of upwards of 3-4%. 

# Machine Learning Classification Types and Performance
The goal of using machine learning is to be able to predict games better than our baseline bets. From watching some basketball we can agree that the home team has an advantage of their opponent so if we figure out how often the home team wins, we can set that as our baseline to beat. By running the program, we have figured out that the home team wins about 57% of the time based on the dataset. Thus if the machine learning algorithms beat this number, we are more accurate than our baseline. 

Ridge Regression Classification
Using an alpha of 1, we get decent performance and one of the highest accuracy.

Random Forest Classification
A more powerful machine learning algorithm than Ridge Regression but takes a much heavier performance hit. Using 20 estimators and a max_depth of 5 running this classification took most of the processing time. For some reason, when I decreased the estimators to 10, I was getting slightly better accuracy but that should be counter intuitive. 

Logistic Regression
The least performance heavy of the ones used is logistic regression. Setting a max iteration of 20,000 gets an accuracy on par with Random Forest Classification. Logistic Regression is easier to understand than the rest, takes the least amount of processing time, and gets decent accuracy. 

# What does this tell us and Why
If you have ever watched the NBA you know that games can be very unpredictable. Teams that should win may lose because a star player was injured or just played games back to back while the other team had been resting for a few days. Teams may also have different strategies such as tanking (losing games on purpose for a better draft pick) or resting star players versus teams. In addition, because of how many players move around from team to team over the offseason and how the draft affects each team, without that data included, using past seasons as a prediction of the future is not as helpful as it should be. 

That is much more data than we have been using in this program and would take dynamically added features for each game / season. While this might give us better accuracy, the main takeaway is that even with a choice of 150 features, the NBA is a fairly hard league to predict. (Thus if you are going to bet, don't do it on the NBA)

# Improvements to be tested
- Dynamic rolling averages (using first 10 games and then adding a game each time one is played)
- Testing KNN and Linear regression


# Display Terminal Output
(Running the program takes ~ 40-50 minutes on my M1 Macbook so be wary)
(Should try Jupyter Notebook to play around with estimators more and reduce processing time)

The home team wins on average 0.5716857978843124. As a rule, our Machine Learning program should be able to beat this number

Using Ridge Classification we are correct 0.6302559414990859% of the time.

Using Random Forest Classification with 20 estimators and a max_depth of 5, we are correct 0.6182586837294333% of the time.

Using Logistic Regression with a max iteration of 20,000, we are correct 0.6214579524680073% of the time.