'''
Description: This program uses machine learning to predict a whether an nba team will win or lose its next game.
Written by Kyle Massie
Data from https://www.basketball-reference.com/
'''

import pandas
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Cleaning Data
def preprocess(file_name):
    '''
    args:
    file_name: the name of the file we will be analyzing

    returns:
    data: the complete pandas data frame after going through the preprocessing steps

    This function takes in the file name of csv we will be using a puts the data through a pre-processing stage.
    First we removed unneccesary features and then we add the target, which is if the team won or lost its next game. 

    '''
    data = pandas.read_csv(file_name,index_col=0)
    data = data.sort_values("date")
    data = data.reset_index(drop=True)
    del data['mp.1']
    del data['mp_opp.1']
    del data['index_opp']

    # Adding target value for if that team one the next game
    def add_target(team):
        team["target"] = team["won"].shift(-1)
        return team
    data = data.groupby("team",group_keys=False).apply(add_target)

    # If there is no next game, that team did not lose or win their next game. Instead of null, we will input a value of 2
    data["target"][pandas.isnull(data["target"])] = 2
    data["target"] = data["target"].astype(int, errors="ignore")

    nulls = pandas.isnull(data).sum()
    nulls = nulls[nulls > 0]
    valid = data.columns[~data.columns.isin(nulls.index)]
    data = data[valid].copy() # reduces splicing 

    return(data)


# Normalizing x value data set into a range between 0-1 like target
def normalize(data):
    '''
    args:
    data: the pandas data frame

    returns:
    x_data: the x values for machine learning 
    removed_cols: the columns of features that we do not want to be 

    This function takes in the data frame and returns the x values we want to train and test with along with the columns we want removed.

    '''
    removed_cols = ["season", "date", "won", "target", "team", "team_opp"]
    x_data = data.columns[~data.columns.isin(removed_cols)]
    scaler = MinMaxScaler()
    data[x_data] = scaler.fit_transform(data[x_data])

    return(x_data, removed_cols)

# Using past data to predict next seasons
def backtest(df, model, predictors, start = 3, step =1):
    # start param means we use first three seasons to backtest
    # step param is by how many seasons we go
    '''
    args:
    df: the pandas data frame
    model: the classification type we are using for this backtest
    predicators: the list of predicators / features we are using

    returns:
    predications: the predicated target values

    This function takes the predicator features, data frame, classifaction model and finds the predicated target values for future seasons 

    '''
    all_predications = []
    seasons = sorted(df["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]

        train = df[df["season"] < season]
        test = df[df["season"] == season]

        model.fit(train[predictors],train["target"])
        preds = model.predict(test[predictors])
        preds = pandas.Series(preds,index=test.index)

        combined = pandas.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "predication"]
        all_predications.append(combined)

    return(pandas.concat(all_predications))


def rolling_avg(data,x_data,n_games,removed_cols,sfs,feature_sel):
    '''
    args:
    data: the pandas data frame
    x_data: the x values of the data frame
    n_games: the number of games we will use for our rolling average
    removed_cols: the columns we do not want to use as our features
    sfs: the type of sequential feature selection we will be using
    feature_sel: the type of classification we will be using

    returns:
    score: the accuracy score of this type of classification

    This function returns the inputted classifcation accuracy score.

    '''

    # Rolling Averages
    df_rolling = data[list(x_data) + ["won", "team", "season"]]

    def find_team_averages(team):
    # use the teams last n games to predict the next game
        rolling = team.rolling(n_games).mean()
        return rolling

    df_rolling = df_rolling.groupby(["team", "season"],group_keys= False).apply(find_team_averages) # only using rolling averages of that season

    rolling_cols = [f"{col}_{n_games}" for col in df_rolling.columns]
    df_rolling.columns = rolling_cols
    data = pandas.concat([data,df_rolling], axis=1)
    data = data.dropna()

    def shift_col(team, col_name):
        next_col = team[col_name].shift(-1)
        return(next_col)

    def add_col(data,col_name):
        return data.groupby("team",group_keys=False).apply(lambda team_data: shift_col(team_data, col_name))

    data["home_next"] = add_col(data,"home")
    data["team_opp_next"] = add_col(data, "team_opp")
    data["date_next"] = add_col(data,"date")
    data = data.copy() # reduces splicing

    # adding in opponet data
    full_data = data.merge(data[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on = ["team","date_next"],right_on = ["team_opp_next", "date_next"])


    # Sequential Feature Selection
    removed_cols = list(full_data.columns[full_data.dtypes == "object"]) + removed_cols
    x_data = full_data.columns[~full_data.columns.isin(removed_cols)]

    sfs.fit(full_data[x_data], full_data["target"])
    predicators_set = list(x_data[sfs.get_support()])
    predication = backtest(full_data, feature_sel,predicators_set)

    return(accuracy_score(predication["actual"], predication["predication"]))
    

if __name__ == "__main__":

    data = preprocess(file_name = 'nba.csv')

    # Home percent we need to beat
    win_percent = data.groupby("home").apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])

    # Classification Types

    rr = RidgeClassifier(alpha=1)
    rfc = RandomForestClassifier(n_jobs=-1,max_depth=5,random_state=0,n_estimators=20)
    lg = LogisticRegression(random_state=0,max_iter = 20000)
    split = TimeSeriesSplit(n_splits=5)

    # Sequential Feature Selector

    sfs_rr = SequentialFeatureSelector(rr, n_features_to_select=20, direction='forward',cv=split,n_jobs=1)
    sfs_rfc = SequentialFeatureSelector(rfc, n_features_to_select=20, direction='forward',cv=split,n_jobs=1)
    sfs_lg = SequentialFeatureSelector(lg, n_features_to_select=20, direction='forward',cv=split,n_jobs=1)
    

    # Normalize

    x_data, removed_cols = normalize(data)


    # Accuracy

    Using_First_Ten_Games_RR = rolling_avg(data,x_data,10,removed_cols,sfs = sfs_rr,feature_sel=rr)
    Using_First_Ten_Games_RFC = rolling_avg(data,x_data,10,removed_cols,sfs = sfs_rfc,feature_sel=rfc)
    Using_First_Ten_Games_LOG = rolling_avg(data,x_data,10,removed_cols,sfs = sfs_lg,feature_sel=lg)

    # Display Statements
    
    print(f'\nThe home team wins on average {win_percent[1]}. As a rule, our Machine Learning program should be able to beat this number\n')
    print(f'Using Ridge Classification we are correct {Using_First_Ten_Games_RR}% of the time.\n')
    print(f'Using Random Forest Classification with 20 estimators and a max_depth of 5, we are correct {Using_First_Ten_Games_RFC}% of the time.\n')
    print(f'Using Logistic Regression with a max iteration of 20,000, we are correct {Using_First_Ten_Games_LOG}% of the time.\n')
    