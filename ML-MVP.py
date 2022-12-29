'''
Description: This program uses linear regression to predict a teams n.

Written in its entirety by Kyle Massie

Data scraped from https://www.basketball-reference.com/
'''
import pandas
from sklearn.model_selection import TimeSeriesSplit # only using past data to predict future
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Cleaning Data

data = pandas.read_csv('nba_games.csv',index_col=0)
data = data.sort_values("date")
data = data.reset_index(drop=True)
del data['mp.1']
del data['mp_opp.1']
del data['index_opp']


def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team

data = data.groupby("team",group_keys=False).apply(add_target)

data["target"][pandas.isnull(data["target"])] = 2
data["target"] = data["target"].astype(int, errors="ignore")

nulls = pandas.isnull(data).sum()
nulls = nulls[nulls > 0]
valid = data.columns[~data.columns.isin(nulls.index)]
data = data[valid].copy()

# Feature selection
rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction='forward',cv=split,n_jobs=1)

# Normalzing data
removed_cols = ["season", "date", "won", "target", "team", "team_opp"]
select_cols = data.columns[~data.columns.isin(removed_cols)]
scaler = MinMaxScaler()
data[select_cols] = scaler.fit_transform(data[select_cols])

# Sequential Feature Selector 
sfs.fit(data[select_cols], data["target"])
predictors = list(select_cols[sfs.get_support()])

# Using past data to predict next seasons
def backtest(df, model, predictors, start = 2, step =1):
    # start param means we use first two seasons to backtest
    # step param is by how many seasons we go
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

predications = backtest(data,rr,predictors)
predications = predications[predications["actual"] != 2]
score = accuracy_score(predications["actual"],predications["predication"])

win_percent = data.groupby("home").apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])


# Rolling Averages
df_rolling = data[list(select_cols) + ["won", "team", "season"]]

def find_team_averages(team):
    # use the teams last n games to predict the next game
    rolling = team.rolling(10).mean()
    return rolling

df_rolling = df_rolling.groupby(["team", "season"],group_keys= False).apply(find_team_averages) # onlt using rolling averages of that season

rolling_cols = [f"{col}_10" for col in df_rolling.columns]
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
data = data.copy()

#opponet data
full = data.merge(data[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on = ["team","date_next"],right_on = ["team_opp_next", "date_next"])


# MORE ACCURATE
removed_cols = list(full.columns[full.dtypes == "object"]) + removed_cols
select_cols = full.columns[~full.columns.isin(removed_cols)]
sfs.fit(full[select_cols], full["target"])
pred = list(select_cols[sfs.get_support()])
preded = backtest(full,rr,pred)
score2 = accuracy_score(preded["actual"], preded["predication"])
print(score2)