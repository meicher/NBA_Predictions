
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from selenium import webdriver
from datetime import date    
today = date.today().isoformat()

import warnings
warnings.simplefilter('ignore')


# ### Training the Model

# In[2]:

#append historical lines data to this file
alldata = pd.read_csv('final.csv')
target = alldata['Diff']
target_classifier = alldata['Win']

#Identify columns to keep and train on
colsToTrain = ['Home', 'B2B','Line','Opp.Line','Opp.B2B','mean_Diff', 'mean_Win','mean_Diff_Opp','mean_FGA','mean_FGA_Opp','mean_PTS','mean_PTS_Opp',
       'mean_Win_Opp', 'mean_FTA', 'mean_FTA_Opp','NET_Rtng', 'NET_Rtng_Opp','AstTO','Ast%','AstTO_Opp','Ast%_Opp']

train = alldata[colsToTrain]


# #### LGB Model

# In[47]:

#%%time

#LGB model
#model = lgb.LGBMRegressor(num_leaves=300,num_trees=4000,objective='regression',metric='mae',learning_rate=0.05,random_state=42)

#split data
#X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.33,random_state=42)

#Scaling data has no affect on tree based models since they can have varying magnitudes.

#predict and store predictions
#model.fit(X_train, y_train)
#lgb_pred = model.predict(X_test)
#mean_absolute_error(y_test,lgb_pred)


# #### NN Model

# In[49]:

#%%time

#NN model
#model_NN = MLPRegressor(hidden_layer_sizes=(10,10,10,10,10),random_state=42)

#split data
#X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.33,random_state=42)
#Line = X_test['Line']
#OppLine = X_test['Opp.Line']

#normalization
#norm = Normalizer()
#X_train = pd.DataFrame(norm.fit_transform(X_train),columns=X_train.columns)
#X_test = pd.DataFrame(norm.fit_transform(X_test),columns=X_test.columns)


#predict and store predictions
#model_NN.fit(X_train, y_train)
#nn_pred = model_NN.predict(X_test)


# In[50]:

#mean_absolute_error(y_test,nn_pred)

#ensemble prediction - average of lgb model and neural network model - PRODUCES LESS ACCURATE PREDICTION
#preds = pd.DataFrame({'lgb':lgb_pred,'nn':nn_pred})
#preds['ensemble'] = preds.mean(axis=1)
#mean_absolute_error(y_test,preds['ensemble'])


# ### Classification Model

# In[3]:

#NN model
model_LR = LogisticRegression()
#split data
X_train,X_test,y_train,y_test = train_test_split(train,target_classifier,test_size=0.33,random_state=42)
Line = X_test['Line']
OppLine = X_test['Opp.Line']
#normalization
#norm = Normalizer()
#X_train = pd.DataFrame(norm.fit_transform(X_train),columns=X_train.columns)
#X_test = pd.DataFrame(norm.fit_transform(X_test),columns=X_test.columns)

#predict and store predictions
model_LR.fit(X_train, y_train)
lr_pred = model_LR.predict(X_test)
lr_proba = model_LR.predict_proba(X_test)


# In[4]:

#accuracy score
model_LR.score(X_test,y_test)


# In[5]:

#create bets file, with prediction in "Bet" variable, and outcome in "Net" variable
predictions = pd.DataFrame({'actual':y_test,'line':Line,'opp.line':OppLine})
predictions['outcomeline1'] = [i/100*1000 if i > 0 else abs(1000/i*100) for i in predictions['line']]
predictions['outcomeline2'] = [i/100*1000 if i > 0 else abs(1000/i*100) for i in predictions['opp.line']]
predictions['PredProba'] = lr_proba[:,1]
predictions['PredProba.opp'] = lr_proba[:,0]
predictions['EV_1'] = predictions['PredProba']*predictions['outcomeline1']
predictions['EV_2'] = predictions['PredProba.opp']*predictions['outcomeline2']
predictions['BestEV'] = predictions[['EV_1','EV_2']].max(axis=1)
predictions['makeBet'] = predictions['BestEV'] >1000
predictions['Bet'] = [1 if row['BestEV'] == row['EV_1'] else 0 for i,row in predictions.iterrows()]
predictions['Net'] = [row['BestEV'] if row['actual'] == row['Bet'] else -1000 for i,row in predictions.iterrows()]


# In[6]:

predictions.to_csv('bets.csv',index=False)


# ### Scraping NBA.com & ESPN.com for Input data

# In[7]:

def getData(url):
    
    #get data w/ selenium webdriver, update path as necessary
    browser = webdriver.Chrome(executable_path='C:/Users/me1035/chromedriver.exe')
    browser.get(url)
    table = browser.find_element_by_class_name('nba-stat-table__overflow').text.split('\n')
    stats = []
    teams = []
    
    #advanced stats has astRatio with a newline character, which causes column issues -- correction for that
    if url == "https://stats.nba.com/teams/advanced/?sort=W&dir=-1&Season=2018-19&SeasonType=Regular%20Season&LastNGames=10":
        table[0] = table[0] + table[1]
        table.pop(1)
        table.pop(1)
    
    #iterate through lines and create lists to create dataframe
    for line_id, lines in enumerate(table):
        if line_id == 0:
            column_names = lines.split(' ')[1:]
        else:
            if line_id % 3 == 2:
                teams.append(lines)
            if line_id % 3 == 0:
                stats.append([i for i in lines.split(' ')])
            
    return pd.DataFrame(stats,columns=column_names,index=[teams])


# In[8]:

def getESPNLines(url):
    browser = webdriver.Chrome(executable_path='C:/Users/me1035/chromedriver.exe')
    browser.get("http://www.espn.com/nba/lines/_/date")
    table = browser.find_elements_by_class_name('evenrow')
    new = []
    games = []
    for i in table:
        new.append(i.text.split('\n'))
    lines=[]
    for i in new[::3]:
        lines.append(i[-1].replace(" ",""))
        lines.append(i[-2].replace(" ",""))
    for i in new[::3]:
        games.append([i[-1],i[-2]])   
    
    return dict([item.split(":") for item in lines]),games


# In[13]:

#next step is to chain out to http://www.espn.com/nba/lines/_/date to get today's games + betting lines
#update URLS depending on number of games to avg for

trad_url = "https://stats.nba.com/teams/traditional/?sort=W_PCT&dir=-1&Season=2018-19&SeasonType=Regular%20Season&LastNGames=10"
adv_url = "https://stats.nba.com/teams/advanced/?sort=W&dir=-1&Season=2018-19&SeasonType=Regular%20Season&LastNGames=10"
espn = "http://www.espn.com/nba/lines/_/date"

traditional = getData(trad_url)
advanced = getData(adv_url)
lines,games = getESPNLines(espn)


# In[14]:

#drop overlapping columns from both trad/adv datasets, then merge
advanced.drop(['W','L','GP','MIN'],axis=1,inplace=True)
currentStats = advanced.merge(traditional,left_index=True,right_index=True)

#get updated stats from today's games, and add moneylines
currentStats.rename(inplace=True,index={"Atlanta Hawks":"ATL",
"Boston Celtics":"BOS",
"Brooklyn Nets":"BRK",
"Charlotte Hornets":"CHO",
"Chicago Bulls":"CHI",
"Cleveland Cavaliers":"CLE",
"Dallas Mavericks":"DAL",
"Denver Nuggets":"DEN",
"Detroit Pistons":"DET",
"Golden State Warriors":"GSW",
"Houston Rockets":"HOU",
"Indiana Pacers":"IND",
"LA Clippers":"LAC",
"Los Angeles Lakers":"LAL",
"Memphis Grizzlies":"MEM",
"Miami Heat":"MIA",
"Milwaukee Bucks":"MIL",
"Minnesota Timberwolves":"MIN",
"New Orleans Pelicans":"NOP",
"New York Knicks":"NYK",
"Oklahoma City Thunder":"OKC",
"Orlando Magic":"ORL",
"Philadelphia 76ers":"PHI",
"Phoenix Suns":"PHO",
"Portland Trail Blazers":"POR",
"Sacramento Kings":"SAC",
"San Antonio Spurs":"SAS",
"Toronto Raptors":"TOR",
"Utah Jazz":"UTA",
"Washington Wizards":"WAS"})

currentStats.reset_index(inplace=True)
currentStats.rename(columns={"level_0":"Team",
                            },inplace=True)

#add to this as new variations for teams come from ESPN.com
lines['BRK'] = lines.pop('BKN', 0)
lines['CHO'] = lines.pop('CHA', 0)
lines['NOP'] = lines.pop('NO', 0)
lines['NYK'] = lines.pop('NY', 0)
lines['SAS'] = lines.pop('SA', 0)
lines['UTA'] = lines.pop('UTAH', 0)

#add the money line column where team name matches
currentStats['Line'] = currentStats['Team'].map(lines)

#get the teams playing tonight
games = pd.DataFrame(games)
games = pd.DataFrame([games[0].apply(lambda x: x.split(":")[0]),games[1].apply(lambda x: x.split(":")[0])]).T
games.rename(columns={0:"Team",1:"Opponent"},inplace=True)

#adjust to 3-string team code
games["Team"] = games["Team"].replace({"BKN":"BRK",
              "NY":"NYK",
              "SA":"SAS",
              "UTAH":"UTA",
              "NO":"NOP",
                "CHA":"CHO"})
games["Opponent"] = games["Opponent"].replace({"BKN":"BRK",
              "NY":"NYK",
              "SA":"SAS",
              "UTAH":"UTA",
              "NO":"NOP",
                "CHA":"CHO"})
games['Date'] = today
games['B2B'] = 0
games['Opp.B2B'] = 0
games['Home'] = 1
games = games.merge(currentStats,left_on="Team",right_on="Team").merge(
    currentStats,left_on="Opponent",right_on="Team",suffixes=("",".Opp"))

games.rename(columns={"Line":"Line",
"Line.Opp":"Opp.Line",
"+/-":"mean_Diff",
"WIN%":"mean_Win",
"+/-.Opp":"mean_Diff_Opp",
"FGA":"mean_FGA",
"FGA.Opp":"mean_FGA_Opp",
"PTS":"mean_PTS",
"PTS.Opp":"mean_PTS_Opp",
"WIN%.Opp":"mean_Win_Opp",
"FTA":"mean_FTA",
"FTA.Opp":"mean_FTA_Opp",
"NETRTG":"NET_Rtng",
"NETRTG.Opp":"NET_Rtng_Opp",
"AST/TO":"AstTO",
"AST%":"Ast%",
"AST/TO.Opp":"AstTO_Opp",
"AST%.Opp":"Ast%_Opp"},inplace=True)

#this is the final dataframe to make predictions on today's data
games = pd.concat([games[['Team','Opponent','Date']],games[colsToTrain]],axis=1)


# In[51]:

def predLive(model,newdata):
    test = pd.read_csv("final_testset.csv")
    
    #append new data for predictions
    test = test.append(newdata)
    
    for index, row in test.iterrows():
        
        #exclude data if game already predicted
        if pd.Series(row['Bet']).isnull().all():            
            
            #predict with train columns from model training
            row_train = row[colsToTrain]
            live = model.predict_proba(row_train)
            
            
            #write probability predictions to row
            test.at[index,'Proba'] = live[:,1]
            test.at[index,'Proba.Opp'] = live[:,0]
            
        else:
            pass
            
    #add EV/Bet information
    test['Line'] = pd.to_numeric(test['Line'])
    test['Opp.Line'] = pd.to_numeric(test['Opp.Line'])
    test['outcomeline1'] = [i/100*1000 if i > 0 else abs(1000/i*100) for i in test['Line']]
    test['outcomeline2'] = [i/100*1000 if i > 0 else abs(1000/i*100) for i in test['Opp.Line']]
    test['EV_1'] = test['Proba']*test['outcomeline1']
    test['EV_2'] = test['Proba.Opp']*test['outcomeline2']
    test['BestEV'] = test[['EV_1','EV_2']].max(axis=1)
    test['makeBet'] = test['BestEV'] >1000
    test['Bet'] = [1 if row['BestEV'] == row['EV_1'] else 0 for i,row in test.iterrows()]
    
    test.drop(['outcomeline1','outcomeline2','EV_1','EV_2','makeBet'],axis=1,inplace=True)
    
    #overwrite testset w/ predicted probabilities + bet decisions
    test.to_csv("final_testset.csv",index=False)
    
    return test


# In[54]:

predLive(model_LR,games)


# In[ ]:



