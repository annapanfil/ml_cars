
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import eli5

def parse_price(val):
    return float(val.replace(" ", "").replace(",", "."))

def get_df(df_train, df_test):
    df_train = df_train[ df_train.index != 106447 ].reset_index(drop=True)
    df = pd.concat([df_train, df_test])
 
    params = df["offer_params"].apply(pd.Series)
    params = params.fillna(-1)

    df = pd.concat([df, params], axis=1)
    print(df.shape)

    obj_feats = params.select_dtypes(object).columns

    for feat in obj_feats:
        df["{}_cat".format(feat)] = df[feat].factorize()[0]
            
    return df

def check_model(df, feats, model, cv=5, scoring="neg_mean_absolute_error", show_eli5=True):
    df_train = df[ ~df["price_value"].isnull() ].copy()
    df_test = df[ df["price_value"].isnull() ].copy()

    X_train = df_train[feats]
    y_train = df_train["price_value"]
    
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    result = np.mean(scores), np.std(scores)
    
    if show_eli5:
        model.fit(X_train, y_train)
        print(result)
        return eli5.show_weights(model, feature_names=feats)
    
    return result

def check_log_model(df, feats, model, cv=5, scoring=mean_absolute_error, show_eli5=True):
    df_train = df[ ~df["price_value"].isnull() ].copy()

    X = df_train[feats]
    y = df_train["price_value"]
    y_log = np.log(y)
    
    cv = KFold(n_splits=cv, shuffle=True, random_state=0)
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_log_train, y_test = y_log.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_log_train)
        y_log_pred = model.predict(X_test)
        y_pred = np.exp(y_log_pred)

        score = scoring(y_test, y_pred)
        scores.append(score)
        
    result = np.mean(scores), np.std(scores)
    
    if show_eli5:
        model.fit(X, y_log)
        print(result)
        return eli5.show_weights(model, feature_names=feats)

    return result
