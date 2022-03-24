import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path
import datetime
from datetime import timedelta
from lightgbm import LGBMRegressor, Booster
import gc
from copy import copy
from typing import Union
import lightgbm
from itertools import combinations
from typing import List
import time

import pickle
from functools import partial

import logging

from collections import Counter

import shap

shap.initjs()

LOGGER = logging.getLogger('tds')

model_name = 'lightgbm_1400'
target = 'stoloto'

WORK_DIR = Path(".")
STUDY_PATH = WORK_DIR.joinpath(
    f'total_dataset_study_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
)

LOG_PATH = Path(WORK_DIR / "experiment.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    level=logging.INFO,
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

horizon = 14

kvant = {7105: 10, 7103: 10, 7115: 5, 7101: 3, 7175: 3, 517: 10}

price_df = pd.DataFrame({'game_code': [7105, 7103, 7115, 7101, 7175],
                         'price': [100, 100, 75, 50, 75]})
price_df = price_df.set_index('game_code')

utilization_coefficients = {7105: 30,
                            7103: 35,
                            7115: 50,
                            7101: 50,
                            7175: 50,
                            517: 30}
utilization_coefficients = {int(game): 100 / (100 - int(utilization_coefficients[game])) for game in
                            list(utilization_coefficients.keys())}


def mse(real, forecast):
    real_array = np.array(real)
    forecast_array = np.array(forecast)
    return np.mean(np.power((real_array - forecast_array), 2))


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def score(test, predict, active_ops=True, compare_with_real_sales=True):
    # return pandas df
    df = copy(test)
    df['predict'] = predict
    df.predict = df.predict.astype(int)
    if compare_with_real_sales:
        s = sales[sales.game_code.isin(list(test.game_code.unique()))]
        s = s[s.ds.isin(list(test.ds.unique()))]
        if 'value' in df.columns:
            df = df.drop(['value'], 1)
        df = df.merge(s[['game_code', 'ds', 'value', 'ops_id']], on=['game_code', 'ds', 'ops_id'], how='left')  # outer
        df.value = df.value.fillna(0)
        df.predict = df.predict.fillna(0)
        # df = df.merge(sales)
    if active_ops:
        f = prod[prod.ds.isin(list(test.ds.values))]
        f = f[f.game_code.isin(list(test.game_code.values))]
        df = df.merge(f[['ops_id', 'ds', 'base_forecast', 'game_code']], on=['ops_id', 'ds', 'game_code'], how='outer')

    df['value'] = df.value.fillna(0)

    # business processing (add utilization and round to kvant)
    df['distributing'] = df.apply(
        lambda x: max(np.ceil(
            x.predict * utilization_coefficients[x.game_code] / kvant[x.game_code]
        ), 1) * kvant[x.game_code]
        , 1)
    df['plan_transfer'] = df.apply(
        lambda x: max(np.ceil(
            x.value * utilization_coefficients[x.game_code] / kvant[x.game_code]
        ), 1) * kvant[x.game_code]
        , 1)

    if active_ops:
        df['distributing'].loc[df.base_forecast.isna()] = 0
        df['distributing'].loc[df.predict.isna()] = df.loc[df.predict.isna()].game_code.map(kvant)
        df['predict'].loc[df.base_forecast.isna()] = 0
        df['predict'].loc[df.predict.isna()] = df.loc[df.predict.isna()].game_code.map(kvant)

    score_result = pd.concat(
        [
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.value)], columns=['sales'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.predict)], columns=['origin_predict'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.distributing)], columns=['predict'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([len(x.predict)], columns=['ops_count'])
            ),
            df[df.value < df.predict].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.predict - x.value)], columns=['origin_over_sales'])
            ),
            df[df.value > df.predict].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.value - x.predict)], columns=['origin_lost_sales'])
            ),
            df[df.value < df.distributing].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.distributing - x.value)], columns=['over_sales'])
            ),
            df[df.plan_transfer < df.distributing].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.distributing - x.plan_transfer)], columns=['over_plan_sales'])
            ),
            df[df.value > df.distributing].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.value - x.distributing)], columns=['lost_sales'])
            ),
            df[df.value > df.distributing].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([len(x.value)], columns=['lost_sales_count'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame(
                    [
                        sum(
                            x[x.value > x.distributing].value - x[x.value > x.distributing].distributing
                        ) / sum(x.value) * 100
                    ], columns=['lost_percent'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([100 - sum(x.value) / sum(x.distributing) * 100], columns=['util_coef'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame(
                    [
                        sum(
                            x[x.distributing > x.plan_transfer].distributing - x[
                                x.distributing > x.plan_transfer].plan_transfer
                        ) / sum(x.value) * 100
                    ], columns=['util_over_plan_percent'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([mse(x.value, x.distributing)], columns=['mse'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([mse(x.value, x.predict)], columns=['origin_mse'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([smape(x.value, x.predict)], columns=['origin_smape'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([smape(x.value, x.distributing)], columns=['smape'])
            ),
            df.groupby(['game_code']).apply(
                lambda x: pd.DataFrame([smape(x.plan_transfer, x.distributing)], columns=['plan_smape'])
            ),

        ], 1
    )

    # score_result = score_result.set_index('game_code')
    score_result = score_result.join(price_df, on='game_code', how='left')
    score_result['over_plan_losses'] = score_result['lost_sales'] * score_result['price'] + score_result[
        'over_plan_sales'] * 5
    score_result['losses'] = score_result['lost_sales'] * score_result['price'] + score_result['over_sales'] * 5

    return score_result


def postprocessing(test: pd.DataFrame, utilization_coefficients: dict, kvants: dict) -> pd.DataFrame:
    """
    Function for added utilization and rounded to kvant
    :param test:  pd.DataFrame with predictions
    :param utilization_coefficients: dict, with coefficients
    :param kvants: dict with games and kvants
    :return df: pd.DataFrame with kvanted predictions
    """
    df = copy(test)
    df.base_forecast = df.base_forecast.astype(int)
    df['forecast'] = df['base_forecast']
    df['forecast'] = df.apply(
        lambda x: max(np.ceil(
            x.base_forecast * utilization_coefficients[x.game_code] / kvants[x.game_code]
        ), 1) * kvants[x.game_code]
        , 1)

    return df


train0 = pd.read_parquet('./splits/train_full_0.parquet')
test0 = pd.read_parquet('./splits/test_full_0.parquet')
train1 = pd.read_parquet('./splits/train_full_1.parquet')
test1 = pd.read_parquet('./splits/test_full_1.parquet')
train2 = pd.read_parquet('./splits/train_full_2.parquet')
test2 = pd.read_parquet('./splits/test_full_2.parquet')
train3 = pd.read_parquet('./splits/train_full_3.parquet')
test3 = pd.read_parquet('./splits/test_full_3.parquet')
train4 = pd.read_parquet('./splits/train_full_4.parquet')
test4 = pd.read_parquet('./splits/test_full_4.parquet')
dss = [(train0, test0), (train1, test1), (train2, test2), (train3, test3), (train4, test4)]

args = {'colsample_bynode': 0.6,
        'colsample_bytree': 0.5,
        'lambda_l1': 6.4,
        'learning_rate': 0.06878228501024089,
        'max_bin': 163,
        'max_depth': 7,
        'min_child_samples': 13,
        'n_estimators': 1400,
        'num_leaves': 87,
        'subsample': 0.8,
        'subsample_freq': 14}
seed = 424242
rgr = LGBMRegressor(
    n_jobs=28,
    silent=True,
    random_state=seed,
    device_type="gpu",
    **args
)


def ret_shap_values(X, rgr):
    explainer = shap.Explainer(rgr)
    shap_values = explainer(X)
    return shap_values


def importance(X_valid, rgr):
    shap_values = ret_shap_values(X_valid, rgr)

    shap_df = pd.DataFrame(shap_values.values, columns=X_valid.columns).mean()

    importance_df = pd.DataFrame([X_valid.columns.tolist(), shap_df.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']

    importance_df['shap_importance'] = importance_df.shap_importance.abs()
    importance_df = importance_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)

    return importance_df


from clearml import Task

task = Task.init(
    project_name="stoloto",
    task_name="select_features_no_id",
    tags=['select_features']
)
logger = task.get_logger()

read_data = []
for i, p in enumerate(
        Path("./sales.parquet").iterdir()
):
    if "parquet" in p.name:
        df = pd.read_parquet(p)
        read_data.append(df)
sales = pd.concat(read_data)
sales.ds = pd.to_datetime(sales.ds)
sales.game_code = sales.game_code.astype(int)
sales.value = sales.value.astype(int)
sales.shape

dates = ["2020-01-12", '2020-01-26', '2020-03-01', '2020-03-08', '2020-04-05']
col_tr = list(set(dss[0][0].columns) - set(['ds', 'value', 'game_code', 'ops_id']))
# col_tr.append('randNumCol')
dcol = ['-']

l = len(col_tr)
print(f'initial column lenth: {l}')

import time

drop_columns = ['ds', 'game_code', 'ops_id', 'value']

for i in range(1, 70):
    start = time.strftime("%Y-%m-%d_%H:%M:%S")
    print(f'{start}---{i} iteration---')
    smape_list = []
    importance_all = pd.DataFrame(index=col_tr)
    all_scores = []

    logger.report_text(f"features: {col_tr}", iteration=i)
    logger.report_text(f"del_features: {dcol}", iteration=i)

    for j, ds in enumerate(dss):
        print(f'{j} date: {dates[j]}')
        X_train = ds[0].drop(drop_columns, 1)
        # X_train['randNumCol'] = np.random.randint(1, 10000, X_train.shape[0])
        # X_train = X_train.loc[:,~X_train.columns.duplicated()]
        Y_train = ds[0]['value']
        drop_test = ds[1][drop_columns].copy()
        X_valid = ds[1].drop(drop_columns, 1)
        # X_valid['randNumCol'] = np.random.randint(1, 10000, X_valid.shape[0])
        # X_valid = X_valid.loc[:,~X_valid.columns.duplicated()]
        y_valid = ds[1]['value']

        X_train = X_train[col_tr]
        X_valid = X_valid[col_tr]

        rgr.fit(X_train, Y_train)
        y_predict = rgr.predict(X_valid)

        importance_df = importance(X_valid, rgr)
        importance_df = importance_df.set_index('column_name')
        importance_all = importance_all.merge(importance_df, left_index=True, right_index=True)

        X_valid['base_forecast'] = y_predict.astype('int')
        X_valid['base_forecast'] = X_valid['base_forecast'].fillna(1)
        X_valid[['ds', 'game_code', 'ops_id', 'value']] = drop_test[['ds', 'game_code', 'ops_id', 'value']]
        score_table = score(X_valid.drop(['base_forecast'], 1), X_valid['base_forecast'], active_ops=False,
                            compare_with_real_sales=True)
        # score_table.to_csv(f"./scores/score_{l} features_{dates[j]}.csv")
        # logger.report_table("scores", f"{l}_features_{dates[j]}", table_plot=score_table, iteration=i)

        all_scores.append(score_table.assign(test_date=dates[j]))

    scores = pd.concat(all_scores)
    scores.to_csv(f"./scores_select_features/scores_{l}_features.csv")
    losses = scores['losses'].sum()

    lost_sales = scores['lost_sales'].sum()
    over_sales = scores['over_plan_sales'].sum()

    # smape_mean = np.mean(smape_list)

    logger.report_scalar("count_features", "base", value=l, iteration=i)
    logger.report_scalar("losses", "base", value=losses, iteration=i)
    logger.report_scalar("lost_sales", "base", value=lost_sales, iteration=i)
    logger.report_scalar("over_plan_sales", "base", value=over_sales, iteration=i)
    # logger.report_scalar("smape", "base", value=smape_mean, iteration=i)

    importance_it = importance_all.mean(axis=1)
    importance_it = importance_it.sort_values(ascending=False)
    logger.report_table("features importance", f"{l} features",
                        table_plot=pd.DataFrame([importance_it]).T.rename(columns={0: 'shap_importance'}), iteration=i)

    if 'randNumCol' in list(importance_it[:5].index.values):
        rand_imp = importance_it['randNumCol']
        logger.report_text(f"Random feature got {rand_imp} importance", iteration=i)
        break

    min_shap_imp = importance_it.drop('randNumCol')[-3]
    min_shap_imp = min_shap_imp if min_shap_imp < 0.1 else importance_it.drop('randNumCol')[-1]
    min_shap_imp = max(min_shap_imp, 0.001)
    print('min_shap_imp:', min_shap_imp)
    dcol = list(importance_it[importance_it <= min_shap_imp].index.values)
    if 'randNumCol' in dcol:
        dcol.remove('randNumCol')
    col = list(X_train.columns)

    col_tr = list(set(col) - set(dcol))
    l = len(col_tr)
    logger.report_text(f"Next del_features: {dcol}", iteration=i)
    logger.report_text(f"Next train features: {col_tr}", iteration=i)

    # logger.report_text(f"params: {params}", iteration=i)

task.mark_completed()