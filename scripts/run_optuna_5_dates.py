import pandas as pd
import optuna
import numpy as np
from pathlib import Path
import datetime
import lightgbm
import pickle
from functools import partial
import logging
import argparse
from clearml import Task

WORK_DIR = Path(".")
STUDY_PATH = WORK_DIR.joinpath(
    f'total_dataset_study_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
)

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


def score(test, predict, active_ops=False, compare_with_real_sales=True):
    # return pandas df
    df = test.copy()
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
            df[df.value < df.distributing].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.distributing - x.value)], columns=['over_sales'])
            ),
            df[df.plan_transfer < df.distributing].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.distributing - x.plan_transfer)], columns=['over_plan_sales'])
            ),
            df[df.value > df.distributing].groupby(['game_code']).apply(
                lambda x: pd.DataFrame([sum(x.value - x.distributing)], columns=['lost_sales'])
            )
        ], 1
    )

    # score_result = score_result.set_index('game_code')
    score_result = score_result.join(price_df, on='game_code', how='left')
    score_result['over_plan_losses'] = score_result['lost_sales'] * score_result['price'] + score_result[
        'over_plan_sales'] * 5
    score_result['lost_sales_losses'] = score_result['lost_sales'] * score_result['price']
    score_result['losses'] = score_result['lost_sales'] * score_result['price'] + score_result['over_sales'] * 5

    return score_result


def train_model(args, X, Y, params):
    """Train LightGBM model"""
    train_params = {key: value for key, value in params.items() if key != "max_bin"}

    if args and args.use_gpu:
        train_params["device"] = "gpu"
        train_params["gpu_device_id"] = 2
        train_params["gpu_platform_id"] = 1

    train_params["num_threads"] = 10
    dataset = lightgbm.Dataset(X, Y, params={"max_bin": params["max_bin"]})
    model = lightgbm.train(train_params, dataset)
    return model


def scoring(
        trial: object,
        dss,
        args
) -> float:
    # Objective function for binary classification.

    # Calculates the average precision in holdout
    # for the model with picked parameters.

    # Args:
    #     trial (object): a process of evaluating an objective funtion
    #     x_train (pd.DataFrame, optional): features for training. Defaults to None.
    #     y_train (pd.DataFrame, optional): labels for training. Defaults to None.
    #     x_val (pd.DataFrame, optional): features for validation. Defaults to None.
    #     y_val (pd.DataFrame, optional): labels for validation. Defaults to None.

    # Returns:
    #     float: average precision on test data.

    trial_params = {
        "seed": 424242,
        "verbosity": -1,
        "num_gpu": 2,
        "n_estimators": trial.suggest_int("n_estimators", 100, 3500, step=100),  #
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "max_bin": trial.suggest_int("max_bin", 63, 255, step=10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "num_leaves": trial.suggest_int("num_leaves", 7, 100, step=10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 1.0, step=0.1),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 10, step=0.1),  #
        "max_delta_step": trial.suggest_float("max_delta_step", 0, 10, step=0.1),  #
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 50, step=1),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 1000, step=10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
        "cegb_penalty_split": trial.suggest_float("cegb_penalty_split", 0.0, 3.0, step=0.1),
        # 'extra_trees': trial.suggest_categorical('extra_trees', [False, True]),
    }

    dates = ["2020-01-12", '2020-01-26', '2020-03-01', '2020-03-08', '2020-04-05']
    drop_columns = ['ds', 'game_code', 'ops_id', 'value']
    scores = []
    lossses_list = []
    for j, ds in enumerate(dss):
        X_train = ds[0].drop(drop_columns, 1)
        Y_train = ds[0]['value']
        drop_test = ds[1][drop_columns].copy()
        X_valid = ds[1].drop(drop_columns, 1)
        y_valid = ds[1]['value']

        # rgr.fit(X_train, Y_train)
        model = train_model(args, X_train, Y_train, trial_params)
        y_predict = model.predict(X_valid)

        X_valid['base_forecast'] = y_predict.astype('int')
        X_valid['base_forecast'] = X_valid['base_forecast'].fillna(0)
        X_valid[['ds', 'game_code', 'ops_id', 'value']] = drop_test[['ds', 'game_code', 'ops_id', 'value']]
        score_table = score(X_valid.drop(['base_forecast'], 1), X_valid['base_forecast'], active_ops=False,
                            compare_with_real_sales=True)
        lossses = score_table["losses"].sum()
        lossses_list.append(lossses)
        scores.append(score_table.assign(test_date=dates[j]))

    # lossses_sum = np.sum(lossses_list)
    all_scores = pd.concat(scores)
    i = trial.number
    all_scores.to_csv(f"./scores_optuna_2/optuna_scores_{i}.csv")

    lossses_sum = all_scores['losses'].sum()
    lost_sales = all_scores['lost_sales'].sum()
    over_sales = all_scores['over_sales'].sum() * 5
    lost_sales_losses = all_scores['lost_sales_losses'].sum()

    logger.report_text(f"itaration: {i}", iteration=i)
    logger.report_text(f"params: {trial_params}", iteration=i)
    logger.report_scalar("losses", "base", value=lossses_sum, iteration=i)
    logger.report_scalar("lost_sales_losses", "base", value=lost_sales_losses, iteration=i)
    logger.report_scalar("over_sales", "base", value=over_sales, iteration=i)
    logger.report_table("scores", f"scores_{i}", table_plot=all_scores, iteration=i)

    return lossses_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, dest="study_name")
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--ntrials", type=int, dest="n_trials")
    args = parser.parse_args()

    LOG_PATH = Path(WORK_DIR / "tuneparams.log")
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="a",
        level=logging.INFO,
    )
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler())

    log.info("Loading data")

    train0 = pd.read_parquet('./optuna_splits/train_14features_0.parquet')
    test0 = pd.read_parquet('./optuna_splits/test_14features_0.parquet')
    train1 = pd.read_parquet('./optuna_splits/train_14features_1.parquet')
    test1 = pd.read_parquet('./optuna_splits/test_14features_1.parquet')
    train2 = pd.read_parquet('./optuna_splits/train_14features_2.parquet')
    test2 = pd.read_parquet('./optuna_splits/test_14features_2.parquet')
    train3 = pd.read_parquet('./optuna_splits/train_14features_3.parquet')
    test3 = pd.read_parquet('./optuna_splits/test_14features_3.parquet')
    train4 = pd.read_parquet('./optuna_splits/train_14features_4.parquet')
    test4 = pd.read_parquet('./optuna_splits/test_14features_4.parquet')
    dss = [(train0, test0), (train1, test1), (train2, test2), (train3, test3), (train4, test4)]

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

    # Run optuna study
    log.info("Starting optuna study")
    log.info(
        f'Starting optuna study_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    )

    # init ClearML
    model_name = 'lightgbm'
    target = 'stoloto'
    task = Task.init(
        project_name="stoloto",
        task_name="optuna_14f",
        tags=['opt_params']
    )
    logger = task.get_logger()

    study_name = (
        args.study_name if args.study_name else "stoloto-optuna-study"
    )  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    total_study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    first_trial_params = {'colsample_bynode': 0.6,
                          'colsample_bytree': 0.5,
                          'lambda_l1': 6.4,
                          'learning_rate': 0.06878228501024089,
                          'max_bin': 163,
                          'max_depth': 7,
                          'min_child_samples': 13,
                          'n_estimators': 1400,
                          'num_leaves': 87,
                          'subsample': 0.8,
                          'subsample_freq': 14,
                          'max_delta_step': 0.0,
                          'cegb_penalty_split': 0.0
                          }

    total_study.enqueue_trial(params=first_trial_params)

    scoring = partial(
        scoring,
        dss=dss,
        args=args,
    )
    try:
        total_study.optimize(scoring, n_trials=args.n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        pass
        with open(STUDY_PATH, "wb") as f:
            pickle.dump(total_study, f)
        log.info("Save study at studypath")

    log.info("Best LightGBM parameters")
    log.info(total_study.best_params)

    log.info(
        f'Save study results_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    )
    with open(STUDY_PATH, "wb") as fs:
        pickle.dump(total_study, fs)

    task.mark_completed()
