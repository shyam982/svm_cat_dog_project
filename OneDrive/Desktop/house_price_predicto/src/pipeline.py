
import numpy as np
import pandas as pd
import warnings
import os
import json
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_dataset(n=2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    sqft        = rng.integers(500, 5000, n)
    bedrooms    = rng.integers(1, 6, n)
    bathrooms   = rng.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n)
    garage_cars = rng.integers(0, 4, n)
    year_built  = rng.integers(1900, 2023, n)
    overall_qual= rng.integers(1, 11, n)
    lot_area    = rng.integers(1500, 20000, n)
    total_bsmt  = rng.integers(0, 3000, n)
    neighborhood= rng.choice(['NoRidge','NridgHt','StoneBr','Timber','Veenker',
                               'Somerst','ClearCr','Crawfor','CollgCr','Blmngtn',
                               'Gilbert','NWAmes','SawyerW','Mitchel','NAmes'], n)
    bldg_type   = rng.choice(['1Fam','2FmCon','Duplx','TwnhsE','TwnhsI'], n)
    sale_cond   = rng.choice(['Normal','Abnorml','Partial','AdjLand','Alloca','Family'], n)

    age         = 2024 - year_built
    neigh_premium = {
        'NoRidge':1.35,'NridgHt':1.30,'StoneBr':1.25,'Timber':1.10,
        'Veenker':1.08,'Somerst':1.05,'ClearCr':1.03,'Crawfor':1.02,
        'CollgCr':1.00,'Blmngtn':0.98,'Gilbert':0.97,'NWAmes':0.95,
        'SawyerW':0.93,'Mitchel':0.90,'NAmes':0.88
    }
    np_arr = np.array([neigh_premium[n_] for n_ in neighborhood])

    price = (
        50_000
        + sqft        * 85
        + bedrooms    * 8_000
        + bathrooms   * 12_000
        + garage_cars * 15_000
        + overall_qual* 10_000
        + lot_area    * 3
        + total_bsmt  * 25
        - age         * 300
    ) * np_arr + rng.normal(0, 15_000, n)

    price = np.clip(price, 50_000, 800_000)

    df = pd.DataFrame({
        'Id'           : range(1, n+1),
        'GrLivArea'    : sqft,
        'BedroomAbvGr' : bedrooms,
        'FullBath'     : bathrooms.astype(int),
        'HalfBath'     : rng.integers(0, 2, n),
        'BsmtFullBath' : rng.integers(0, 2, n),
        'GarageArea'   : garage_cars * 200 + rng.integers(0, 100, n),
        'GarageCars'   : garage_cars,
        'YearBuilt'    : year_built,
        'YearRemodAdd' : year_built + rng.integers(0, 30, n),
        'OverallQual'  : overall_qual,
        'OverallCond'  : rng.integers(1, 10, n),
        'LotArea'      : lot_area,
        'TotalBsmtSF'  : total_bsmt,
        'Neighborhood' : neighborhood,
        'BldgType'     : bldg_type,
        'SaleCondition': sale_cond,
        'SalePrice'    : price.astype(int),
    })

    for col in ['GarageArea', 'TotalBsmtSF', 'BsmtFullBath']:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['TotalBath']         = df['FullBath'] + 0.5*df['HalfBath'] + df.get('BsmtFullBath', 0)
    df['HouseAge']          = 2024 - df['YearBuilt']
    df['YearsSinceRemodel'] = 2024 - df['YearRemodAdd']
    df['TotalSF']           = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
    df['QualArea']          = df['OverallQual'] * df['GrLivArea']
    df['PricePerRoom']      = df['GrLivArea'] / (df['BedroomAbvGr'] + 1)
    df['IsNew']             = (df['HouseAge'] <= 5).astype(int)
    df['IsRemodeled']       = (df['YearsSinceRemodel'] <= 10).astype(int)

    le = LabelEncoder()
    for col in ['Neighborhood', 'BldgType', 'SaleCondition']:
        if col in df.columns:
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    df.drop(columns=['Neighborhood','BldgType','SaleCondition'], inplace=True, errors='ignore')
    df.drop(columns=['Id'], inplace=True, errors='ignore')

    return df


def build_preprocessor():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  RobustScaler()),
    ])


MODELS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression' : Ridge(alpha=10.0),
    'Lasso Regression' : Lasso(alpha=100.0),
    'ElasticNet'       : ElasticNet(alpha=100.0, l1_ratio=0.5),
    'XGBoost'          : xgb.XGBRegressor(
                            n_estimators=300, learning_rate=0.05,
                            max_depth=5, subsample=0.8,
                            colsample_bytree=0.8, random_state=42,
                            verbosity=0),
    'LightGBM'         : lgb.LGBMRegressor(
                            n_estimators=300, learning_rate=0.05,
                            max_depth=5, subsample=0.8,
                            colsample_bytree=0.8, random_state=42,
                            verbose=-1),
}


def evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    mae   = mean_absolute_error(y_test, y_pred)
    r2    = r2_score(y_test, y_pred)
    rmsle = np.sqrt(mean_squared_error(
        np.log1p(np.maximum(y_test, 0)),
        np.log1p(np.maximum(y_pred, 0))
    ))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train,
                                scoring='neg_root_mean_squared_error', cv=cv)

    return {
        'model'  : name,
        'RMSE'   : round(rmse, 2),
        'MAE'    : round(mae, 2),
        'R2'     : round(r2, 4),
        'RMSLE'  : round(rmsle, 4),
        'CV_RMSE': round(-cv_scores.mean(), 2),
        'y_pred' : y_pred,
    }


def run_pipeline():
    print("=" * 65)
    print("  ADVANCED HOUSE PRICE PREDICTION — Task-01 | Prodigy Infotech")
    print("=" * 65)

    print("\n[1/5] Generating dataset ...")
    df = generate_dataset(2000)
    df.to_csv(os.path.join(DATA_DIR, 'house_data.csv'), index=False)
    print(f"      Shape  : {df.shape}")
    print(f"      Prices : ${df['SalePrice'].min():,.0f} – ${df['SalePrice'].max():,.0f}")
    print(f"      Missing: {df.isnull().sum().sum()} cells")

    print("\n[2/5] Engineering features ...")
    df_fe = engineer_features(df)
    feature_cols = [c for c in df_fe.columns if c != 'SalePrice']
    print(f"      Total features: {len(feature_cols)}")

    X = df_fe[feature_cols].values
    y = df_fe['SalePrice'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("\n[3/5] Preprocessing ...")
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))

    print("\n[4/5] Training models ...\n")
    results    = []
    best_model = None
    best_r2    = -np.inf

    for name, model in MODELS.items():
        res = evaluate(model, X_train_p, X_test_p, y_train, y_test, name)
        results.append(res)
        print(f"  {name:<22}  R²={res['R2']:.4f}  "
              f"RMSE=${res['RMSE']:>10,.0f}  RMSLE={res['RMSLE']:.4f}")
        if res['R2'] > best_r2:
            best_r2    = res['R2']
            best_model = (name, model)

    print(f"\n  Best model : {best_model[0]}  (R² = {best_r2:.4f})")
    joblib.dump(best_model[1], os.path.join(MODEL_DIR, 'best_model.pkl'))

    print("\n[5/5] Saving report ...")
    report = {
        'generated_at' : datetime.now().isoformat(),
        'dataset_rows' : int(len(df)),
        'feature_count': int(len(feature_cols)),
        'best_model'   : best_model[0],
        'best_r2'      : float(best_r2),
        'feature_names': feature_cols,
        'results'      : [{k:v for k,v in r.items() if k != 'y_pred'} for r in results],
    }
    with open(os.path.join(REPORT_DIR, 'results.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("  Pipeline complete!")
    print("=" * 65)
    return results, feature_cols, df_fe


if __name__ == '__main__':
    run_pipeline()