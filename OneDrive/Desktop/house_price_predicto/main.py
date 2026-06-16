"""
main.py — Run training + all visualizations
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline  import run_pipeline, engineer_features, MODELS
from src.visualize import (plot_eda, plot_model_comparison,
                            plot_predictions, plot_feature_importance)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def main():
    results, feature_names, df_fe = run_pipeline()

    print("\n[+] Generating visualizations ...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'house_data.csv'))
    plot_eda(df)

    _, y_test = train_test_split(df_fe['SalePrice'].values,
                                 test_size=0.2, random_state=42)
    plot_model_comparison([{k:v for k,v in r.items() if k != 'y_pred'} for r in results])
    plot_predictions(results, y_test)

    for r in results:
        if r['model'] in ('XGBoost', 'LightGBM'):
            plot_feature_importance(MODELS[r['model']], feature_names,
                                    f"Feature Importance — {r['model']}")
            break

    print("\n  All done! Check the reports/ folder.")


if __name__ == '__main__':
    main()