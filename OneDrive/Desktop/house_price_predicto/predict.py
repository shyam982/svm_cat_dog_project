"""
predict.py — Predict price for any house
Run: python predict.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


def predict_price(
    sqft=1800, bedrooms=3, full_baths=2, half_baths=1,
    garage_cars=2, year_built=1995, overall_qual=7,
    lot_area=8000, total_bsmt=900.0,
    neighborhood='CollgCr', bldg_type='1Fam', sale_cond='Normal'
):
    model        = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))

    neigh_map = {'NoRidge':0,'NridgHt':1,'StoneBr':2,'Timber':3,'Veenker':4,
                 'Somerst':5,'ClearCr':6,'Crawfor':7,'CollgCr':8,'Blmngtn':9,
                 'Gilbert':10,'NWAmes':11,'SawyerW':12,'Mitchel':13,'NAmes':14}
    bldg_map  = {'1Fam':0,'2FmCon':1,'Duplx':2,'TwnhsE':3,'TwnhsI':4}
    sale_map  = {'Normal':0,'Abnorml':1,'Partial':2,'AdjLand':3,'Alloca':4,'Family':5}

    age          = 2024 - year_built
    total_sf     = sqft + (total_bsmt or 0)
    total_bath   = full_baths + 0.5 * half_baths
    qual_area    = overall_qual * sqft
    price_per_rm = sqft / (bedrooms + 1)
    garage_area  = garage_cars * 200

    row = np.array([[
        sqft, bedrooms, full_baths, half_baths, 0,
        garage_area, garage_cars, year_built, year_built,
        overall_qual, 5, lot_area, total_bsmt or 0,
        neigh_map.get(neighborhood, 8),
        bldg_map.get(bldg_type, 0),
        sale_map.get(sale_cond, 0),
        total_bath, age, 0, total_sf, qual_area, price_per_rm,
        int(age <= 5), int(age <= 15),
    ]])

    price = model.predict(preprocessor.transform(row))[0]

    return {
        'predicted_price': f"${price:,.0f}",
        'low_estimate'   : f"${max(price*0.90,0):,.0f}",
        'high_estimate'  : f"${price*1.10:,.0f}",
    }


if __name__ == '__main__':
    print("\n🏠  House Price Predictor\n" + "─"*40)

    test_houses = [
        dict(sqft=1500, bedrooms=3, full_baths=2, year_built=1990, overall_qual=6),
        dict(sqft=2800, bedrooms=4, full_baths=3, year_built=2005,
             overall_qual=8, neighborhood='NridgHt'),
        dict(sqft=4500, bedrooms=5, full_baths=4, year_built=2018,
             overall_qual=10, neighborhood='NoRidge'),
    ]

    for i, house in enumerate(test_houses, 1):
        r = predict_price(**house)
        print(f"\nHouse {i}: {house}")
        print(f"  Estimate : {r['predicted_price']}")
        print(f"  Range    : {r['low_estimate']} – {r['high_estimate']}")