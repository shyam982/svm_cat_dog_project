"""
Visualization module — EDA + model comparison charts
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')

PALETTE = ['#6C63FF','#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7']
plt.rcParams.update({
    'figure.facecolor' : '#0F0F1A',
    'axes.facecolor'   : '#1A1A2E',
    'axes.edgecolor'   : '#2D2D4E',
    'text.color'       : '#E0E0F0',
    'axes.labelcolor'  : '#E0E0F0',
    'xtick.color'      : '#A0A0C0',
    'ytick.color'      : '#A0A0C0',
    'grid.color'       : '#2D2D4E',
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.5,
})


def plot_eda(df: pd.DataFrame):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('House Price Prediction — Exploratory Data Analysis',
                 fontsize=22, fontweight='bold', color='#E0E0F0', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Price distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['SalePrice']/1000, bins=40, color=PALETTE[0], edgecolor='none', alpha=0.85)
    ax1.axvline(df['SalePrice'].median()/1000, color=PALETTE[1], lw=2, ls='--', label='Median')
    ax1.set_title('Sale Price Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Price ($k)')
    ax1.set_ylabel('Count')
    ax1.legend(framealpha=0.3)

    # 2. Living Area vs Price
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(df['GrLivArea'], df['SalePrice']/1000,
                     c=df['OverallQual'], cmap='plasma', alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax2).set_label('Overall Quality', color='#A0A0C0')
    ax2.set_title('Living Area vs Price\n(colour = quality)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('GrLivArea (sqft)')
    ax2.set_ylabel('Price ($k)')

    # 3. Bedrooms vs median price
    ax3 = fig.add_subplot(gs[0, 2])
    bd_med = df.groupby('BedroomAbvGr')['SalePrice'].median() / 1000
    bars = ax3.bar(bd_med.index, bd_med.values, color=PALETTE[2], alpha=0.85)
    ax3.set_title('Bedrooms vs Median Price', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Bedrooms')
    ax3.set_ylabel('Median Price ($k)')
    for bar in bars:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 f'${bar.get_height():.0f}k', ha='center', fontsize=9, color='#E0E0F0')

    # 4. Quality vs Price boxplot
    ax4 = fig.add_subplot(gs[1, 0])
    qual_groups = [df[df['OverallQual']==q]['SalePrice'].values/1000
                   for q in sorted(df['OverallQual'].unique())]
    bp = ax4.boxplot(qual_groups, patch_artist=True,
                     medianprops={'color':'#FF6B6B','lw':2})
    for patch, color in zip(bp['boxes'],
                             plt.cm.plasma(np.linspace(0.1, 0.9, len(qual_groups)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax4.set_xticklabels(sorted(df['OverallQual'].unique()))
    ax4.set_title('Quality Grade vs Price', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Overall Quality (1–10)')
    ax4.set_ylabel('Price ($k)')

    # 5. Bathrooms vs Price
    ax5 = fig.add_subplot(gs[1, 1])
    bt_med = df.groupby('FullBath')['SalePrice'].median() / 1000
    ax5.bar(bt_med.index, bt_med.values, color=PALETTE[3], alpha=0.85)
    ax5.set_title('Full Baths vs Median Price', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Full Bathrooms')
    ax5.set_ylabel('Median Price ($k)')

    # 6. Correlation heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    num_df = df.select_dtypes(include='number')
    corr   = num_df.corr()['SalePrice'].drop('SalePrice').abs().nlargest(8)
    corr_vals = num_df[corr.index.tolist() + ['SalePrice']].corr()
    mask  = np.triu(np.ones_like(corr_vals, dtype=bool))
    sns.heatmap(corr_vals, mask=mask,
                cmap=sns.diverging_palette(240, 10, as_cmap=True),
                center=0, ax=ax6, annot=True, fmt='.2f',
                annot_kws={'size':7}, linewidths=0.5, linecolor='#2D2D4E',
                cbar_kws={'shrink':0.7})
    ax6.set_title('Feature Correlation', fontsize=13, fontweight='bold')
    ax6.tick_params(labelsize=7)

    path = os.path.join(REPORT_DIR, 'eda.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F0F1A')
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_model_comparison(results: list):
    names  = [r['model'] for r in results]
    r2s    = [r['R2']   for r in results]
    rmses  = [r['RMSE'] for r in results]
    rmsles = [r['RMSLE']for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Model Comparison', fontsize=18, fontweight='bold', color='#E0E0F0')
    fig.patch.set_facecolor('#0F0F1A')
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    for ax, vals, title in zip(
        axes,
        [r2s, [v/1000 for v in rmses], rmsles],
        ['R² Score (higher=better)', 'RMSE ($k, lower=better)', 'RMSLE (lower=better)']
    ):
        bars = ax.barh(names, vals, color=colors, alpha=0.85, height=0.6)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_facecolor('#1A1A2E')
        ax.spines[:].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                    f'{v:.3f}', va='center', fontsize=9, color='#E0E0F0')

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, 'model_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F0F1A')
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_predictions(results: list, y_test: np.ndarray):
    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows*5))
    fig.suptitle('Actual vs Predicted — All Models',
                 fontsize=18, fontweight='bold', color='#E0E0F0')
    fig.patch.set_facecolor('#0F0F1A')
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax   = axes[i]
        pred = res['y_pred']
        ax.scatter(y_test/1000, pred/1000, alpha=0.35, s=12,
                   color=PALETTE[i % len(PALETTE)])
        lo = min(y_test.min(), pred.min()) / 1000
        hi = max(y_test.max(), pred.max()) / 1000
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect')
        ax.set_title(f"{res['model']}\nR²={res['R2']:.4f}  RMSE=${res['RMSE']/1000:.1f}k",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Actual ($k)')
        ax.set_ylabel('Predicted ($k)')
        ax.set_facecolor('#1A1A2E')
        ax.spines[:].set_color('#2D2D4E')
        ax.legend(fontsize=8, framealpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, 'predictions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F0F1A')
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_feature_importance(model, feature_names: list, title='Feature Importance'):
    if not hasattr(model, 'feature_importances_'):
        return
    imp  = model.feature_importances_
    idxs = np.argsort(imp)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0F0F1A')
    ax.set_facecolor('#1A1A2E')
    ax.barh([feature_names[i] for i in idxs][::-1], imp[idxs][::-1],
            color=plt.cm.plasma(np.linspace(0.1, 0.9, len(idxs))), alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#E0E0F0')
    ax.set_xlabel('Importance', color='#A0A0C0')
    ax.spines[:].set_color('#2D2D4E')

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, 'feature_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F0F1A')
    plt.close(fig)
    print(f"  Saved → {path}")