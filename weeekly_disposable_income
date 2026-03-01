"""
This script is a linear regression model that predicts weekly disposable
income using age and weekly income as predictors. It uses OLS linear regression
for intepretability, where the coefficients translate directly into £/week
changes.
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# style for the global plot
plt.rcParams.update({
    'font.family':        'serif',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linestyle':     '--',
})


def load_data(file_path, ):
    """
    reads and returns the data from excel
    :param file_path: string containing file path
    """
    sheet_name="ModelData"
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        raise SystemExit(f"Error: '{file_path}' not found. Check the path and try again.")

    required = [
        "Weekly Income (£)",
        "Total Essential Cost (£/week)",
        "Age (midpoint)",
        "Disposable Income (£/week)",
        "Age Group",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing columns in sheet: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values.")

    return df


def build_features(df):
    """
    prepares the input for the model
    :param df: array
    """

    df = df.copy()
    df["Age_x_Income"] = df["Age (midpoint)"] * df["Weekly Income (£)"]

    X = df[["Age (midpoint)", "Weekly Income (£)", "Age_x_Income"]].values
    y = df["Disposable Income (£/week)"].values
    return X, y, df


def fit_model(X, y):
    """
    fits the linear regression and prints the results
    :param X: array
    :param y: array
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2_train = model.score(X_train, y_train)
    r2_test  = model.score(X_test,  y_test)
    mae      = mean_absolute_error(y_test, model.predict(X_test))

    print("\n── Model Results ──────────────────────────────────")
    print(f"  R² (train): {r2_train:.4f}")
    print(f"  R²  (test): {r2_test:.4f}")
    print(f"  MAE (test): £{mae:.2f}/wk")
    print(f"  Intercept:  {model.intercept_:.4f}")
    for name, coef in zip(["Age", "Income", "Age×Income"], model.coef_):
        print(f"  {name:<12}: {coef:.6f}")
    print("───────────────────────────────────────────────────\n")

    return model, X_train, X_test, y_train, y_test


def plot_actual_vs_predicted(y_test, y_pred_test):
    """
    draws the scatter graph of real vs predicted values
    :param y_test: array
    :param x_test: array
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(y_test, y_pred_test, alpha=0.75, color='steelblue',
               edgecolors='k', linewidths=0.4, zorder=3, label='Test observations')
    lims = [min(y_test.min(), y_pred_test.min()),
            max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect fit (y = x)')

    ax.set_xlabel("Actual Disposable Income (£/week)")
    ax.set_ylabel("Predicted Disposable Income (£/week)")
    ax.set_title("Actual vs Predicted Disposable Income\n(test set only)", fontsize=13)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_residuals(y_pred_test, residuals):
    """
    creates residual plot to check assumptions
    :param y_pred_test: array
    :param residuals: array
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(y_pred_test, residuals, alpha=0.7, color='#2E4057',
               edgecolors='k', linewidths=0.3, zorder=3)
    ax.axhline(0, color='red', linestyle='--', lw=2)

    ax.set_xlabel("Predicted Disposable Income (£/week)")
    ax.set_ylabel("Residual (Actual − Predicted) (£/week)")
    ax.set_title("Residual Plot — Checking Model Assumptions", fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_income_vs_disposable(df, model):
    """
    creates the visuals for the regression line
    :param df: array
    :param model: linear regression object
    """
    col_income     = "Weekly Income (£)"
    col_disposable = "Disposable Income (£/week)"
    col_age        = "Age (midpoint)"

    age_groups = sorted(df["Age Group"].unique())
    cmap       = plt.cm.get_cmap('tab10', len(age_groups))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ag in enumerate(age_groups):
        mask = df["Age Group"] == ag
        ax.scatter(df.loc[mask, col_income], df.loc[mask, col_disposable],
                   label=f"Age group: {ag}", color=cmap(i),
                   s=65, alpha=0.85, zorder=3)

    # regression line
    income_range = np.linspace(df[col_income].min(), df[col_income].max(), 300)
    age_mean     = df[col_age].mean()
    X_line = np.column_stack([
        np.full(300, age_mean),
        income_range,
        np.full(300, age_mean) * income_range,
    ])
    y_line = model.predict(X_line)

    ax.plot(income_range, y_line, color='red', linewidth=2.5,
            label=f'Regression line (age held at mean: {age_mean:.0f} yrs)')

    c = model.coef_
    eq = (f"D = {model.intercept_:.2f}\n"
          f"+ ({c[0]:.4f} × Age)\n"
          f"+ ({c[1]:.4f} × Income)\n"
          f"+ ({c[2]:.6f} × Age×Income)")
    ax.text(0.04, 0.96, eq, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.55))

    ax.set_title("Weekly Income vs Disposable Income by Age Group", fontsize=13)
    ax.set_xlabel("Weekly Income (£)")
    ax.set_ylabel("Disposable Income (£/week)")
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    FILE_PATH = "" # add file path

    df = load_data(FILE_PATH)
    X, y, df = build_features(df)
    model, X_train, X_test, y_train, y_test = fit_model(X, y)

    y_pred_test = model.predict(X_test)
    residuals = y_test - y_pred_test

    plot_actual_vs_predicted(y_test, y_pred_test)
    plot_residuals(y_pred_test, residuals)
    plot_income_vs_disposable(df, model)
