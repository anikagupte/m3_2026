from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# style for the global plot
plt.rcParams.update({
'font.family': 'serif',
'axes.spines.top': False,
'axes.spines.right': False,
'axes.grid': True,
'grid.alpha': 0.25,
'grid.linestyle': '--',
})

def load_data(file_path, sheet_name="ModelData"):
	"""
	Reads the Excel sheet and drops rows with missing values in the columns/
	"""
	try:
		df = pd.read_excel(file_path, sheet_name=sheet_name)
		print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")
	except FileNotFoundError:
		raise SystemExit(f"Error: '{file_path}' not found. Check the path and try again.")

	required = [
		"Weekly Income (£)",
		"Total Essential Cost (£/wk)",
		"Age (midpoint)",
		"Disposable Income (£/wk)",
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
	Adds an interaction term (Age × Income) to capture the idea that income's
	effect on disposable income may differ across age groups 
	"""
	df = df.copy()
	df["Age_x_Income"] = df["Age (midpoint)"] * df["Weekly Income (£)"]

	X = df[["Age (midpoint)", "Weekly Income (£)", "Age_x_Income"]].values
	y = df["Disposable Income (£/wk)"].values
	return X, y, df


def fit_model(X, y):
	"""
	Splits 80/20 into train and test sets before fitting. This is the difference 
	between knowing how well the model fits and knowing how well it generalises.
	random_state=42 makes the split reproducible.
	"""
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	model = LinearRegression()
	model.fit(X_train, y_train)

	r2_train = model.score(X_train, y_train)
	r2_test = model.score(X_test, y_test)
	mae = mean_absolute_error(y_test, model.predict(X_test))

	print("\n Model Results ")
	print(f" R² (train): {r2_train:.4f}")
	print(f" R² (test): {r2_test:.4f}")
	print(f" MAE (test): £{mae:.2f}/wk")
	print(f" Intercept: {model.intercept_:.4f}")
	for name, coef in zip(["Age", "Income", "Age×Income"], model.coef_):
		print(f" {name:<12}: {coef:.6f}")
	

	return model, X_train, X_test, y_train, y_test


def plot_actual_vs_predicted(y_test, y_pred_test):
	"""
    Ideally, points should cluster around the red dashed line (perfect fit).
	"""
	fig, ax = plt.subplots(figsize=(9, 6))

	ax.scatter(y_test, y_pred_test, alpha=0.75, color='steelblue',
		edgecolors='k', linewidths=0.4, zorder=3, label='Test observations')
	lims = [min(y_test.min(), y_pred_test.min()),
		max(y_test.max(), y_pred_test.max())]
	ax.plot(lims, lims, 'r--', lw=2, label='Perfect fit (y = x)')

	ax.set_xlabel("Actual Disposable Income (£/wk)")
	ax.set_ylabel("Predicted Disposable Income (£/wk)")
	ax.set_title("Actual vs Predicted Disposable Income\n(test set only)", fontsize=13)
	ax.legend()
	fig.tight_layout()
	plt.show()


def plot_residuals(y_pred_test, residuals):
	"""
	Residuals should appear as a horizontal band centred on zero with no
	obvious pattern. A funnel shape would suggest heteroscedasticity —
	the model's errors grow with predicted income, which is common in
	financial data and worth acknowledging.
	"""
	fig, ax = plt.subplots(figsize=(9, 5))

	ax.scatter(y_pred_test, residuals, alpha=0.7, color='#2E4057',
		edgecolors='k', linewidths=0.3, zorder=3)
	ax.axhline(0, color='red', linestyle='--', lw=2)

	ax.set_xlabel("Predicted Disposable Income (£/wk)")
	ax.set_ylabel("Residual (Actual − Predicted) (£/wk)")
	ax.set_title("Residual Plot — Checking Model Assumptions", fontsize=13)
	fig.tight_layout()
	plt.show()


def plot_income_vs_disposable(df, model):
	"""
	Shows the regression line with age held at its mean. Points are coloured by age group
	so the reader can judge whether age groupings align with the model's slope.
	"""
	col_income = "Weekly Income (£)"
	col_disposable = "Disposable Income (£/wk)"
	col_age = "Age (midpoint)"

	age_groups = sorted(df["Age Group"].unique())
cmap = plt.cm.get_cmap('tab10', len(age_groups))

fig, ax = plt.subplots(figsize=(10, 6))

for i, ag in enumerate(age_groups):
	mask = df["Age Group"] == ag
	ax.scatter(df.loc[mask, col_income], df.loc[mask, col_disposable],
		label=f"Age group: {ag}", color=cmap(i),
		s=65, alpha=0.85, zorder=3)

# Regression line: vary income, fix age at mean, fix interaction accordingly
income_range = np.linspace(df[col_income].min(), df[col_income].max(), 300)
age_mean = df[col_age].mean()
X_line = np.column_stack([
	np.full(300, age_mean),
	income_range,
	np.full(300, age_mean) * income_range, # interaction term
])
y_line = model.predict(X_line)

ax.plot(income_range, y_line, color='red', linewidth=2.5,
	label=f'Regression line (age held at mean: {age_mean:.0f} yrs)')

# Show the equation
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
ax.set_ylabel("Disposable Income (£/wk)")
ax.legend(fontsize=8, loc='lower right')
fig.tight_layout()
plt.show()


def sensitivity_analysis(model, df):
	col_income = "Weekly Income (£)"
	col_age = "Age (midpoint)"

	age_mean = df[col_age].mean()
	income_mean = df[col_income].mean()

	print("\nSensitivity Analysis ")
	print(f" Base prediction (mean age {age_mean:.0f}, mean income £{income_mean:.0f}):")
	
	if len(model.coef_) == 3:
		base = model.predict([[age_mean, income_mean, age_mean * income_mean]])[0]
	else:
		base = model.predict([[age_mean, income_mean]])[0]
	print(f"  -> GBP {base:.2f}/wk\n")

	# Vary income, hold age at mean
	print(" Effect of income change (age held at mean):")
	for income in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2800, 3200, 3600, 4000]:
		if len(model.coef_) == 3:
			pred = model.predict([[age_mean, income, age_mean * income]])[0]
		else:
			pred = model.predict([[age_mean, income]])[0]
		print(f"  Income £{income:<6} -> GBP {pred:.2f}/wk")

	# Vary age, hold income at mean
	print("\n Effect of age change (income held at mean):")
	for age in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
		if len(model.coef_) == 3:
			pred = model.predict([[age, income_mean, age * income_mean]])[0]
		else:
			pred = model.predict([[age, income_mean]])[0]
		diff = pred - base
		print(f"  Age {age:<4} -> GBP {pred:.2f}/wk  ({diff:+.2f} vs base)")

	# Plot sensitivity curves
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))

	# Income sweep
	incomes = np.linspace(200, 4000, 300)
	if len(model.coef_) == 3:
		preds_income = model.predict([[age_mean, i, age_mean * i] for i in incomes])
	else:
		preds_income = model.predict([[age_mean, i] for i in incomes])
	axes[0].plot(incomes, preds_income, color='steelblue', lw=2)
	axes[0].set_xlabel("Weekly Income (GBP)")
	axes[0].set_ylabel("Predicted Disposable Income (GBP/wk)")
	axes[0].set_title(f"Income Sensitivity\n(age fixed at {age_mean:.0f})")

	# Age sweep
	ages = np.linspace(20, 100, 300)
	if len(model.coef_) == 3:
		preds_age = model.predict([[a, income_mean, a * income_mean] for a in ages])
	else:
		preds_age = model.predict([[a, income_mean] for a in ages])
	axes[1].plot(ages, preds_age, color='darkorange', lw=2)
	axes[1].set_xlabel("Age (years)")
	axes[1].set_ylabel("Predicted Disposable Income (GBP/wk)")
	axes[1].set_title(f"Age Sensitivity\n(income fixed at GBP {income_mean:.0f})")

	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	FILE_PATH = r"c:\Users\mshah\.vscode\m3models\expenditure (3).xlsx"

	df = load_data(FILE_PATH)
	X, y, df = build_features(df)
	model, X_train, X_test, y_train, y_test = fit_model(X, y)

	y_pred_test = model.predict(X_test)
	residuals = y_test - y_pred_test

	plot_actual_vs_predicted(y_test, y_pred_test)
	plot_residuals(y_pred_test, residuals)
	plot_income_vs_disposable(df, model)
	sensitivity_analysis(model, df)

	print("── Predict Disposable Income ──────────────────────")
	print(" Enter values below to get a prediction.")
	print(" Type 'quit' at any prompt to exit.\n")

	while True:
		try:
			age_input = input(" Age (years): ").strip()
			if age_input.lower() == 'quit':
				break

			income_input = input(" Weekly Income (GBP): ").strip()
			if income_input.lower() == 'quit':
				break

			age = float(age_input)
			income = float(income_input)

			if len(model.coef_) == 3:
				X_input = [[age, income, age * income]]
			else:
				X_input = [[age, income]]

			prediction = model.predict(X_input)[0]
			print(f"\n -> Predicted Disposable Income: GBP {prediction:.2f}/wk\n")

			again = input(" Predict another? (yes/no): ").strip().lower()
			if again != 'yes':
				break
			print()

		except ValueError as e:
			print(f" Invalid input ({e}) - please enter a number.\n")
