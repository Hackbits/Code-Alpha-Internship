import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("car_data.csv")
df_vis = df.copy()
df['Car_Age'] = 2025 - df['Year']
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

plt.figure(figsize=(10, 8))
numerical_df = df.select_dtypes(include=[np.number])
mask = np.triu(np.ones_like(numerical_df.corr(), dtype=bool))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", 
            linewidths=0.5, mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix of Numerical Car Features", fontsize=16, fontweight='bold')
plt.xlabel("Features", fontsize=12, fontweight='bold')
plt.ylabel("Features", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

price_bins = [0, 3, 6, 10, 20]
price_labels = ['Low (<3L)', 'Mid (3–6L)', 'High (6–10L)', 'Premium (10L+)']
df_vis['Price_Category'] = pd.cut(df_vis['Selling_Price'], bins=price_bins, labels=price_labels)
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_vis, x='Price_Category', palette='Set2', 
                   order=['Low (<3L)', 'Mid (3–6L)', 'High (6–10L)', 'Premium (10L+)'])
plt.title("Car Count by Price Category", fontsize=14, fontweight='bold')
plt.xlabel("Selling Price Category (in Lakhs)", fontsize=12, fontweight='bold')
plt.ylabel("Number of Cars", fontsize=12, fontweight='bold')

for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=sns.color_palette('Set2')[i], label=label) 
                  for i, label in enumerate(['Low Price', 'Mid Price', 'High Price', 'Premium Price'])]
plt.legend(handles=legend_elements, loc='upper right', title='Price Categories', 
          title_fontsize=10, fontsize=9)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(data=df_vis, x='Present_Price', y='Selling_Price', 
                         alpha=0.7, s=50, label='Car Data Points')
plt.title("Relationship: Selling Price vs Present Price", fontsize=14, fontweight='bold')
plt.xlabel("Present Price (in Lakhs)", fontsize=12, fontweight='bold')
plt.ylabel("Selling Price (in Lakhs)", fontsize=12, fontweight='bold')

z = np.polyfit(df_vis['Present_Price'], df_vis['Selling_Price'], 1)
p = np.poly1d(z)
plt.plot(df_vis['Present_Price'], p(df_vis['Present_Price']), "r--", 
         alpha=0.8, linewidth=2, label=f'Trend Line (R² = {np.corrcoef(df_vis["Present_Price"], df_vis["Selling_Price"])[0,1]**2:.3f})')

correlation = df_vis['Present_Price'].corr(df_vis['Selling_Price'])
plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes,
         fontsize=10, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
df_vis['Car_Age'] = 2025 - df_vis['Year']
scatter = sns.scatterplot(data=df_vis, x='Car_Age', y='Selling_Price', 
                         alpha=0.6, s=50, label='Car Data Points')
plt.title("Car Age vs Selling Price Relationship", fontsize=14, fontweight='bold')
plt.xlabel("Car Age (Years)", fontsize=12, fontweight='bold')
plt.ylabel("Selling Price (in Lakhs)", fontsize=12, fontweight='bold')

z = np.polyfit(df_vis['Car_Age'], df_vis['Selling_Price'], 1)
p = np.poly1d(z)
correlation = df_vis['Car_Age'].corr(df_vis['Selling_Price'])
plt.plot(df_vis['Car_Age'], p(df_vis['Car_Age']), "r--", alpha=0.8, linewidth=2, 
         label=f'Trend Line (Negative Correlation)')

plt.text(0.7, 0.9, f'Correlation: {correlation:.3f}', 
         transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

if correlation < -0.5:
    interpretation = "Strong Negative Relationship"
elif correlation < -0.3:
    interpretation = "Moderate Negative Relationship"
else:
    interpretation = "Weak Negative Relationship"

plt.text(0.02, 0.02, f'Interpretation: {interpretation}', 
         transform=plt.gca().transAxes, fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.6))

plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 LINEAR REGRESSION MODEL EVALUATION:")
print("="*50)
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} Lakhs")
print(f"Root Mean Square Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} Lakhs")
print(f"R² Score (Coefficient of Determination): {r2_score(y_test, y_pred):.3f}")
print("="*50)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, s=50, label='Predicted vs Actual', color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', 
         linewidth=2, label='Perfect Prediction Line')

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
plt.text(0.02, 0.98, f'R² Score: {r2:.3f}\nMAE: {mae:.2f} Lakhs', 
         transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8))

plt.title("Linear Regression: Actual vs Predicted Car Prices", fontsize=14, fontweight='bold')
plt.xlabel("Actual Selling Price (in Lakhs)", fontsize=12, fontweight='bold')
plt.ylabel("Predicted Selling Price (in Lakhs)", fontsize=12, fontweight='bold')
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
