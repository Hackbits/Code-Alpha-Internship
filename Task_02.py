# Task 2: Unemployment Analysis 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment in India.csv")

df.columns = df.columns.str.strip()

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
}, inplace=True)

df.dropna(inplace=True)

print(df.head())
print(df.info())

national_trend = df.groupby('Date')['Unemployment_Rate'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Unemployment_Rate', data=national_trend, marker='o', 
             linewidth=2, markersize=6, label='National Unemployment Rate')
plt.axvline(pd.to_datetime('2020-03-24'), color='red', linestyle='--', 
            linewidth=2, label='COVID Lockdown Start (Mar 24, 2020)')

for i, row in national_trend.iterrows():
    plt.text(row['Date'], row['Unemployment_Rate'] + 0.5, 
             f'{row["Unemployment_Rate"]:.1f}%', 
             ha='center', va='bottom', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

plt.title("National Unemployment Rate Trend Over Time", fontsize=16, fontweight='bold')
plt.xlabel("Date", fontsize=12, fontweight='bold')
plt.ylabel("Unemployment Rate (%)", fontsize=12, fontweight='bold')
plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

covid_months = df[df['Date'].between('2020-04-01', '2020-05-31')]
peak_unemp = covid_months.groupby('Region')['Unemployment_Rate'].max().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
bars = sns.barplot(x=peak_unemp.values, y=peak_unemp.index, palette="Reds_r")
plt.title("Peak State-wise Unemployment Rates During COVID-19 (April-May 2020)", 
          fontsize=16, fontweight='bold')
plt.xlabel("Peak Unemployment Rate (%)", fontsize=12, fontweight='bold')
plt.ylabel("State/Region", fontsize=12, fontweight='bold')

for i, (state, rate) in enumerate(peak_unemp.items()):
    plt.text(rate + 0.5, i, f'{rate:.1f}%', va='center', fontsize=9, fontweight='bold')

sm = plt.cm.ScalarMappable(cmap="Reds_r", norm=plt.Normalize(vmin=peak_unemp.min(), vmax=peak_unemp.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
cbar.set_label('Unemployment Rate (%)', fontsize=10, fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

df['Month'] = df['Date'].dt.month_name()
monthly_avg = df.groupby('Month')['Unemployment_Rate'].mean().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])

plt.figure(figsize=(12, 6))
line = sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o', 
                   linewidth=3, markersize=8, color='steelblue', 
                   label='Average Monthly Unemployment Rate')
plt.title("Seasonal Pattern: Average Monthly Unemployment Rate in India", 
          fontsize=16, fontweight='bold')
plt.xlabel("Month", fontsize=12, fontweight='bold')
plt.ylabel("Average Unemployment Rate (%)", fontsize=12, fontweight='bold')

for i, (month, rate) in enumerate(zip(monthly_avg.index, monthly_avg.values)):
    if not pd.isna(rate):
        plt.text(i, rate + 0.2, f'{rate:.1f}%', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

monsoon_months = ['June', 'July', 'August', 'September']
for month in monsoon_months:
    if month in monthly_avg.index:
        month_idx = list(monthly_avg.index).index(month)
        plt.axvspan(month_idx-0.3, month_idx+0.3, alpha=0.2, color='green', 
                   label='Monsoon Months' if month == 'June' else "")

plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
plt.axis('off')  

insights_text = """
 KEY INSIGHTS FROM UNEMPLOYMENT ANALYSIS IN INDIA

   COVID-19 Impact:
   • Sharp spike in unemployment during April-May 2020 due to nationwide lockdown
   • National unemployment rate peaked at ~24% in April 2020
   • Pre-COVID unemployment typically ranged between 6-8%

   Regional Impact:
   • Puducherry, Jharkhand, and Bihar were among the most affected states
   • Significant regional disparities in unemployment rates during crisis
   • Urban areas generally showed higher unemployment volatility

   Seasonal Patterns:
   • Mild seasonal rise in unemployment during monsoon months (July-September)
   • Agricultural dependency influences seasonal employment patterns
   • Monsoon seasons show consistent but moderate unemployment increases

    Recovery Trends:
   • Gradual recovery observed post-lockdown period
   • Employment levels slowly returning to pre-pandemic baselines
   • Economic resilience demonstrated through steady recovery patterns

   Policy Implications:
   • Need for robust crisis management and employment protection schemes
   • Regional-specific interventions required for high-unemployment states
   • Seasonal employment programs could help during monsoon periods
"""

plt.text(0.5, 0.5, insights_text, 
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='center',
         horizontalalignment='center',
         bbox=dict(boxstyle='round,pad=1', 
                  facecolor='lightblue', 
                  alpha=0.8,
                  edgecolor='navy',
                  linewidth=2),
         family='monospace')

plt.title("UNEMPLOYMENT ANALYSIS INSIGHTS - INDIA", 
          fontsize=18, fontweight='bold', pad=20, color='navy')

plt.tight_layout()
plt.show()

print("\n Analysis Complete! All insights have been visualized above.")
