"""
Created on Thu Feb 29 06:59:12 2024

@author: OLUWATOBA ADEOYE (23032031)
"""
# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Reading the data frame
df_capita_country = "C:/Users/user/Desktop/ADS1/CO2perGDP (1).csv"
df_capita = pd.read_csv(df_capita_country, skiprows = 4)

df_capita.columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + list(df_capita.columns[4:])

df_capita.head()

df_capita.describe()

# List of countries needed
countries_needed = ['Senegal', 'Sri Lanka', 'Kenya', 'France', 'Algeria', 'Suriname', 
                    'Ghana', 'United States', 'Malawi', 'Papua New Guinea', 'Nepal', 
                    'Colombia', 'Seychelles', 'Lesotho', 'Philippines', 'Puerto Rico', 
                    'Mexico', 'Singapore', 'Japan', 'Cote dIvoire', 'Fiji', 'Rwanda', 
                    'Hong Kong SAR', 'China', 'Australia', 'Congo, Dem. Rep', 
                    'St. Vincent and the Grenadines', 'Sweden', 'Portugal', 'United Kingdom']

# Selecting the countries needed
df_capita = df_capita[df_capita['Country Name'].isin(countries_needed)]
print(df_capita)

df_capita = df_capita.drop(['Indicator Name', 'Country Code', 'Indicator Code'], axis=1)
print(df_capita)

# Resetting the index
df_capita.reset_index(drop=True, inplace=True)
print(df_capita)

# Extracting years from our urban out dataset
df_capita_s=df_capita[['Country Name','1990', '2000','2010', '2019']]
df_capita_s.describe()

# Checking for missing values
df_capita_s.isna().sum()

# Transposing the data
df_capita_t = df_capita_s.T
df_capita_t.columns = df_capita_t.iloc[0]
df_capita_t = df_capita_t.iloc[1:]
df_capita_t.describe()
df_capita_t = df_capita_t.apply(pd.to_numeric, errors='coerce')

# Sorting by mean and selecting top 10 countries
top_10_countries = df_capita_t.mean().sort_values(ascending=False).head(10).index
df_top_10 = df_capita_t[top_10_countries]

# Plotting for top 10 countries
df_top_10.plot(figsize=(10, 6))
plt.title('GDP per Capita Over Time for Top 10 Countries')
plt.xlabel('Year')
plt.ylabel('GDP per Capita')
plt.legend(title='Country')

# In this data, i will be working with 2019 data
# Extracting 30 years of data at an interval of 10 years from out dataset
df_capita_year=df_capita_s[['1990', '2000','2010', '2019']]
df_capita_year.describe()

# Checking for missing values
df_capita_year.isna().sum()

# Create a copy of the DataFrame
df_capita_year_copy = df_capita_year.copy().dropna()

# Checking for correlation between our years choosen
corr = df_capita_year_copy.corr()
print(corr)

# Plotting the heatmap
plt.figure(figsize=(12, 10))  # Adjusted figure size for clarity
sns.heatmap(corr, cmap='viridis', annot=True, fmt=".2f", annot_kws={'size': 10})
plt.title("Correlation Heatmap of GDP per Capita Over Years", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Year", fontsize=14)
plt.xticks(fontsize=12)  # Adjust x-axis labels' font size
plt.yticks(fontsize=12)  # Adjust y-axis labels' font size
plt.show()

# Transposing
df_capita_trans = df_capita.T

# Assuming df_capita_trans is your transposed DataFrame
df_capita_trans.columns = df_capita_trans.iloc[0]  # Setting the first row as the header
df_capita_years = df_capita_trans[1:].copy()  # Dropping the first row if it's now redundant

# Convert all data to numeric, forcing non-convertible values to NaN
df_capita_trans_numeric = df_capita_years.apply(pd.to_numeric, errors='coerce')

# Now, calculate the mean across each country (column), skipping NaN values
country_means = df_capita_trans_numeric.mean(skipna=True)

# Get the top 5 countries by their mean GDP per capita
top_5_countries = country_means.nlargest(5)

# Customizing the color palette for the pie chart
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']

# Using the explode parameter to highlight the top country
explode = (0.1, 0, 0, 0, 0)  # only "explode" the 1st slice (i.e., 'Top Country')

# Plotting the pie chart for the top 5 countries by GDP per capita
plt.figure(figsize=(10, 8))
plt.pie(top_5_countries.values, labels=top_5_countries.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)

plt.title('Top 5 Countries with Highest Average GDP per Capita', fontsize=16)
plt.legend(top_5_countries.index, title="Countries", loc="best")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# Selecting only the years the data frame
df_capita_years=df_capita_trans.iloc[1:]
df_capita_years


df_capita_years=df_capita_years.apply(pd.to_numeric)
df_capita_years

# Resetting the index
df_capita_years.reset_index(inplace=True)
df_capita_years

# Renaming the country name to years
df_capita_years.rename(columns={'index': 'Year'}, inplace=True)
df_capita_years

# Checking the values of Fiji and years
df_capita_years['Fiji'].values
print(df_capita_years.head())
df_capita_years['Year'].values

df_capita_years.dtypes

df_capita_years['Year'] = df_capita_years['Year'].astype('int')
df_capita_years

df_capita_years.dtypes
df_capita_years
df_capita_years.columns

# Exponential function for Fiji
def exponential(t, a, b):
    """Computes exponential growth
    
    Parameters:
        t: The current time
        a: The initial value
        b: The growth rate
        
    Returns:
        The value at the given time
    """
    return a * np.exp(b * t)

years = df_capita_years['Year'].values
GDP = df_capita_years['Fiji'].values

# Provide initial guess for exponential function
initial_guess = [min(GDP), 0.01]  # You can adjust the initial guess if needed

try:
    popt, pcov = curve_fit(exponential, years, GDP, p0=initial_guess, maxfev=10000)
except RuntimeError as e:
    print("Curve fitting failed:", str(e))
    popt = initial_guess
    
# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
curve_GDP = exponential(curve_years, *popt)

# Predictions for 2030 and 2040
prediction_years = np.array([2030, 2040])
predictions = exponential(prediction_years, *popt)
print("GDP per capita prediction for 2030:", predictions[0])
print("GDP per capita prediction for 2040:", predictions[1])

# Plot the data, fitted curve, and predictions
plt.plot(years, GDP, 'ro', label='Data')
plt.plot(curve_years, curve_GDP, 'b-', label='Fitted Curve')
plt.plot(prediction_years, predictions, 'g*', label='Predictions')
plt.plot([curve_years[-1], prediction_years[0]], [curve_GDP[-1], predictions[0]], 'g--')
plt.plot([curve_years[-1], prediction_years[1]], [curve_GDP[-1], predictions[1]], 'g--')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('Exponential Growth Fit for GDP per capita of Fiji')
plt.legend('outer')
plt.grid(True)
plt.show()

# Fitting for United States using Polynomial function
def polynomial(t, *coefficients):
    """Computes a polynomial function
    
    Parameters:
        t: The current time
        coefficients: Coefficients of the polynomial function
        
    Returns:
        The value at the given time
    """
    return np.polyval(coefficients, t)

# Obtain the years and GDP data
years = df_capita_years['Year'].values
GDP = df_capita_years['United States'].values

# Define the degree of the polynomial
degree = 3

# Perform polynomial curve fitting
coefficients = np.polyfit(years, GDP, degree)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
curve_GDP = polynomial(curve_years, *coefficients)

# Predictions for 2030 and 2040
prediction_years = np.array([2030, 2040])
predictions = polynomial(prediction_years, *coefficients)
print("GDP per capita prediction for 2030:", predictions[0])
print("GDP per capita prediction for 2040:", predictions[1])

# Plot the data, fitted curve, and predictions
plt.plot(years, GDP, 'ro', label='Data')
plt.plot(curve_years, curve_GDP, 'b-', label='Fitted Curve')
plt.plot(prediction_years, predictions, 'g*', label='Predictions')
plt.plot([curve_years[-1], prediction_years[0]], [curve_GDP[-1], predictions[0]], 'g--')
plt.plot([curve_years[-1], prediction_years[1]], [curve_GDP[-1], predictions[1]], 'g--')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('GDP per capita Polynomial Fitting of United States')
plt.legend()
plt.grid(True)
plt.show()

# Fitting for Colombia using Polynomial function
def polynomial(t, *coefficients):
    """Computes a polynomial function
    
    Parameters:
        t: The current time
        coefficients: Coefficients of the polynomial function
        
    Returns:
        The value at the given time
    """
    return np.polyval(coefficients, t)

# Obtain the years and GDP data
years = df_capita_years['Year'].values
GDP = df_capita_years['Colombia'].values

# Define the degree of the polynomial
degree = 3

# Perform polynomial curve fitting
coefficients = np.polyfit(years, GDP, degree)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
curve_GDP = polynomial(curve_years, *coefficients)

# Predictions for 2030 and 2040
prediction_years = np.array([2030, 2040])
predictions = polynomial(prediction_years, *coefficients)
print("GDP per capita prediction for 2030:", predictions[0])
print("GDP per capita prediction for 2040:", predictions[1])

# Plot the data, fitted curve, and predictions
plt.plot(years, GDP, 'ro', label='Data')
plt.plot(curve_years, curve_GDP, 'b-', label='Fitted Curve')
plt.plot(prediction_years, predictions, 'g*', label='Predictions')
plt.plot([curve_years[-1], prediction_years[0]], [curve_GDP[-1], predictions[0]], 'g--')
plt.plot([curve_years[-1], prediction_years[1]], [curve_GDP[-1], predictions[1]], 'g--')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('GDP per capita Polynomial Fitting of Colombia')
plt.legend()
plt.grid(True)
plt.show()
