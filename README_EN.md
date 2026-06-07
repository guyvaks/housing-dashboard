# Housing Dashboard

An interactive Streamlit dashboard for exploring housing data and predicting property prices.

## Features

- **Filters** (sidebar): price, bedrooms/bathrooms, floors, condition, waterfront, living area, year built
- **KPIs**: average/median price, number of properties, average living area, average bedrooms
- **Charts**: price distribution, living area vs. price, average price by bedroom count, correlation heatmap
- **Linear regression model**: preprocessing with `StandardScaler`, 80/20 train/test split, displays R², Adjusted R², RMSE, MAE, and an actual-vs-predicted plot
- **Interactive prediction form**: enter property attributes and get a price estimate from the model

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run housing_dashboard.py
```

## Data

`Project housing data .csv` contains 2999 properties with 12 numeric features (including `price` as the target variable).
