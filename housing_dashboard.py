import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="דאשבורד נדל\"ן", layout="wide")

DATA_PATH = "Project housing data .csv"
TARGET = "price"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()
features = [c for c in df.columns if c != TARGET]

# ---------------- Sidebar filters ----------------
st.sidebar.header("פילטרים")

price_min, price_max = int(df[TARGET].min()), int(df[TARGET].max())
price_range = st.sidebar.slider("טווח מחיר", price_min, price_max, (price_min, price_max))

bedrooms_sel = st.sidebar.multiselect("חדרי שינה", sorted(df["bedrooms"].unique()), default=sorted(df["bedrooms"].unique()))
bathrooms_sel = st.sidebar.multiselect("חדרי רחצה", sorted(df["bathrooms"].unique()), default=sorted(df["bathrooms"].unique()))
floors_sel = st.sidebar.multiselect("קומות", sorted(df["floors"].unique()), default=sorted(df["floors"].unique()))
condition_sel = st.sidebar.multiselect("מצב הנכס", sorted(df["condition"].unique()), default=sorted(df["condition"].unique()))
waterfront_sel = st.sidebar.multiselect("חזית מים (waterfront)", sorted(df["waterfront"].unique()), default=sorted(df["waterfront"].unique()))

sqft_min, sqft_max = int(df["sqft_living"].min()), int(df["sqft_living"].max())
sqft_range = st.sidebar.slider("שטח מגורים (sqft_living)", sqft_min, sqft_max, (sqft_min, sqft_max))

yr_min, yr_max = int(df["yr_built"].min()), int(df["yr_built"].max())
yr_range = st.sidebar.slider("שנת בנייה", yr_min, yr_max, (yr_min, yr_max))

filtered = df[
    (df[TARGET].between(*price_range)) &
    (df["bedrooms"].isin(bedrooms_sel)) &
    (df["bathrooms"].isin(bathrooms_sel)) &
    (df["floors"].isin(floors_sel)) &
    (df["condition"].isin(condition_sel)) &
    (df["waterfront"].isin(waterfront_sel)) &
    (df["sqft_living"].between(*sqft_range)) &
    (df["yr_built"].between(*yr_range))
]

st.title("🏠 דאשבורד אינטראקטיבי - נתוני דיור")
st.markdown(f"מציג **{len(filtered)}** מתוך **{len(df)}** נכסים לאחר סינון")

# ---------------- KPIs ----------------
st.subheader("מדדים מרכזיים (KPIs)")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("מחיר ממוצע", f"${filtered[TARGET].mean():,.0f}" if len(filtered) else "—")
k2.metric("מחיר חציוני", f"${filtered[TARGET].median():,.0f}" if len(filtered) else "—")
k3.metric("מספר נכסים", f"{len(filtered):,}")
k4.metric("שטח מגורים ממוצע", f"{filtered['sqft_living'].mean():,.0f} sqft" if len(filtered) else "—")
k5.metric("חדרי שינה ממוצע", f"{filtered['bedrooms'].mean():.1f}" if len(filtered) else "—")

st.divider()

# ---------------- Charts ----------------
st.subheader("ויזואליזציות")

if len(filtered) == 0:
    st.warning("אין נתונים להצגה עבור הסינון הנוכחי.")
else:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**התפלגות מחירים**")
        fig, ax = plt.subplots()
        sns.histplot(filtered[TARGET], bins=40, kde=True, ax=ax)
        ax.set_xlabel("מחיר")
        st.pyplot(fig)

    with c2:
        st.markdown("**שטח מגורים מול מחיר**")
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered, x="sqft_living", y=TARGET, hue="condition", palette="viridis", ax=ax)
        ax.set_xlabel("שטח מגורים")
        ax.set_ylabel("מחיר")
        st.pyplot(fig)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**מחיר ממוצע לפי מספר חדרי שינה**")
        fig, ax = plt.subplots()
        grp = filtered.groupby("bedrooms")[TARGET].mean().reset_index()
        sns.barplot(data=grp, x="bedrooms", y=TARGET, ax=ax, color="steelblue")
        ax.set_xlabel("חדרי שינה")
        ax.set_ylabel("מחיר ממוצע")
        st.pyplot(fig)

    with c4:
        st.markdown("**מפת קורלציות בין הפיצ'רים**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(filtered.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

st.divider()

# ---------------- Model ----------------
st.subheader("מודל רגרסיה לחיזוי מחיר")


@st.cache_resource
def train_model():
    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    n, p = X_test.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return model, scaler, r2, adj_r2, rmse, mae, y_test, y_pred


model, scaler, r2, adj_r2, rmse, mae, y_test, y_pred = train_model()

st.markdown("המודל אומן באמצעות **רגרסיה לינארית** על 80% מהנתונים (train), לאחר נרמול הפיצ'רים (StandardScaler), ונבחן על 20% הנותרים (test).")

with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{r2:.3f}")
    m2.metric("Adjusted R²", f"{adj_r2:.3f}")
    m3.metric("RMSE", f"${rmse:,.0f}")
    m4.metric("MAE", f"${mae:,.0f}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.4)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--")
ax.set_xlabel("מחיר אמיתי")
ax.set_ylabel("מחיר חזוי")
ax.set_title("מחיר אמיתי מול מחיר חזוי (test set)")
st.pyplot(fig)

st.divider()

# ---------------- Prediction form ----------------
st.subheader("חיזוי מחיר לנכס חדש")
st.markdown("הזן את מאפייני הנכס וקבל הערכת מחיר מהמודל:")

with st.form("predict_form"):
    f1, f2, f3 = st.columns(3)
    with f1:
        in_bedrooms = st.number_input("חדרי שינה", min_value=0, max_value=20, value=int(df["bedrooms"].median()))
        in_bathrooms = st.number_input("חדרי רחצה", min_value=0, max_value=20, value=int(df["bathrooms"].median()))
        in_sqft_living = st.number_input("שטח מגורים (sqft_living)", min_value=0, value=int(df["sqft_living"].median()))
        in_sqft_lot = st.number_input("שטח מגרש (sqft_lot)", min_value=0, value=int(df["sqft_lot"].median()))
    with f2:
        in_floors = st.number_input("קומות", min_value=1, max_value=10, value=int(df["floors"].median()))
        in_waterfront = st.selectbox("חזית מים (waterfront)", sorted(df["waterfront"].unique()))
        in_view = st.selectbox("דירוג נוף (view)", sorted(df["view"].unique()))
        in_condition = st.selectbox("מצב הנכס (condition)", sorted(df["condition"].unique()))
    with f3:
        in_sqft_above = st.number_input("שטח מעל הקרקע (sqft_above)", min_value=0, value=int(df["sqft_above"].median()))
        in_sqft_basement = st.number_input("שטח מרתף (sqft_basement)", min_value=0, value=int(df["sqft_basement"].median()))
        in_yr_built = st.number_input("שנת בנייה", min_value=1800, max_value=2026, value=int(df["yr_built"].median()))

    submitted = st.form_submit_button("חזה מחיר")

if submitted:
    input_row = pd.DataFrame([{
        "bedrooms": in_bedrooms,
        "bathrooms": in_bathrooms,
        "sqft_living": in_sqft_living,
        "sqft_lot": in_sqft_lot,
        "floors": in_floors,
        "waterfront": in_waterfront,
        "view": in_view,
        "condition": in_condition,
        "sqft_above": in_sqft_above,
        "sqft_basement": in_sqft_basement,
        "yr_built": in_yr_built,
    }])[features]

    input_scaled = scaler.transform(input_row)
    predicted_price = model.predict(input_scaled)[0]

    st.success(f"המחיר החזוי לנכס: **${predicted_price:,.0f}**")
