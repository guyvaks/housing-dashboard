# Housing Dashboard

[English README](README_EN.md)

דאשבורד אינטראקטיבי ב-Streamlit לניתוח נתוני דיור וחיזוי מחירי נכסים.

## תכונות

- **פילטרים** (sidebar): מחיר, חדרי שינה/רחצה, קומות, מצב הנכס, חזית מים, שטח מגורים, שנת בנייה
- **KPIs**: מחיר ממוצע/חציוני, מספר נכסים, שטח מגורים ממוצע, חדרי שינה ממוצע
- **גרפים**: התפלגות מחירים, שטח מגורים מול מחיר, מחיר ממוצע לפי חדרי שינה, מפת קורלציות
- **מודל רגרסיה לינארית**: preprocessing עם `StandardScaler`, חלוקת train/test 80/20, הצגת R², Adjusted R², RMSE, MAE וגרף actual vs predicted
- **טופס חיזוי אינטראקטיבי**: הזנת מאפייני נכס וקבלת הערכת מחיר מהמודל

## התקנה והרצה

```bash
pip install -r requirements.txt
streamlit run housing_dashboard.py
```

## נתונים

הקובץ `Project housing data .csv` מכיל 2999 נכסים עם 12 פיצ'רים מספריים (כולל `price` כמשתנה המטרה).
