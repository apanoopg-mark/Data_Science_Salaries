import pandas as pd
from scipy.stats import zscore
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("DataScience_salaries_2024.csv")
print (df.head())
DuplicateCount = df.duplicated().sum()

# ------------------------------------- this is the duplicates
print("the duplicate are ",DuplicateCount, "rows")
print(df[df.duplicated(keep=False)])
Cleaned= DfCleaned = df.drop_duplicates()
CleanedData = DfCleaned.duplicated().sum()
print("the duplicate are ",CleanedData, "rows after the cleaning")

# ------------------------------------- this is the missing value
print(df.isna().sum())
print("there is no missing value")

# ------------------------------------- this is the Aggregation (group by)
JobSalaryAgg = df.groupby("job_title")["salary_in_usd"].agg(
    MeanSalary="mean",
    MaxSalary="max",
    JobCount="count"
)
print(JobSalaryAgg)

ExperienceAgg = df.groupby("experience_level")["salary_in_usd"].agg(
    MeanSalary="mean",
    MaxSalary="min",
    JobCount="max"
)
print(ExperienceAgg)
# ------------------------------------- Unique values
print(df.nunique())
print("all values are unique")

# ------------------------------------- Outleirs detection
df["salary_in_usd"].describe()

# ------------------ IQR
Q1 = df["salary_in_usd"].quantile(0.25)
Q3 = df["salary_in_usd"].quantile(0.75)
IQR = Q3 - Q1

OutliersIqr = df[
    (df["salary_in_usd"] < Q1 - 1.5 * IQR) |
    (df["salary_in_usd"] > Q3 + 1.5 * IQR)
]
print(OutliersIqr)
# ------------------ Z-score
df["salary"] = zscore(df["salary_in_usd"])
Outliers_z = df[df["salary"].abs() > 3]
print(Outliers_z)

# ------------------------------------- Outleirs handling
# ------------------ IQR
Q1 = df["salary_in_usd"].quantile(0.25)
Q3 = df["salary_in_usd"].quantile(0.75)
IQR = Q3 - Q1

LowerBound = Q1 - 1.5 * IQR
UpperBound = Q3 + 1.5 * IQR

DfNoOutliers = df[
    (df["salary_in_usd"] >= LowerBound) &
    (df["salary_in_usd"] <= UpperBound)
]
# ------------------ Capping
DfCapped = df.copy()

DfCapped["salary_in_usd"] = np.where(
    DfCapped["salary_in_usd"] > UpperBound,
    UpperBound,
    np.where(
        DfCapped["salary_in_usd"] < LowerBound,
        LowerBound,
        DfCapped["salary_in_usd"]
    )
)
# ------------------------------------- Sorting
df.sort_values(by="salary_in_usd", ascending=True)

# ------------------------------------- KNN
encoder = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

#---------- Define features and target
X = df.drop("salary_in_usd", axis=1)
y = df["salary_in_usd"]

#---------- Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#---------- Scale features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#---------- Train KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

#---------- Make predictions
predictions = knn.predict(X_test)

#---------- Evaluate model
mae = mean_absolute_error(y_test, predictions)
print("KNN Mean Absolute Error:", mae)
# ------------------------------------- Tree decision

#---------- Train Decision Tree model
dt = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)
dt.fit(X_train, y_train)

#---------- Predict
y_pred_dt = dt.predict(X_test)

#---------- Evaluate Decision Tree
print("MAE :", mean_absolute_error(y_test, y_pred_dt))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print("R2  :", r2_score(y_test, y_pred_dt))

# ------------------------------------- Export
DfCleaned.to_csv(
    "DataScience_salaries_2024_cleaned.csv",
    index=False
)
