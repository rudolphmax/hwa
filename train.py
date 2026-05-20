import joblib
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

model = svm.SVC()
df = pd.read_csv("df.csv", index_col = 0)

scaler = StandardScaler()
scaler.fit(df.iloc[:, :-1].to_numpy())
X_train, X_test, y_train, y_test = train_test_split(
  scaler.transform(df.iloc[:, :-1]),
  df["target"],
  test_size = 0.3,
  random_state = 20,
)
model.fit(X_train, y_train)

joblib.dump(scaler, "scaler1.joblib")
joblib.dump(model, "model1.pkl")

model = svm.SVC()
df = pd.read_csv("df_p2.csv", index_col = 0)

scaler = MinMaxScaler()
scaler.fit(df.iloc[:, :-1].to_numpy())
X_train, X_test, y_train, y_test = train_test_split(
  scaler.transform(df.iloc[:, :-1]),
  df["target"],
  test_size = 0.3,
  random_state = 20,
)
model.fit(X_train, y_train)

joblib.dump(scaler, "scaler2.joblib")
joblib.dump(model, "model2.pkl")
