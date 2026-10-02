
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , r2_score
df = pd.read_csv("insurance.csv")
df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)
x = df_encoded.drop("charges", axis = 1)
y = df_encoded['charges']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=42 )
model = LinearRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test , y_pred)
r2 = r2_score(y_test, y_pred)
coefficients = pd.DataFrame({'feature':x.columns,'Coefficient':model.coef_})
new_client = pd.DataFrame([{
    'age':30,
    'bmi': 25.0,
    'children':1,
    'sex_male': 1,
    'region_northwest': 1,
    'region_southeast': 0,
    'region_southwest': 0,
    'smoker_yes': 0
}])
new_client = new_client[x.columns]
predicted_cost = model.predict(new_client)
print(f"nEstimated insurance for new client is ${predicted_cost[0]:.2f}")