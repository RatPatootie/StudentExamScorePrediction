import pickle
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# custom files
import model_best_hyperparameters
import columns
import warnings
warnings.simplefilter('ignore')

# read train data
#ds = pd.read_csv("train_data.csv")
ds = pd.read_csv("..\\data\\train.csv")

# feature engineering
# Missing data imputation

def impute_na(df, variable, value):
    return df[variable].fillna(value)


# Let's create a dict and impute mode values
mode_impute_values = dict()
for column in columns.mode_impute_columns:
    mode_impute_values[column] = ds[column].mode()[0]
    ds[column] = impute_na(ds, column, mode_impute_values[column])

# Outlier Engineering
ds = ds[ds['exam_score'] <= 100].reset_index(drop=True)

def find_skewed_boundaries(df, variable, distance):
    df[variable] = pd.to_numeric(df[variable],errors='coerce')
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

upper_lower_limits = dict()
for column in columns.outlier_columns:
    upper_lower_limits[column+'_upper_limit'], upper_lower_limits[column+'_lower_limit'] = find_skewed_boundaries(ds, column, 5)
for column in columns.outlier_columns:
    ds = ds[~ np.where(ds[column] > upper_lower_limits[column+'_upper_limit'], True,
                       np.where(ds[column] < upper_lower_limits[column+'_lower_limit'], True, False))]

# Categorical encoding
map_dicts=columns.mapping_dict
ds.replace(map_dicts, inplace=True)
ds=ds.astype(int)


#нормалізація
scaler = MinMaxScaler()
ds[columns.normis] = scaler.fit_transform(ds[columns.normis])

# Збереження скейлера у файл
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f) 
     
# save parameters 
param_dict = {
              'mode_impute_values':mode_impute_values,
              'upper_lower_limits':upper_lower_limits,
              'map_dicts':map_dicts,
             }
with open('param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Define target and features columns
X = ds[columns.X_columns]
y = ds[columns.y_column]

# Let's say we want to split the data in 90:10 for train:test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)

#smote

kn = Ridge(**model_best_hyperparameters.params)
kn.fit(X_train, y_train)
y_pred = kn.predict(X_test)

# Обчислення регресійних метрик
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Вивід метрик
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R²: {r2:.2f}')

# Збереження метрик у файл
with open('../docs/regression_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f'MSE: {mse:.2f}\n')
    f.write(f'MAE: {mae:.2f}\n')
    f.write(f'R: {r2:.2f}\n')

print('Звіт про метрики збережено у файл regression_metrics.txt')

# Важливість ознак за допомогою перестановок
result = permutation_importance(kn, X_test[:300], y_test[:300], n_repeats=10, random_state=42)

# Створення DataFrame з результатами важливості ознак
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': result.importances_mean})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Збереження важливості ознак у файл CSV
importance_df.to_csv('../docs/feature_importance.csv', index=False)

# Збереження моделі у файл за допомогою pickle
filename = '../models/finalized_model.sav'
pickle.dump(kn, open(filename, 'wb'))
print('Звіт про метрики та модель збережено.')