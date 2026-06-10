"""
Author: Benjamin Arroquia Cuadros
10/06/2026

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.metrics import mean_absolute_percentage_error
import sqlite3


# 1. MÓDULO DE PREPARACIÓN DE DATOS
# ==========================================
def prepare_data(df, met, prov, y_col, ls_col):
    """
    Filtra, limpia y genera las características temporales necesarias.
    """
    mask = (df['method'] == met) & (df['cod_entity'] == prov)
    df_sel = df.loc[mask, :].copy()

    # Formato fecha y orden cronológico
    df_sel['time'] = pd.to_datetime(df_sel['time'])
    df_sel.sort_values(by="time", inplace=True)

    # Variable objetivo desplazada
    df_sel['y_lag_1'] = df_sel[y_col].shift(-1)
    df_sel.dropna(subset=['y_lag_1'], inplace=True)

    # Año Estacional
    df_sel['epi_year'] = np.where(df_sel['time'].dt.month < 7, 
                                  df_sel['time'].dt.year - 1, 
                                  df_sel['time'].dt.year)

    y = df_sel['y_lag_1']
    X = df_sel.loc[:, ls_col].copy() 

    # Preprocesamiento de variables temporales
    X['quarter'] = df_sel['time'].dt.quarter
    # X['month'] = df_sel['time'].dt.month
    # X['day_of_week'] = df_sel['time'].dt.dayofweek
    X = pd.get_dummies(data=X, columns=['quarter'])
    X["weekend"] = df_sel['time'].dt.dayofweek > 4
    X['epi_year'] = df_sel['epi_year']

    unique_years = sorted(X['epi_year'].unique())
    
    return X, y, unique_years, df_sel['time']


# 2. MÓDULO DE RANDOM SEARCH
# ==========================================
def tune_hyperparameters(X_train, y_train, base_estimator, param_distributions, random_state, n_iter=10):
    
    print(f"Optimizando {base_estimator.__class__.__name__} en la ventana inicial...")
    tscv = TimeSeriesSplit(n_splits=3) 
    
    # Clonamos el modelo base para no alterar el original
    model_to_tune = clone(base_estimator)
    if 'random_state' in model_to_tune.get_params():
        model_to_tune.set_params(random_state=random_state)
    
    search = RandomizedSearchCV(
        estimator=model_to_tune,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=random_state, 
        n_jobs=-1
    )
    
    search.fit(X_train, y_train.values.ravel())
    best_params = search.best_params_
    
    if 'random_state' in model_to_tune.get_params():
        best_params['random_state'] = random_state
        
    print(f"Mejores parámetros: {best_params}")
    return best_params

# 3. MÓDULO DE EVALUACIÓN
# ==========================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Calcula y devuelve las métricas de rendimiento del modelo.
    Incluye R2, MAE, MAPE y RMSE.
    """
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)), # <--- NUEVO RMSE TEST
        
        'R2_train': r2_score(y_train, y_pred_train),
        'MAE_train': mean_absolute_error(y_train, y_pred_train),
        'MAPE_train': mean_absolute_percentage_error(y_train, y_pred_train),
        'RMSE_train': np.sqrt(mean_squared_error(y_train, y_pred_train)) # <--- NUEVO RMSE TRAIN
    }
    return metrics, y_pred, y_pred_train

# 4. MÓDULO PRINCIPAL: ROLLING WINDOW (Corregido para Años Epidemiológicos)
# ==========================================
def run_rolling_window_validation(X, y, unique_years, window_size, base_estimator, param_distributions, random_state=42):
    """
    Implementación de un Rolling Window Cross-Validation respetando años epidemiológicos.
    Agrupa el último año incompleto con el penúltimo para la evaluación final.
    """
    best_params = None
    dc_metrics = {}
    final_model = None 
    
    # Identificamos cuál es el último y penúltimo año en nuestra lista
    ultimo_ano = unique_years[-1]
    penultimo_ano = unique_years[-2]

    # Ajustamos el límite del bucle. 
    # Como el penúltimo y último año se evalúan juntos, el bucle termina antes de llegar al último.
    limite_bucle = len(unique_years) - 1 

    for i in range(window_size, limite_bucle):
        
        # 1. Definir Ventana de Entrenamiento (Rolling Window estricta de tamaño 'window_size')
        train_years = unique_years[i - window_size : i]
        
        # 2. Definir Ventana de Test
        # Si estamos en la iteración que debe predecir el penúltimo año (ej. 2019),
        # lo agrupamos con el último año (ej. 2020, que termina en diciembre).
        if unique_years[i] == penultimo_ano:
            test_years = [penultimo_ano, ultimo_ano]
        else:
            # Para el resto de iteraciones, simplemente predice el año siguiente
            test_years = [unique_years[i]] 
        
        train_mask = X['epi_year'].isin(train_years)
        test_mask = X['epi_year'].isin(test_years)
        
        print(f"Train epi_years: {train_years} | Test epi_years: {test_years}")
        
        X_train = X[train_mask].drop(columns=['epi_year'])
        y_train = y[train_mask]
        
        X_test = X[test_mask].drop(columns=['epi_year'])
        y_test = y[test_mask]

        # Tuning de hiperparámetros (solo en la primera ventana)
        if i == window_size:
            best_params = tune_hyperparameters(
                X_train, y_train, base_estimator, param_distributions, random_state
            )
            
        # --- ENTRENAMIENTO ---
        final_model = clone(base_estimator)
        final_model.set_params(**best_params)
        final_model.fit(X_train, y_train.values.ravel())
        
        # --- EVALUACIÓN ---
        # Si agrupamos años, lo indicamos en la etiqueta (ej. "test_2019_2020")
        test_label = f"test_{'_'.join(map(str, test_years))}"
        metrics, _, _ = evaluate_model(final_model, X_train, y_train, X_test, y_test)
        dc_metrics[test_label] = metrics
        
        print(f"-> Evaluación para test ({test_label}) completada.\n")

    return dc_metrics, best_params


Y_COL = 'num_cases'
LS_COL = ['mean_DOA', 'mean_PRE', 'mean_RHU', 'mean_TEM', 'max_DOA', 'max_PRE',
       'max_RHU', 'max_TEM', 'min_DOA', 'min_PRE', 'min_RHU', 'min_TEM',
       'doa_hipo', 'doa_hiper', 'pre_hipo', 'pre_hiper', 'tem_hipo',
       'tem_hiper', 'rhu_hipo', 'rhu_hiper']
window_size_rollwind = 6
random_state = 42
param_distributions = {
    'max_depth': range(3, 20),
    'min_samples_split': range(5, 50),
}

# 2. Definición de espacios de búsqueda por algoritmo
dt_param_distributions = {
    'max_depth': range(2, 15),
    'min_samples_split': range(6, 50),
    'min_samples_leaf': range(6, 50),
    'max_features': ['sqrt', 'log2', None]
}

rf_param_distributions = {
    'n_estimators': range(50, 300),
    'max_depth': range(3, 20),
    'min_samples_split': range(5, 50),
    'min_samples_leaf': range(5, 50),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

lr_param_distributions = {
    'fit_intercept': [True, False]
}

pathdb = "./../data/GeoBiomet.db"
sql1 = """SELECT * FROM biometcmbd03_18_model"""
con = sqlite3.connect(pathdb)
df = pd.read_sql_query(sql1, con)
con.close()
df['num_cases'] = (df['DIAG1'] * 1000000)/ df['total_inter'] 
df["time"] = pd.to_datetime(df["dia"], format="%Y%m%d")

ls_metrics = []
valores_unicos = df[['cod_entity', 'method']].drop_duplicates()
for index, row in valores_unicos.iterrows():
    base_estimator = RandomForestRegressor(n_jobs=-1, random_state=random_state)
    print(row['cod_entity'],  row['method'])
    X, y, unique_years, _ = prepare_data(df, met=row['method'], prov=row['cod_entity'], y_col=Y_COL, ls_col=LS_COL)
    
    dc_metrics = run_rolling_window_validation(X, y, unique_years,  window_size=window_size_rollwind,
                                  base_estimator=base_estimator, param_distributions=rf_param_distributions)
    metadata = {
        'provincia': row['cod_entity'],
        'algoritmo': 'RFscikit',
        'metodo': row['method']
    }

    ls_metrics.append((dc_metrics[0], dc_metrics[1], metadata))


flattened_data = []

# Ahora desempaquetamos 3 elementos si aplicas la sugerencia
for metrics_dict, params_dict, metadata in ls_metrics:
    for window_name, metrics in metrics_dict.items():
        row_data = {}
        row_data.update(metadata)      # 1. Provincia, algoritmo, método
        row_data['ventana_cv'] = window_name # 2. La ventana de tiempo
        row_data.update(metrics)       # 3. Métricas de error y precisión
        row_data.update(params_dict)   # 4. Hiperparámetros
        
        flattened_data.append(row_data)

df_resultados_completos = pd.DataFrame(flattened_data)

df_resultados_completos.to_csv('./data/results_rollingwindow.csv')