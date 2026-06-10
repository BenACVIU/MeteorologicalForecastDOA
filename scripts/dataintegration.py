"""
Author: Benjamin Arroquia Cuadros
12/06/2023

Join dataframes script.

Dependencies: sqlite, numpy, pandas

Workflow of data processing:
1. Read meteo data extracted four hours of data by day.
2. Group data by mean of data by day.
3. Get 24 hours differences of each variable of meteo data.
4. Apply rolling functions by each method and provinces.
5. Merge meteo data and hospital admissions 
6. Load data into database.

Use case: merge two datasets: cases hospital admissions and
        meteorological data, creating a time series with several 
        variables such as diference of temperature in 24 hours.
"""


import sqlite3
import pandas as pd
import settings

gdf = None

def minmax_meteo_hipo(row):
    dft = gdf.loc[row.index, ['min_DOA', 'max_DOA']]
    return  dft.iloc[1, :].min() - dft.iloc[0, :].max()

def minmax_meteo_hiper(row):
    dft = gdf.loc[row.index, ['min_DOA', 'max_DOA']]
    return dft.iloc[1, :].max() - dft.iloc[0, :].min()

def minmax_pres_hipo(row):
    dft = gdf.loc[row.index, ['min_PRE', 'max_PRE']]
    return dft.iloc[1, :].min() - dft.iloc[0, :].max()

def minmax_pres_hiper(row):
    dft = gdf.loc[row.index, ['min_PRE', 'max_PRE']]
    return dft.iloc[1, :].max() - dft.iloc[0, :].min()

def minmax_temp_hipo(row):
    dft = gdf.loc[row.index, ['min_TEM', 'max_TEM']]
    return dft.iloc[1, :].min() - dft.iloc[0, :].max()

def minmax_temp_hiper(row):
    dft = gdf.loc[row.index, ['min_TEM', 'max_TEM']]
    return dft.iloc[1, :].max() - dft.iloc[0, :].min()

def minmax_humi_hipo(row):
    dft = gdf.loc[row.index, ['min_RHU', 'max_RHU']]
    return dft.iloc[1, :].min() - dft.iloc[0, :].max()

def minmax_humi_hiper(row):
    dft = gdf.loc[row.index, ['min_RHU', 'max_RHU']]
    return dft.iloc[1, :].max() - dft.iloc[0, :].min()


def get_meteodata_by_day():
    global gdf
    pathdb = settings.DB_PATH
    sql = """SELECT * FROM stats_cluster"""
    # Read sqlite query results into a pandas DataFrame
    con = sqlite3.connect(pathdb)
    df = pd.read_sql_query(sql, con)
    con.close()
    # work with times in string format
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.strftime('%Y%m%d')
    # rename values in df in order to short words
    list(df.variable.unique())
    ls_names = ['DOA', 'RHU', 'TEM', 'PRE']
    dc_names = dict(zip(list(df.variable.unique()), ls_names))
    df["variable"].replace(dc_names, inplace=True)
    df.cod_entity = df["cod_entity"].astype(float).astype(int)
    ls_col = ['method', 'variable', 'time', 'cod_entity', 'mean']
    col_agg = ['method', 'time', 'cod_entity', 'variable']
    # Get values per day 
    gdf = df.loc[:, ls_col].groupby(by=col_agg).agg(['mean', 'max', 'min'])
    gdf = gdf.unstack(level=-1)
    gdf.columns = [f'{lev1}_{lev2}' for col, lev1, lev2 in gdf.columns]
    gdf.reset_index(inplace=True)

    # Loop over provinces and methods to get gradient statistics
    ls_dfs = []
    for i, prov in enumerate(gdf['cod_entity'].unique()):
        sel_prov = gdf.loc[(gdf['cod_entity'] == prov)].copy()
        for m, method in enumerate(gdf['method'].unique()):
            sel_pm = sel_prov.loc[(sel_prov['method'] == method)].copy()
            sel_pm.sort_values("time", inplace=True)
            # print(sel_pm.iloc[0, :3])
            ro_w2 = sel_pm["mean_DOA"].rolling(window=2, center=False,
                                            min_periods=2)
            sel_pm['doa_hipo'] = ro_w2.apply(minmax_meteo_hipo, raw=False)
            sel_pm['doa_hiper'] = ro_w2.apply(minmax_meteo_hiper, raw=False)
            sel_pm['pre_hipo'] = ro_w2.apply(minmax_pres_hipo, raw=False)
            sel_pm['pre_hiper'] = ro_w2.apply(minmax_pres_hiper, raw=False)
            sel_pm['tem_hipo'] = ro_w2.apply(minmax_temp_hipo, raw=False)
            sel_pm['tem_hiper'] = ro_w2.apply(minmax_temp_hiper, raw=False)
            sel_pm['rhu_hipo'] = ro_w2.apply(minmax_humi_hipo, raw=False)
            sel_pm['rhu_hiper'] = ro_w2.apply(minmax_humi_hiper, raw=False)
            ls_dfs.append(sel_pm.copy())
    gdf2 = pd.concat(ls_dfs)
    return gdf2.copy()

def get_hospital_population():
    con = sqlite3.connect(settings.DB_PATH)
    sql = """SELECT * FROM cmbdcompleto"""
    df_hosp = pd.read_sql_query(sql, con)
    con.close()
    df_hosp['time'] = pd.to_datetime(df_hosp['time'])
    df_hosp['time'] = df_hosp['time'].dt.strftime('%Y%m%d')
    return df_hosp

def df_to_sql(df):
    path = settings.DB_PATH
    con = sqlite3.connect(path)
    df.to_sql(name='biometcmbd03_18', con=con, if_exists='replace')
    con.close()

if __name__ == "__main__":
    df_metep = get_meteodata_by_day()
    df_hosp = get_hospital_population()
    df_pm = gdf.set_index(["time", "cod_entity"]).join(
            df_hosp.set_index(["time", "cod_entity"]), how='left')
    df_to_sql(df_pm)
    print("Finish")