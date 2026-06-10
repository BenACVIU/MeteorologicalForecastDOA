"""
Author: Benjamin Arroquia Cuadros
10/06/2026

Module to create a layer with clusters to extract data from raster files.

Dependencies: osgeo, numpy, pandas, pysal

Install GDAL: 
https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html
Note, check ogrinfo: pip install GDAL==<GDAL VERSION FROM OGRINFO>

Workflow of data processing:
1. Define provinces or polygonal entities to extract data from raster
2. Read process files and polygonal data
3. Process clusters 
4. Save geodataframe into GPKG
"""
import settings
import numpy as np
import pandas as pd
import geopandas as gpd               # Spatial data manipulation
from pysal.explore import esda   # Exploratory Spatial analytics
from pysal.lib import weights    # Spatial weights
from spopt.region.skater import Skater
# Clustering 
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise as skm
# Visualization
# %matplotlib widget
import matplotlib.pyplot as plt
import functools


class SpatialClustering:
    col_prov = ['NAMEUNIT', 'COD_UNIT']
    def __init__(self, dfg, df_prov, col_name, province=None, year=None):
        """
        Class for spatial cluster analisys.
        First two geopandas ralated needed to find a province.
        Select this province with name or part of it.
        dfg : geodataframe of geopandas. 
            Polygons. This should be in cartesian coordinates.
            Municipalities with population
        df_prov : geodataframe of geopandas. 
            Polygons. This should be in cartesian coordinates.
            Names of provinces related with councils gdf
        col_name : str
            Name of column with data of population
        province : str
            Name of province. This search if contains the string.
            First ocurrence is selected.
        """
        self.dfg = dfg
        self.dfprov = df_prov
        self.lisa = None
        self.df_sel = None
        self.name_unit = None
        self.cod_unit = None
        self.w = None
        self.col_name = col_name
        self.df_diss = None
        self.dc_dfg = {}
        if province and col_name:
            self.selection(province)
            
    def set_unit_selection(self, province):
        self.selection(province)
        
    def population_clusters(self, one_method=None, verbose=1, gdf=None):
        """
        Here the relevance in process.
        Select criteria by density pop.
        No matter pop or wathever variable.
        Next improve look for select like Decision trees with entropy.
        """
        if gdf is None:
            gdf = self.df_sel
        cols = [c for c in gdf.columns if '_label' in c]
        if one_method:
            if f"{one_method}_label" in cols:
                cols = [f"{one_method}_label"]
        for i, col in enumerate(cols):
            gdf_diss = gdf.dissolve(by=col, aggfunc=['sum'])
            gdf_diss['pop_dens'] = gdf_diss[(self.col_name, "sum")] / gdf_diss.geometry.area * 1000000
            gdf_diss.sort_values(by='pop_dens', ascending=False, inplace=True)
            # A custom evaluation metric.  
            gdf_diss["eval"] = np.sqrt(gdf_diss[(self.col_name, "sum")] + gdf_diss["pop_dens"]**2)
            gdf_diss = gdf_diss.loc[:, [(self.col_name, "sum"), "pop_dens", "eval", "geometry"]]
            gdf_diss.reset_index(inplace=True)
            gdf_diss.columns = [col, self.col_name, "pop_dens", "eval", "geometry"]
            gdf_diss['entity'] =  self.name_unit
            gdf_diss['cod_entity'] = self.cod_unit
            #TODO: index correct reset one
            if verbose == 1:
                print(gdf_diss.loc[:, [col, self.col_name, "pop_dens", "eval"]])
            self.dc_dfg[col] = gdf_diss.copy()

    def evaluate_clustering(self, col):
        #TODO: return shilouette and population over unit
        sil = silhouette_score(self.df_sel.loc[:, ["xutm", "yutm"]], self.df_sel[col].values)
        print("Shilouette:", sil)
    
    def set_cluster_column(self, dc):
        name_col = dc['method'] + '_label'
        self.df_sel[name_col] = dc['clusters']
    
    def set_column(func):
        @functools.wraps(func)
        def set_column_dec(self, *args, **kwargs):
            dc = func(self, *args, **kwargs)
            name_col = dc['method'] + '_label'
            self.df_sel[name_col] = dc['clusters']
            return dc
        return set_column_dec
        
    def selection(self, prov_name):
        # search province
        sel_cont = self.dfprov["NAMEUNIT"].str.contains(prov_name)
        prov = self.dfprov.loc[sel_cont, self.col_prov].values[0]
        self.name_unit = prov[0]
        self.cod_unit = prov[1]
        # select rows by province name
        self.df_sel = self.dfg.loc[self.dfg["COD_UNIT"] == str(int(self.cod_unit)), 
                          ["geometry", self.col_name, "NAMEUNIT"]].copy()
        # update max value in selection sample
        # val = self.df_sel.sort_values(self.col_name, ascending=False).iloc[1, 1]
        # self.df_sel.loc[self.df_sel[self.col_name].idxmax(), self.col_name] = val
        # get density
        self.df_sel["pop_dens"] = self.df_sel[self.col_name] / self.df_sel.geometry.area * 1000000
        self.df_sel["xutm"] = self.df_sel.geometry.centroid.x
        self.df_sel["yutm"] = self.df_sel.geometry.centroid.y
        # Weights matrix in province
        self.w = weights.contiguity.Queen.from_dataframe(self.df_sel)
        
    @set_column
    def get_moran_local(self):
        col_lag = f"{self.col_name}_lag"
        col_lag_std = f"{self.col_name}_lag_std"
        col_std = f"{self.col_name}_std"
        self.df_sel[col_lag] = weights.spatial_lag.lag_spatial(self.w, self.df_sel[self.col_name])
        self.df_sel[col_std] = (self.df_sel[self.col_name] - self.df_sel[self.col_name].mean())
        self.df_sel[col_lag_std] = (self.df_sel[col_lag] - self.df_sel[col_lag].mean())
        moran = esda.moran.Moran(y=self.df_sel[self.col_name], w=self.w, permutations=999)
        # potencia de 1/999+1, p < 0.001
        self.lisa = esda.moran.Moran_Local(self.df_sel[self.col_name], self.w)
        clustering = dict(name_unit=self.name_unit, clusters=self.lisa.q, 
                          method='lisa',
                          moranI=moran.I,moranIp=moran.p_sim)
        return clustering
    
    @set_column
    def get_skater_clustering(self, k=5):
        n_clusters = k
        floor = 5
        trace = False
        islands = "ignore"
        spanning_forest_kwds = dict(
            dissimilarity=skm.euclidean_distances, affinity=None, 
            reduction=np.sum, center=np.mean
        )
        model = Skater(self.df_sel, self.w, [self.col_name], n_clusters, floor, trace, 
                       islands, spanning_forest_kwds)
        model.solve()
        clustering = dict(name_unit=self.name_unit, clusters=model.labels_,
                         method='skater')
        return clustering 

    @set_column
    def get_kmeans(self, k=5):
        ls_cols = [self.col_name, "xutm", "yutm"]
        # scaler = StandardScaler()
        kmeans = KMeans(init="k-means++", n_clusters=k, 
                    n_init=4, random_state=0)
        estimator = make_pipeline(StandardScaler(), kmeans).fit(self.df_sel.loc[:, ls_cols])
        clustering = dict(name_unit=self.name_unit, clusters=estimator[-1].labels_,
                         method='kmeans')
        return clustering

    def dissolve_clusters(self, k):
        self.get_moran_local()
        self.get_skater_clustering(k=k)
        self.get_kmeans(k=k)
        self.population_clusters(verbose=0)
        cols = [c for c in self.df_sel.columns if '_label' in c]
        for i, df in self.dc_dfg.items():
            # df has to be ordered by density
            if 'lisa' not in i:
                clas_max = df.loc[df.index == 0, i].values[0]
                df.loc[df.index == 1, i] = clas_max
            else:
                clas_max = 1
            df = df.loc[df[i] == clas_max, :].copy()
            # Copy just polygons to zonal statistics and assign 1 label
            df.loc[df[i] == clas_max, i] = 1
            self.population_clusters(verbose=0, gdf=df)
        
    def plot_clustering_compare(self, cols):
        #TODO: plot every cluster method in df_sel
        # Set up figure and axes
        f, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        axs = axs.flatten()
        for i, col in enumerate(cols):
            self.df_sel.plot(column=col, categorical=True, cmap='Paired',
                             linewidth=0.1, edgecolor='white', legend=True, ax=axs[i])
            axs[i].set_title(col)
        f.tight_layout()
        # Display the figure
        plt.show()
        pass

    def plot_clustering_dissolved(self):
        #TODO: plot every cluster method in df_sel
        # Set up figure and axes
        f, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        axs = axs.flatten()
        cols = [c for c in self.df_sel.columns if '_label' in c]
        for i, col in enumerate(cols):
            self.dc_dfg[col].plot(column=col, categorical=True, cmap='Paired',
                             linewidth=0.1, edgecolor='white', legend=True, ax=axs[i])
            axs[i].set_title(col)
        f.tight_layout()
        # Display the figure
        plt.show()
    
    def save_data_dissolved(self, path):
        for k, dfg in self.dc_dfg.items():
            dfg.to_file(path, layer=f"{self.name_unit}_{k}", driver="GPKG")

    def get_dfg_joined_dissolved(self):
        """
        Return gdf with all dissolved and selected clusters. 
        """
        if len(self.dc_dfg.keys()) == 0:
            raise Exception("There is no geodf dissolved yet, \
                            run dissolve_clusters defining a k") 
        
        ls_df = []
        for k, dfv in self.dc_dfg.items():
            ls_c = [[i, i.split('_')[0]] for i in dfv.columns if '_label' in i]
            name_col, method = ls_c[0]
            dfv['k_method'] = method
            dfv = dfv.drop([name_col], axis=1)
            ls_df.append(dfv)
        return pd.concat(ls_df)
    

def get_province_row(df_prov, df_pob, name_unit, ls_columns):
    """
    Create a geodataframe for province.

    Parameters
    ----------
    df_pob: geodataframe, of polygons and attribute use for clustering
        LAYER_POPULATION
        Column CLUSTER_COLUM_NAME used to clustering process.
    df_prov: geodataframe, of entities over municipalities.
        LAYER_PROVINCES
    name_unit: str, name of unit to select
    ls_columns: list, columns to select for final dataframe.

    Return 
    ----------
    type:
        geodataframe
    """
    sel_cont = df_prov["NAMEUNIT"].str.contains(name_unit)
    prov = df_prov.loc[sel_cont, :].copy()
    cod = prov["COD_UNIT"].values[0]
    df_sel = df_pob.loc[df_pob["COD_UNIT"] == str(int(cod)), 
                            ["geometry", 'POB16', "NAMEUNIT"]].copy()
    pop_tot = df_sel['POB16'].sum() 
    dens = pop_tot / prov.geometry.area * 1000000
    prov['POB16'] = pop_tot
    prov['pop_dens'] = dens
    prov['eval'] = np.sqrt(pop_tot + dens**2)
    prov['k_method'] = 'province'
    prov = prov.rename(columns={'NAMEUNIT': 'entity', 'COD_UNIT': 'cod_entity'})
    return prov.loc[:, ls_columns]


def process_spatial_clustering(ls_prov, df_pob, df_prov, k):
    """
    Process polygon layer to obtain a geodataframe 
    with clusters based in attribute.

    Parameters
    ----------
    df_pob: geodataframe, of polygons and attribute use for clustering
        LAYER_POPULATION
        Column CLUSTER_COLUM_NAME used to clustering process.
    df_prov: geodataframe, of entities over municipalities.
        LAYER_PROVINCES
    name_unit: str, name of unit to select
    k: int, number of clusters defined in clustering process.

    Return 
    ----------
    type:
        None
    The function create a layer with clusters over 
    provinces defined in ls_prov. Layer name:
        LAYER_CLUSTERS
    """
    ls_dfg = []
    for prov in ls_prov:
        prov_cluster = SpatialClustering(dfg=df_pob,
                                         df_prov=df_prov, 
                                         province=prov, 
                                         col_name=settings.CLUSTER_COLUM_NAME)
        prov_cluster.dissolve_clusters(k=k)
        dfg_prov = prov_cluster.get_dfg_joined_dissolved()
        prov_pol = get_province_row(df_prov, df_pob, 
                                    name_unit=prov, 
                                    ls_columns=list(dfg_prov.columns))
        ls_dfg.append(dfg_prov)
        ls_dfg.append(prov_pol)
    # Join df and persist
    dfg_clusters = pd.concat(ls_dfg)
    dfg_clusters.to_file(settings.POL_PROVINCES,
                         layer=settings.LAYER_CLUSTERS,
                         driver="GPKG")


if __name__ == "__main__":
    ls_prov = ['Madrid', 'Valen', 'Cantabria']
    n_clusters = 4
    df_pob = gpd.read_file(settings.POL_PROVINCES,
                           layer=settings.LAYER_POPULATION,
                           driver="GPKG")
    df_prov = gpd.read_file(settings.POL_PROVINCES,
                           layer=settings.LAYER_PROVINCES,
                           driver="GPKG")
    process_spatial_clustering(ls_prov, df_pob, df_prov, n_clusters)
    print("Finish")