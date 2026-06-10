# Project spatial data science

# Data processing

Data process workflow is defined as follow:

1. [copernicusdata.py](https://github.com/BenACVIU/MeteorologicalForecastDOA/scripts/copernicusdata.py). An example of download data from CERRA catalog of Copernicus CDS. Data is stored in download folder inside data folder (in parent path project).

2. [processdoa.py](https://github.com/BenACVIU/MeteorologicalForecastDOA/scripts/processdoa.py). Module to process DOA index from grib files downloaded from Copernicus.

3. [clusteringpop.py](https://github.com/BenACVIU/MeteorologicalForecastDOA/clusteringpop.py). Process to obtain clusters in provinces based on municipalities population.

4. [extractmeteo.py](https://github.com/BenACVIU/MeteorologicalForecastDOA/scripts/extractmeteo.py). Extract data from raster files using areal data (spatial clusters). nput data 

5. [dataintegration.py](https://github.com/BenACVIU/MeteorologicalForecastDOA/scripts/dataintegration.py). Join dataframes script and load into SQLite.

6. [ModelingBiomet.py](https://github.com/BenACVIU/MeteorologicalForecastDOA/scripts/modelingbiomet.py). Modeling data using machine learning algorithms and storing data in MLflow.

Settings module has variables to define paths and important information. This module is imported in every data process. 