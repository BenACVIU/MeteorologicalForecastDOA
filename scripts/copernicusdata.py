"""
Author: Benjamin Arroquia Cuadros
12/06/2023

1. Download data every month from 2016 to 2018
"""

import os
import settings
import cdsapi
        
# FUNCTIONS

def create_folders():
    ls_folders = [
        './data/download',
        './data/failures',
        './data/processed'
    ]
    for path in ls_folders:
        if not os.path.exists(path):
            os.mkdir(path)

        
def download_datagrib(client_cop, dc_download):
    prefix = dc_download['year']+dc_download['month']
    filename = f"{prefix}_{settings.COPERNICUS_CATALOGE}.grib"
    path_file = os.path.join(settings.DOWNLOAD_FOLDERS,filename)
    print(path_file)
    client_cop.retrieve(settings.COPERNICUS_CATALOGE,
                        dc_download,
                        path_file)

    
if __name__ == "__main__":    
    dc_vars = settings.DC_DOWNLOAD_COPERNICUS.copy()
    create_folders()
    c = cdsapi.Client()
    for year in range(2016, 2017):
        dc_vars['year'] = str(year)
        for month in range(8, 10):
            dc_vars['month'] = f"{month:02}"
            print(dc_vars['year'], dc_vars['month'])
            download_datagrib(client_cop=c, dc_download=dc_vars)
    
    print("--END--")
