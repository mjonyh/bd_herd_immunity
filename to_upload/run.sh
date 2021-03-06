#!/bin/sh

# download the mobility data from google and process for Bangladesh
wget https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip
unzip Region_Mobility_Report_CSVs.zip
cp 2020_BD_Region_Mobility_Report.csv data/test.csv
rm 2020_*
rm Region_Mobility_Report_CSVs.zip

### Calculate the SIRD model
python3 compute.py 30
python3 compute.py 90

### Calculate Rt
python3 rt.py 1
python3 rt.py 2

### Plot the data
python3 google_mobility.py
python3 owid_data.py
