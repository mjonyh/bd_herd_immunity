#!/bin/sh

rm ~/Downloads/Global_Mobility_Report*
open https://www.google.com/covid19/mobility/
echo "download the csv file and place it in the root directory"
read ok

cp ~/Downloads/Global_Mobility_Report.csv data/mobility.csv
cat data/mobility.csv | grep country > data/test.csv && cat data/mobility.csv | grep Bangladesh >> data/test.csv

python3 compute.py 30
python3 compute.py 90

python3 rt.py 1
python3 rt.py 2

python3 final_plot.py
python3 owid_data.py
