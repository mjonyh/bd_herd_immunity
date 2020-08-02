#!/bin/sh

open https://www.google.com/covid19/mobility/
echo "download the csv file and place it in the root directory"
read ok

python3 compute.py 30
python3 compute.py 90
python3 rt.py 1
python3 rt.py 2
python3 final_plot.py
