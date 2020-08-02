#!/bin/sh

python3 compute.py 30
python3 compute.py 90
python3 rt.py 1
python3 rt.py 2
python3 final_plot.py
