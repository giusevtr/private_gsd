#!/bin/bash

python3 run_nist.py national2019 simple 1  N 3
python3 postprocess.py national2019 simple sync_data/national2019_simple/3/1.00/N/oneshot/sync_data_0.csv post_sync_data/sync_national2019_3_simple_1_N.csv

python3 run_nist.py national2019 simple 10 N 3
python3 postprocess.py national2019 simple sync_data/national2019_simple/3/10.00/N/oneshot/sync_data_0.csv post_sync_data/sync_national2019_3_simple_10_N.csv

python3 run_nist.py tx2019       simple 1  N 3
python3 postprocess.py tx2019 simple sync_data/tx2019_simple/3/1.00/N/oneshot/sync_data_0.csv  post_sync_data/sync_tx2019_3_simple_1_N.csv
python3 run_nist.py tx2019       simple 10 N 3
python3 postprocess.py tx2019 simple sync_data/tx2019_simple/3/10.00/N/oneshot/sync_data_0.csv post_sync_data/sync_tx2019_3_simple_10_N.csv


python3 run_nist.py ma2019       simple 1  N 3
python3 postprocess.py ma2019 simple sync_data/ma2019_simple/3/1.00/N/oneshot/sync_data_0.csv  post_sync_data/sync_ma2019_3_simple_1_N.csv
python3 run_nist.py ma2019       simple 10 N 3
python3 postprocess.py ma2019 simple sync_data/ma2019_simple/3/10.00/N/oneshot/sync_data_0.csv post_sync_data/sync_ma2019_3_simple_10_N.csv


