#!/bin/bash

mkdir "post_sync_data"
python3 postprocess.py national2019 simple sync_data/national2019_simple/1.00/oneshot/sync_data_0.csv post_sync_data/sync_national2019_simple_1.csv
python3 postprocess.py national2019 simple sync_data/national2019_simple/10.00/oneshot/sync_data_0.csv post_sync_data/sync_national2019_simple_10.csv

python3 postprocess.py tx2019 simple sync_data/tx2019_simple/1.00/oneshot/sync_data_0.csv  post_sync_data/sync_tx2019_simple_1.csv
python3 postprocess.py tx2019 simple sync_data/tx2019_simple/10.00/oneshot/sync_data_0.csv post_sync_data/sync_tx2019_simple_10.csv
python3 postprocess.py ma2019 simple sync_data/ma2019_simple/1.00/oneshot/sync_data_0.csv  post_sync_data/sync_ma2019_simple_1.csv
python3 postprocess.py ma2019 simple sync_data/ma2019_simple/10.00/oneshot/sync_data_0.csv post_sync_data/sync_ma2019_simple_10.csv
