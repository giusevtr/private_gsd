#!/bin/bash


python3 postprocess.py national2019 all sync_data/national2019_all/1.00/oneshot/sync_data_0.csv sync_national2019_all_1.csv
python3 postprocess.py national2019 all sync_data/national2019_all/10.00/oneshot/sync_data_0.csv sync_national2019_all_10.csv
python3 postprocess.py all tx2019 sync_data/tx2019_all/1.00/oneshot/sync_data_0.csv  sync_tx2019_all_1.csv
python3 postprocess.py all tx2019 sync_data/tx2019_all/10.00/oneshot/sync_data_0.csv sync_tx2019_all_10.csv
python3 postprocess.py all ma2019 sync_data/ma2019_all/1.00/oneshot/sync_data_0.csv  sync_ma2019_all_1.csv
python3 postprocess.py all ma2019 sync_data/ma2019_all/10.00/oneshot/sync_data_0.csv sync_ma2019_all_10.csv
