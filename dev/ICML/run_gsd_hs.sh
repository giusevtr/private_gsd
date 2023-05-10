#!/bin/bash
YEAR=2018
ROUNDS=50
SAMPLES=10
SAVE_PATH=icml_results/singletask/gsd_ada_hs_${ROUNDS}_${SAMPLES}.csv
for STATE in CA
do
    for TASK in coverage mobility employment income travel
    do
      for EPS in 0.07 0.23 0.52 0.74 1.00
      do
        for SEED in 0 1 2
        do
          DATASET=folktables_${YEAR}_${TASK}_${STATE}
          echo $DATASET
          echo  $EPS
          echo $SEED
          python gsd_adaptive_hs.py $SAVE_PATH $DATASET $ROUNDS $SAMPLES  $EPS  $SEED
        done
      done
    done
done
