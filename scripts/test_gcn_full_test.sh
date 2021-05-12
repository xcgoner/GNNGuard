#!/bin/bash

export OMP_NUM_THREADS=44
export KMP_AFFINITY=granularity=fine,compact,1,0;

# mean
for seed in 0 1 3 5 7 9
do
    python -W ignore ../RobustAggregation/test_no_attack.py --dataset cora  --modelname GCN_V2 --seed ${seed} --aggr mean --log full_test_cora_gcn_mean --save model2
done


for b in 2 3 4 5
do
    for ratio in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45
    do
        for seed in 0 1 3 5 7 9
        do
            # without compensate
            python -W ignore ../RobustAggregation/test_no_attack.py --dataset cora  --modelname GCN_V2 --seed ${seed} \
                    --aggr median --n-neigh-threshold ${b} --trim-ratio ${ratio} --log full_test_cora_gcn_median_nocompensate --save model2
        done

        for seed in 0 1 3 5 7 9
        do
            # with compensate
            python -W ignore ../RobustAggregation/test_no_attack.py --dataset cora  --modelname GCN_V2 --seed ${seed} \
                    --aggr median --n-neigh-threshold ${b} --trim-ratio ${ratio} --trim-compensate --log full_test_cora_gcn_median_compensate --save model2
        done
    done
done

