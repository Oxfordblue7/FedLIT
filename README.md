# FedLIT

This is the official implementation for the paper "[Federated Node Classification over Graphs with
Latent Link-type Heterogeneity](https://doi.org/10.1145/3543507.3583471)"

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Data

The dataset ```data/pubmed_diabetes/``` contains a global graph ```graph_oracle_linkType0-1-2-3.bin``` and 
its train/val/test splits for 10-fold cross-validation, and local graphs with different partitioning ways 
(```graphs_oracle_distinct_numClient10/```, ```graphs_oracle_oneDominant_numClient10/```, ```graphs_oracle_balanced_numClient10/```).

## Run FedLIT

To run __FedLIT__, use the scripts below:

```bash
bash run_FedLIT_trainer.sh
```
or
```bash
python -m src.trainers.FedLIT_trainer --foldk ${k} --dataset ${dataset} --datapath ${datapath} --outpath ${outpath} --test_linktypes ${test_linktypes} --partition ${partition} --nfeature ${nfeature} --nclass ${nclass} --nlinktype ${nlinktype} --nClients ${nClients} --num_round ${num_round}
```

## Run Baselines

To run the six baselines that are implemented in our paper, use the scripts below:

```bash
bash run_baselines.sh
```
or
```bash
python -m src.trainers.baselines --baseline ${bl} --foldk ${k} --dataset ${dataset} --datapath ${datapath} --outpath ${outpath_baseline} --test_linktypes ${test_linktypes} --partition ${partition} --nfeature ${nfeature} --nclass ${nclass} --nlinktype ${nlinktype} --nClients ${nClients} --num_round ${num_round}
```

## Outputs

The outputs of __FedLIT__ will be saved as 
```bash
outputs/${dataset}/${test_linktypes}/${partition}/${foldk}_result_local.csv
outputs/${dataset}/${test_linktypes}/${partition}/${foldk}_result_global.csv
```

## If you find this work helpful, please cite
```
@inproceedings{10.1145/3543507.3583471,
      title={Federated Node Classification over Graphs with Latent Link-type Heterogeneity}, 
      author={Xie, Han and Xiong, Li and Yang, Carl},
      booktitle={Proceedings of the ACM Web Conference 2023},
      year={2023},
      url={https://doi.org/10.1145/3543507.3583471},
      doi={10.1145/3543507.3583471},
      series={WWW '23}
}
```