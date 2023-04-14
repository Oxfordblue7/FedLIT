## run repeats
dataset='pubmed_diabetes'
datapath='./data/pubmed_diabetes/'
outpath_baseline='./outputs/pubmed_diabetes/baselines_dgl'
test_linktypes='0-1-2-3'
task='classification'
partition='dominant'
nClients=10
nfeature=200
nclass=3

num_round=100
local_epoch=1
nlayer=2
nhidden=64

num_iterEM=1

nlinktype=4 # for mGCN and Fed-mGCN baselines, nlinktype should be equal to the number of oracle link-type
lr=0.01
weight_decay=0.0005
dropout=0.5

foldk=(0)
#foldk=(0 1 2 3 4 5 6 7 8 9)

## for baselines
baseline=("FedGCN")
#baseline=("GCN" "cGCN" "mGCN" "FedGCN" "FedmGCN" "local_GCN")

for bl in ${baseline[@]}; do
  for k in ${foldk[@]}; do
    python -m src.trainers.baselines --baseline ${bl} --foldk ${k} --dataset ${dataset} --datapath ${datapath} --outpath ${outpath_baseline} --test_linktypes ${test_linktypes} --partition ${partition} --nfeature ${nfeature} --nclass ${nclass} --nlinktype ${nlinktype} --nClients ${nClients} --task ${task} --lr ${lr} --weight_decay ${weight_decay} --dropout ${dropout} --num_iterEM ${num_iterEM} --num_round ${num_round}
  done
done
