## run repeats
dataset='pubmed_diabetes'
datapath='./data/pubmed_diabetes/'
outpath='./outputs/pubmed_diabetes/'
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

nlinktype=4
lr=0.01
weight_decay=0.0005
dropout=0.5

foldk=(0 1 2 3 4 5 6 7 8 9)

for k in ${foldk[@]}; do
  python src/trainers/FedLIT_trainer.py --foldk ${k} --dataset ${dataset} --datapath ${datapath} --outpath ${outpath} --test_linktypes ${test_linktypes} --partition ${partition} --nfeature ${nfeature} --nclass ${nclass} --nlinktype ${nlinktype} --nClients ${nClients} --task ${task} --lr ${lr} --weight_decay ${weight_decay} --dropout ${dropout} --num_iterEM ${num_iterEM} --num_round ${num_round}
done
