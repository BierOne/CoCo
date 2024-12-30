#  VLCS PACS OfficeHome DomainNet TerraIncognita
#  ERM CoCo_SelfReg CoCo_CondCAD

method=$1
dataset=$2

for env in 0 1 2 3; do
      python3 -m domainbed.scripts.train_ours_coverage\
             --data_dir=/home/lyb/data/domainbed/\
             --algorithm $method\
             --dataset $dataset\
             --steps 5000\
             --resnet18 True\
             --checkpoint_freq 1000\
             --output_dir ./results/${dataset}_${method}_test/env_${env} \
             --hparams '{"use_c_weights":true}' \
             --clustering kmeans \
             --test_env $env
done
