#{900..3000..300}
#  VLCS PACS OfficeHome DomainNet
# --model_dir results/resnet50/${dataset}_${method}_${env}
for dataset in PACS; do
for method in CondCAD; do
for steps in 3000; do
  python dissect_neurons.py \
                --steps $steps \
                --topk 20 \
                --layer_name featurizer.network.layer4 \
                --quantile 0.007 \
                --miniou 0.3 \
                --model_dir results/resnet50/${method}/${dataset}/0
done
done
done
