for dataset in PACS; do
for method in SelfReg CondCAD; do
for steps in 3000; do
  python dissect_concepts.py \
                --steps $steps \
                --clustering kmeans \
                --layer_name featurizer.network.layer4\
                --quantile 0.007 \
                --miniou 0.3 \
                --model_dir results/resnet50/${method}/${dataset}/0
#                --model_dir results/resnet50/${dataset}_${method}_vs/hp_0_trial_0_env_0
done
done
done
