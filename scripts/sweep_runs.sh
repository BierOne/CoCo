#  VLCS PACS OfficeHome DomainNet TerraIncognita

# CoCo_CondCAD CoCo_SelfReg
# delete_incomplete launch
Method=$1
Dataset=$2


python -m domainbed.scripts.sweep_ours delete_and_launch\
        --data_dir=/home/lyb/data/domainbed/\
        --output_dir=./results/resnet50/${Dataset}_${Method}_sweep\
        --command_launcher multi_gpu\
        --algorithms $Method\
        --datasets $Dataset \
        --steps 5001\
        --single_test_envs\
        --hparams '{"use_c_weights":true}' \
        --skip_confirmation


