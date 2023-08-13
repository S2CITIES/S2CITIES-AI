python train_SFH_3dcnn.py \
    --exp mn2-dataset-no-temp-trans-norm255-size112-complete \
    --model mobilenetv2 \
    --sample_size 112 \
    --output_features 1 \
    --norm_value 255 \
    --recompute_mean_std

python train_SFH_3dcnn.py \
    --exp mn2-dataset-no-temp-trans-no-norm255-size112-complete \
    --model mobilenetv2 \
    --sample_size 112 \
    --output_features 1 \
    --norm_value 255 \
    --no_norm \

python train_SFH_3dcnn.py \
    --exp mn2-dataset-no-temp-trans-norm255-size112-nesterov-complete \
    --model mobilenetv2 \
    --sample_size 112 \
    --output_features 1 \
    --norm_value 255 \
    --nesterov 
