# UNet segmentation using pytorch_lightning
This is a Unet implementation using [pytorch_lightning](https://pytorch-lightning.readthedocs.io/en/latest/).  
By default, the [CVC_Clinic Polyp dataset](https://www.kaggle.com/datasets/balraj98/cvcclinicdb?resource=download) is used in this template.

## Run training
```shell
python lightning_train.py \
        --max_epochs 5 \
        --devices 2,3 \
        --strategy "dp" \
        --accelerator "gpu" \
        --batch_size 4 \
        --num_workers 3 \
        --lr 1e-4 \
        --es_patience 5 \
        --es_mode "min" \
        --es_monitor "val_loss" 
```