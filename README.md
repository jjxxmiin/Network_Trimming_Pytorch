# Network_Trimming_Pytorch
Implementation [network trimming](https://arxiv.org/abs/1607.03250) using pytorch

# ImageNet

```shell

- datasets
    - imagenet - train
               - val
               - valprep.sh
- Prune_QTorch
```

Download : [valprep.sh](https://github.com/pytorch/examples/tree/master/imagenet)

```shell script
./valprep.sh
```

## How to use

|compress rate|Conv 5-3|FC 6|
|------|---|---|
|1.00|512|4096|
|1.19|488|3477|
|1.45|451|2937|
|1.71|430|2479|
|1.96|420|2121|
|2.28|400|1787|
|2.59|390|1513|

### pruning

```shell script
python prune.py --data_path ../datasets/imagenet \
                --save_path ./apoz_prune_model.pth.tar \
                --apoz_path ./vgg_apoz_fc.pkl \
                --select_rate 0
```

- pruning layer : `Conv 5-3`, `FC 6`

### fine tune

```shell script
python finetune.py --data_path ../datasets/imagenet \
                   --save_path ./apoz_fine_tune_model.pth.tar \
                   --prune_path ./apoz_prune_model.pth.tar \
                   --batch_size 128 \
                   --epoch 5
```

## Benchmark

- prune

```shell script
0 : 488, 3477

Before Pruning

Acc@1: 71.59 
Acc@5: 90.38

After Pruning

Acc@1: 70.37
Acc@5: 89.76
```

- finetune

```shell script
Conv 5-3 : 512 -> 488
FC 6 : 4096 -> 3477

Before Fine tune

Acc@1: 70.37
Acc@5: 89.76

After Fine tune

Acc@1: 71.48
Acc@5: 90.26
```

## Reference
- [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250)
- [https://github.com/Eric-mingjie/rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning)
