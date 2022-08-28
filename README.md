# MDR-GCN
Multi-Dimensional Refinement Graph Convolutional Network with Robust Decouple Loss for Fine-Grained Skeleton-Based Action Recognition

## Results

By updating parameters and using new devices, our model achieved better peformance on NTU RGB+D than we noted before (92.4 on Xsub in our paper)

| Datasets         | Top-1 |
|------------------|-------|
| FineGym99        | 92.81 |
| FSD-10           | 93.88 |
| NTU RGB+D (XSub) | 92.59 |
| NTU RGB+D (XView) | 96.87 |

The accuracies of four modalities on NTU RGB+D (Xsub):

| Modalities   | Top-1 |
|--------------|-------|
| joint        | 89.66 |
| bone         | 90.31 |
| joint motion | 88.12 |
| bone motion  | 87.68 |

The accuracies of four modalities on NTU RGB+D (Xview):

| Modalities   | Top-1 |
|--------------|-------|
| joint        | 95.21 |
| bone         | 95.31 |
| joint motion | 93.94 |
| bone motion  | 92.53 |

## Datasets

### FineGym99

The data processing is borrowed from https://github.com/kennymckormick/pyskl

### NTU RGB+D

Request dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition

The data processing is borrowed from CTR-GCN: https://github.com/Uason-Chen/CTR-GCN

### FSD-10

Request dataset: https://shenglanliu.github.io/fsd10/

## Train

For FineGym99,
```shell
python main_cl_new.py --config config/gym/default2.yaml --device 0
```

For NTU RGB+D,
```shell
python main_cl_new.py --config config/nturgbd-cross-subject/default.yaml --device 0
```

After all modalities, ensemble the results of different modalities, run
```shell
python ensemble.py --dataset ntu/xsub --joint-dir work_dir/ntu60/xsub/FTS_joint --bone-dir work_dir/ntu60/xsub/FTS_bone --joint-motion-dir work_dir/ntu60/xsub/FTS_jointM --bone-motion-dir work_dir/ntu60/xsub/FTS_BoneM
```

For FSD-10,
```shell
python main_cl_new.py --config config/skat/train_25.yaml --device 0
```


