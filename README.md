# MDR-GCN
Multi-Dimensional Refinement Graph Convolutional Network with Robust Decouple Loss for Fine-Grained Skeleton-Based Action Recognition

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

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{liu2023multidimensional,
      title={Multi-Dimensional Refinement Graph Convolutional Network with Robust Decouple Loss for Fine-Grained Skeleton-Based Action Recognition}, 
      author={Sheng-Lan Liu and Yu-Ning Ding and Jin-Rong Zhang and Kai-Yuan Liu and Si-Fan Zhang and Fei-Long Wang and Gao Huang},
      year={2023},
      eprint={2306.15321},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


