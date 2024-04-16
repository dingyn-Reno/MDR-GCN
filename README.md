# MDR-GCN
Multi-Dimensional Refinement Graph Convolutional Network with Robust Decouple Loss for Fine-Grained Skeleton-Based Action Recognition

（2024.3.26:Accepted by IEEE Transactions on Neural Networks and Learning Systems,TNNLS)

pdf：https://arxiv.org/abs/2306.15321

![image text](https://github.com/dingyn-Reno/MMFS/blob/main/MDRGCN.png)

## Datasets

### FineGym99

The data processing is motivated by https://github.com/kennymckormick/pyskl

### NTU RGB+D

Request dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition

The data processing is motivated by CTR-GCN: https://github.com/Uason-Chen/CTR-GCN

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
@ARTICLE{10499829,
  author={Liu, Sheng-Lan and Ding, Yu-Ning and Zhang, Jin-Rong and Liu, Kai-Yuan and Zhang, Si-Fan and Wang, Fei-Long and Huang, Gao},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Multidimensional Refinement Graph Convolutional Network With Robust Decouple Loss for Fine-Grained Skeleton-Based Action Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Task analysis;Convolution;Skeleton;Robustness;Feature extraction;Topology;Convolutional neural networks;Fine-grained action;graph convolutional network (GCN);robust decouple loss (RDL);spatial–temporal attention},
  doi={10.1109/TNNLS.2024.3384770}}
```


