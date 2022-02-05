# Temporal Alignment Prediction for Supervised Representation Learning and Few-Shot Sequence Classification

Pytorch implementation of using TAP for few-shot action recognition.

# Acknowledgments
We based our code on [Few Shot Action Recognition Library] which is publicly available at https://github.com/tobyperrett/few-shot-action-recognition.
Please also check the license and usage there if you want to make use of this code. 


# Usage
Please check the README file in https://github.com/tobyperrett/few-shot-action-recognition. Here we quote and slightly adapt a part of the descriptions for quick start and running as follows:
###############################################################################

### Datasets supported

- Something-Something V2 ([splits from OTAM](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf))
- UCF101 ([splits from ARN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf))
- HMDB51 ([splits from ARN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf))


### Requirements

- python >= 3.6
- pytorch >= 1.8
- einops
- ffmpeg (for extracting data)
- soft_dtw (https://github.com/Sleepwalking/pytorch-softdtw)


### Data preparation

Download the datasets from their original locations:

- [Something-Something V2](https://20bn.com/datasets/something-something#download)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

Once you've downloaded the datasets, you can use the extract scripts to extract frames and put them in train/val/test folders. You'll need to modify the paths at the top of the scripts.
To remove unnecessary frames and save space (e.g. just leave 8 uniformly sampled frames), you can use shrink_dataset.py. Again, modify the paths at the top of the sctipt.


### Running examples

Training TAP-CNN on the UCF101 dataset, 5-way 1-shot:
```bash
python runfull.py -c ./checkpoints/ucfoap --dataset ./data/ucf101_l8 --method oap --shot 1 --tasks_per_batch 1 -i 35000 --val_iters 5000 10000 15000 20000 25000 30000 35000
```

Training SA-TAP-CNN on the UCF101 dataset, 5-way 5-shot:
```bash
python runfull.py -c ./checkpoints/ucfsaoap5shot --dataset ./data/ucf101_l8 --method saoap --shot 5 --tasks_per_batch 1 -i 35000 --val_iters 5000 10000 15000 20000 25000 30000 35000
```

Training TAP-CNN on the SSv2 dataset, 5-way 5-shot:
```bash
python runfull.py -c ./checkpoints/ssv2oap5shot --dataset ./data/ssv2_l8 --method oap --shot 5 --tasks_per_batch 1 -i 50000 --val_iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000
```

For training TAP-TL in any case, please in uncomment line 587 and comment line 588 in "model.py" (the "__init__" function of class "CNN_OAP").