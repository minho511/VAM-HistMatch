# VAM-HistMatch
Official PyTorch implementation of Value Histogram Matching module (VAM) proposed in 

> Cross-Domain Person Re-Identification Using Value Distribution Alignment  
> Minho Kim, Yeejin Lee  
> ICCAS 2023  

[Paper (IEEE Xplore)](https://ieeexplore.ieee.org/document/10316745)

This repository provides a simplified reimplementation of the original Value Alignment Module using a standalone `torchvision.transforms` style interface.
While this version is significantly easier to integrate and apply in existing PyTorch pipelines, the performance may slightly differ from that reported in the original paper due to differences in implementation details.

### Motivation

---

Cross-domain person re-identification often suffers from significant performance degradation due to domain gaps especially illumination differences.

> "What if we transform the brightness of all input images during both training and testing to follow a common reference distribution? Would the model then become more robust to illumination variation?"

-> My approach: Align the value (brightness) distribution of HSV images to a reference Gaussian distribution using histogram matching.

### Usage

---

This method can be easily integrated into the popular [reid-strong-baseline repository](https://github.com/michuanhaohao/reid-strong-baseline).

To apply the value histogram alignment transform, simply modify the dataset transform code as follows:

```python
    from value_align import ValueHistogramAlign

    MEAN, STD = 125, 60 # hyperparam

    def build_transforms(cfg, is_train=True):
        normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        
        if is_train:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                ValueHistogramAlign(mean=MEAN, std=STD), # --> value_align
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TEST),
                ValueHistogramAlign(mean=MEAN, std=STD), # --> value_align
                T.ToTensor(),
                normalize_transform
            ])

        return transform
```

### Additional Remarks

1. Performance in the original domain may slightly decrease

    Since this method transforms the brightness distribution of all input images, it may slightly harm performance when training and testing within the same domain. However, it significantly improves generalization to new domains with unseen illumination conditions making it ideal for cross-domain person re-identification.

2. Potential for broader applicability

    While this method is designed for cross-domain person re-ID, its core principle (aligning brightness distributions) is general. We expect it could also be beneficial for other vision tasks where illumination variation is a major challenge.
