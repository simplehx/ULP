# ULP: Unlabeled Location Prediction from Text
With the popularity of smart mobile devices, location-based services (LBS) have been widely applied. Predicting geographical locations from text holds significant value for smart cities and personalized travel. Existing research primarily focuses on the retrieval or prediction of labeled locations, such as cities or points of interest (POIs). However, in scenarios like autonomous driving navigation and autonomous logistics delivery, it is necessary to precisely predict the coordinates of unlabeled locations, for example, 200 meters northwest of a certain location. Consequently, we introduce a new task to infer fine-grained unlabeled locations from text. This task is particularly challenging because of the ambiguous text and the semantic gap between geographic and textual modalities. In this paper, we aim to construct an end-to-end fine-grained location prediction model to accurately predict the unlabeled locations mentioned in texts. First, we encode the geographic coordinates and transform the location prediction problem into a geographic encoding generation problem. Second, we propose a multi-scale cross-modal loss (MCL) to learn the implicit mapping between geographic and textual modalities. Lastly, we design a multi-task prediction model ULP to predict the coordinates of unlabeled locations. We conducted experiments on two real-world datasets, and the results show that our proposed method outperforms existing state-of-the-art retrieval-based methods.

![The overall architecture of ULP](./img/img.png "The overall architecture of ULP")


## Requirements
* python == 3.10
* numpy == 1.26.4
* pandas == 2.2.1
* pytorch == 2.4.1
* scikit-learn == 1.4.1
* tqdm == 4.66.2
* st-moe-pytorch == 0.1.8
* transformers == 4.39.1
* python-geohash == 0.8.5
* haversine == 2.8.1


## Datasets
The original dataset can be downloaded from:
* GeoGLUE: https://modelscope.cn/datasets/iic/GeoGLUE

## Reference
```
@inproceedings{he2025ulp,
  title={ULP: Unlabeled Location Prediction from Text},
  author={He, Xi and Liu, Yilin and Sun, Yijie and Xing, Xin and Lu, Xingyu and Liu, Yanbing},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={266--275},
  year={2025}
}
```

<!-- Our paper has been accepted by SIGIR 2025, and the complete code will be organized and released soon. -->