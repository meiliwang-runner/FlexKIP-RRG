# FlexKIP-RRG
Official implementation of ClinKI-RRG framework for radiology report generation

## Usage
## Environment Setup

```bash
conda env create -f environment.yml
conda activate flexkip
```


## Dataset

We conduct experiments on the MIMIC-CXR dataset, a large-scale chest X-ray dataset
released by the MIT Laboratory for Computational Physiology.
Due to data usage restrictions, we do not redistribute the dataset.
Researchers must obtain credentialed access via PhysioNet.
Dataset URL:
https://physionet.org/content/mimic-cxr-jpg/
After downloading the data, users should organize the files according to the
structure described in this repository.
```
data/
├── images/                 
├── data.json                
├── promote
    ├── boxes
    ├── feature_boxes_flat
    └── sip_features

data.json
[
  train:{
    "id": "uuid-string",
    "study_id": xxxxxxxx,
    "subject_id": xxxxxxxx,
    "report": "There is no ...",
    "image_path": [
      "xxx/xxxx/xxxx/xxxx.jpg"
    ],
    "split": "train"
  }
...
]
```

## Acknowledgement
Thanks PairAug, CXPMRG-Bench for serving as building blocks of FlexKIP-RRG.
```
@inproceedings{xie2024pairaug,
  title={PairAug: What Can Augmented Image-Text Pairs Do for Radiology?},
  author={Xie, Yutong and Chen, Qi and Wang, Sinuo and To, Minh-Son and Lee, Iris and Khoo, Ee Win and Hendy, Kerolos and Koh, Daniel and Xia, Yong and Wu, Qi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  pages={4553--4563}
}

@inproceedings{wang2025cxpmrg,
  title={CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset},
  author={Wang, Xiao and Wang, Fuling and Li, Yuehang and Ma, Qingchuan and Wang, Shiao and Jiang, Bo and Tang, Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
