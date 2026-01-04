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
data
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

promote
├── boxes
├── feature_boxes_flat
└── sip_features [Tensor]
```


