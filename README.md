# DeepSolar_timelapse
Use computer vision to construct the time-series solar PV installation dataset.

An image is first classfied into one of the three classes according to its resolution: high resolution (HR), low resolution (LR), and extreme blurred/out of distribution (OOD). The training and inference scripts are hp_search_ood_multilabels.py and predict_ood_multilabels.py.

A HR image is classified by a single-branch CNN into two classes: positive (containing solar PV) and negative (otherwise). The training and inference scripts are hp_search_HR.py and predict_HR.py.

A LR image is classified by a two-branch Siamese CNN into two classes: positive (containing solar PV) and negative (otherwise). The training and inference scripts are hp_search_LR_rgb.py and predict_LR_rgb.py.

An OOD image is not used for determining the PV installation year.
