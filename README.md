# DeepSolar++: Understanding Residential Solar Adoption Trajectories with Computer Vision and Technology Diffusion Model

A deep learning framework to detect solar PV installations from historical satellite/aerial images and predict the installation year of PV. The model is applied to different places across the U.S. for uncovering solar adoption trajectories across time. The heterogeneity in solar adoption trajectories is further analyzed from the perspective of technology diffusion model.

To use the code, please cite:

* Wang, Z., Arlt, M. L., Zanocco, C., Majumdar, A., & Rajagopal, R. (2022). DeepSolar++: Understanding Residential Solar Adoption Trajectories with Machine Learning and Technology Diffusion Model. To appear in Joule.

The operating system for developing this code repo is Ubuntu 16.04, but it should also be able to run in other environments. The Python version used for developing this code repo is Python 3.6.


## Install required packages

Run the following command line:

```
$ pip install -r requirements.txt
```

**Note**: multi-variate OLS and logit regressions are run by the R code blocks inserted in the Python Jupyter Notework `bass_model_parameter_regression.ipynb`. It is based on the `rpy2` package. `lmtest` and `sandwich` are required libraries for R (which may need to be installed). 
For further details about using R in Python notebook, see [this](https://stackoverflow.com/questions/39008069/r-and-python-in-one-jupyter-notebook).

## Download data and model checkpoints

Run the following command lines to download the ZIP files right under the code repo directory:

```
$ curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/DeepSolar2/checkpoint.zip
$ curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/DeepSolar2/data.zip
$ curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/DeepSolar2/results.zip
```

Unzip them such that the directory structure looks like:

```
DeepSolar_timelapse/checkpoint/...
DeepSolar_timelapse/data/...
DeepSolar_timelapse/results/...
```

**Note 1**: for the satellite/aerial imagery datasets under `data` directory (subdirectory `HR_images` for high-resolution (HR) images, `LR_images` for low-resolution (LR) images,  `blur_detection_images` for blur detection images, and `sequences` for image sequences), due to the restriction of imagery data sources, we are not able to publicly share the full data. Instead, for each subset (training/validation/test) and each class (e.g., positive/negative), we share two example images as a demo. For the image sequence dataset (`sequences`), we share one demo sequence (`sequences/demo_sequences/1`). Each image sequence contains satellite/aerial images captured in different years at the same location of a solar installation (image file name examples: `2006_0.png`, `2007_0.png`, `2007_1.png`, `2008_0.png`, etc). Users can put their own data under these directories.

**Note 2**: to run Jupyter Notebook, the default kernel/environment is "conda_tensorflow_p36", which does not necessarily exist in your computer. Please change the kernel to the one where all required packages are installed.

## Functionality of each script/notebook

### Part 1: model training with hyperparameter search

An image is first classfied by the blur detection model into one of the three classes according to its resolution: high resolution (HR), low resolution (LR), and extreme blurred/out of distribution (OOD). An OOD image is not used for determining the solar installation year; a HR image is classified by a single-branch CNN into two classes: positive (containing solar PV) and negative (otherwise); a LR image is classified by a two-branch Siamese CNN into two classes: positive (containing solar PV) and negative (otherwise). 

For training the blur detection model with hyperparameter search:
```
$ python hp_search_ood_multilabels.py
```
For training the HR model with hyperparameter search:
```
$ python hp_search_HR.py
```
For training the LR model with hyperparameter search:
```
$ python hp_search_LR_rgb.py
```

By default, all three scripts above are run on a machine with GPU.

### Part 2: deploying models to predict installation year

For each solar installation, we can retrieve a sequence of images captured in different years at its location and put them in the same folder. The images are named as `{image_capture_year}_{auxiliary_index}.png`. For example, if there are three images captured in 2012, they are named as `2012_0.png`, `2012_1.png`, and `2012_2.png`, respectively. 

For each image sequence, we deploy the blur detection model, HR model, and LR model. Their model outputs are combined together predict the installation year of the solar PV system.

First, we deploy the blur detection model and HR model to image sequences:
```
$ python predict_HR.py
```
```
$ python predict_ood_multilabels.py
```

Combining the prediction results of the above two models, we can generate the "anchor_images_dict" that maps a target image in a sequence to all its reference images in this sequence. This needs to be run before deploying the LR model, as the LR model needs to take a target image and one of its corresponding reference image as inputs. To do this, run the code blocks in this Jupyter Notebook:

```
generate_anchor_image_dict.ipynb
```

Then, deploy the LR model to image sequences:
```
$ python predict_LR_rgb.py
```

By default, all three `.py` scripts above are run on a machine with GPU.

Finally, run the code blocks in this Jupyter Notebook that combines all model prediction outputs to predict the installation year for each solar PV system:

```
predict_installation_year_from_image_sequences.ipynb
```

### Part 3: analyzing solar adoption trajectories across time using Bass model

By predicting the installation year of each system, we are able to obtain the number of installations in each year in each place. We provide such solar installation time-series dataframe at the census block group level covering the randomly-sampled 420 counties in ``results``. This dataframe is used as inputs for the following analysis. 

For the solar adoption trajectory in each block group, we use a classical technology adoption model, [Bass model](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.15.5.215?casa_token=PXhDNyJRVhgAAAAA:ZbFnu9tpKcAJoUDE6JlpMyWvaaa0hyXeuFA2Edbg8EORBlPTUVHBWShq6c1yuA5SBaPBRyLCW1Q), to parameterize its shape. First, the Bass model fitting based on Non-linear least squares (NLS):

```
$ python bass_model_fit_adoption_curves.py
```

Then we can use two Jupyter Notebook to analyze the Bass model parameters that have been fitted. They can be run without running the Bass model fitting code as the intermediate result of model fitting has already been provided in `results`:

`bass_model_parameter_phase_analysis.ipynb`: Based on the fitted Bass model parameters, segment each solar adoption trajectory into four phases: pre-diffusion, ramp-up, ramp-down, and saturation, and analyze the fractions of block groups in each phase.

`bass_model_parameter_regression.ipynb`: Run multivariate regressions with various Bass model parameters as dependent variables and socioconomic characteristics (demographics, PV benefit, the presence of incentives, etc) as independent variables. R code blocks are inserted for running these regressions.

