Precision Ag
====

This pipeline uses the IBM Watson Visual Recognition API to identify plants that have been subject to water deprivation. The pipeline uses drone images of crop fields taken in various spectra.

> Drone image taken in ultraviolet:
![Image1](https://github.com/danxfreeman/precision_ag/blob/master/IMG_7578.JPG)

> Drone image taken in near-infrared:
![Image2](https://github.com/danxfreeman/precision_ag/blob/master/2017_0810_145135_118.JPG)

Here is the basic workflow:

* Upload images to Labelbox and draw bounding boxes
* Get bounding box coordinates and crop images with `work_data.py`
* Group images into test and training sets with `index.R`
* Train, test, and assess models with `predict.py`
* Interpret results

## Setup

Install the Watson API package by entering the command `pip install --upgrade ibm-watson` in Terminal

Download the skeleton directory `precision_ag`

Move original drone images to the subdirectory `Images`

## Crop Images in Labelbox

> Summary...

1.) Set up Labelbox...

> Each image of the same plot must have the same orientation. Rotate image if necessary.

2.) Draw bounding boxes...

> Because subsequent functions perform 1:1 matching, it’s important that each image from the same dataset contains the same number of plants in each condition (for example, all MAPIR FLT1 images have exactly five Buddleia plants labelled high water stress). This means that if part of the plot is cut off in one image, those plants should not be labelled in any other image.

3.) Use the functions `function1` and `function2` to ensure that plots are correctly labelled before moving on.

## Pull Cropped Images with the Labelbox API

> Summary...

1.) Pull coordinates from Labelbox...

2.) Crop images...

## Split Cropped Images

> Because the drone takes multiple images of the same plot, it’s important that cropped images of the same plant don’t end up in the test set *and* the training set. `index.R` loops through each image of the same plot and assigns a numeric id to each plant based on its location. This way, for example, all images of plants 1 to 6 end up in the training set and all images of plants 7 and 8 end up in the test set.

1). If neccessary, download required libraries by entering the commands `install.packages("spatstat")` and `install.packages("tidyverse")` into the console.

2). Update the following arguments in `index.R` and run.

* coord_path: path to csv file created by `work_data.py`
* image_path: path to directory containing cropped images (may contain subdirectories)
* to_path: path to directory where split images will be saved (must already exist)

> Here is the basic workflow of `index.R`:
> * Calculate the center of each bounding box (`dat$x_center` and `dat$y_center`).
> * Subset combinations of dataset, species, and condition (`dat$model`) and loop through each image (`dat$image`).
> * Normalize each axis to a 0 to 1 range (`image$x_norm` and `image$y_norm`).
> * Perform one-to-one matching with a reference image and assign numeric id (`image$position`).
> * Match each row in `coord_path` with each image in `image_path` using the column `df$id` and the last number in the image name (e.g. `*_123.JPG`).
> * Group images into four-fold training and test sets and zip.

You can also modify data by manipulating the object `dat`. For example, you can delete rows corresponding to a certain dataset or pool different stress conditions into one. See `# Modify data` for examples.

## Model

> The Watson API bills your account for every run of `pipeline_train` so ensure that previous steps were successful before moving on. The following checkpoints can help:
> * `function2` confirms that each image has the same number of plants in each condition
> * Each cropped image appears exactly once within the directory `Split`
> * Cropped image names match directory names
> * Training and test sets contain multiple images of the same plant (you can sometimes tell by looking at the plant's shape)

1.) Access cloud using an [IAM key](https://cloud.ibm.com/docs/services/watson?topic=watson-iam).

```python
visual_recognition = VisualRecognitionV3(
    version = '2018-03-19',
    iam_apikey = '...'
)
```

2.) Train model using `modelID = pipeline_train(...)`.

* model_name: what to name the model (e.g. `'MAPIR_hq_HWS_k1'`)
* parent: working directory (defaults to current)
* stress_train, ns_train: paths to zip files containing stress and no_stress training set images
* log: path to csv file to which model information will be appended (defaults to `'Results/Log.csv'`)

> `Log.csv` automatically updates itself on every run of `pipeline_train` and documents the image sets used to train each mode.

3.) Save modelID in comments (this is different from `model_name`).

4.) Wait for model to train. The function `wait(modelID)` pings the cloud every 30 seconds until the model is ready (optional).

> Make sure you have the right modelID before proceeding, especially if you're working with multiple models.

5.) Test model with `pred = pipeline_test(...)`.

* modelID: modelID returned by `pipeline_train`
* stress_test, ns_test: paths to zip files containing stress and no_stress test set images
* pred_save: path to directory where image-level predictions will be saved
* trained: path to directory to which images will be moved to after use (set `trained=''` if you do not want to move images)

> Moving training and test sets after use ensures that they aren't accidentally re-used.

6). Assess model performance with `perf = pipeline_assess(...)`.

* pred: image-level predictions returned by `pipeline_test`
* stress_test, ns_test: paths to zip files containing stress and no_stress test set images
* perf_save: path to csv file to which model performance metrics will be appended (defaults to `'Results/Performance.csv'`)

> `Performance.csv` automatically updates itself on every run of `pipeline_assess` and documents performance metrics and test set paths. Review `Log.csv`and `Performance.csv` to ensure that each model was trained and tested with the correct image sets.

## Interpret Results

The [AUC (Area Under The Curve) ROC (Receiver Operating Characteristics) curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) is most direct measure of how well the model distinguishes between classes. An AUC of 0 means that the model mis-classified every image. When AUC is 0.5, model performance is equal to random chance. An excellect model has an AUC approaching 1.

Four-fold cross-validation helps assess generalizability. By calculating the standard deviation in ROC-AUC between the four folds, we can determine the extent to which performance metrics are sensitive to variations in data.

High ROC-AUCs and low standard deviations indicate that models demonstrate high seperability and high generalizability. 
