## Different machine learning models for hair type classification


## To segment hair images using trained DeepLabv3 model
- segment_images: script to create a segmentation mask and segment the hair images
- my_transforms1: data augmentation methods
- predictor: script for the predictor which creates the segmentation mask
- config_inference: YAML file which contains configurations for inference
- model folder: contains files for the DeepLab model
- datagen_utils: misc functions for encoding and decoding segmentation maps

## To compare different ML models and dimensionality reduction methods
- compare_classification_models: script to compare different ML models and dimensionality reduction methods. The data is loaded here and methods are tested iteratively.
- dim_reduction: file which contains different dimensionality reduction and ML models. Both have been fine-tuned for the segmented Internet hair dataset

## To investigate using LDA with different classifiers
- lda_classifier: script which performs LDA. Different classifiers can then be used for hair type classification
- dim_reduction: file which contains different dimensionality reduction and ML models. Both have been fine-tuned for the segmented Internet hair dataset
