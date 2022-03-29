# Results - README

> This is a README file for the results of evaluating classifiers on the MITBIH and PTBDB datasets.

## Layout

A dataframe that contains information on training domain, dataset, and metrics is available as both CSV and xlsx files in the root level of this repo. For the others:

1. Metrics for each classifier trained on the MITBIH dataset and the PTBDB dataset as individual models are in the `results\initial-training` folder.
2. Metrics for each classifier trained on the MITBIH dataset and the PTBDB dataset as ensemble models are in the `results\ensemble-training` folder.
3. In addition to our efforts, we also tested autoML solutions on the MITBIH dataset and the PTBDB dataset from the [autogluon library](https://github.com/awslabs/autogluon). The results are in the `results\automl-baseline` folder.
4. All images, plots, etc. are in the `results\figures` folder.

## Trained Model Weights

Weights for the individually (i.e. not-ensemble) trained models are available in the `results\weights` folder.

- all `.h5` files should be used with keras api (`model.load_weights(file_path)`)
- models trained with the lightning-flash library are also available and specify this in their directory name. To use these checkpoints for prediction in lightning-flash, load them into a trainer object as specified in [this tutorial](https://lightning-flash.readthedocs.io/en/latest/reference/tabular_classification.html)

---
