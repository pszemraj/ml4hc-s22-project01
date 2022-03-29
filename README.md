# Machine Learning for Healthcare - Problem 1

Lou Ancillon & Peter Szemraj

## about

this repo contains the code for the first problem of the Machine Learning for Healthcare course. The majority of development is done in Google Colab for the GPU compute power, the _NOTE: access required_ Drive folder is [here](https://drive.google.com/drive/folders/1vZYgrdunRwJBmDzBnQnolPrN9BoagP54?usp=sharing).

This repo primarily aims as a central home for the data processing piece, EDA and analysis, as well as the model training and evaluation.

## Results - Preview

### MITBIH dataset

Here, we compare accuracy of several different models, trained via standard keras, autoML, and ensemble methods on the MITBIH dataset.

<!-- display the HTML plot at local path results\figures\model-analysis\mitbih_accuracy_comparison.html inline -->
<iframe src="results/figures/model-analysis/mitbih_accuracy_comparison.html" width="100%" height="500px"></iframe>

- an interactive version of our model results on MIT-BIH (tooltips, etc.) is available as an [app on Netlify](https://mitbih-pred-acc.netlify.app/)

### PTBDB dataset

Here, we compare the ROC AUC score of several different models, trained via standard keras, autoML, and ensemble methods on the PTBDB dataset.

<!-- display the HTML plot at local path results\figures\model-analysis\ptbdb_roc_auc_comparison.html inline -->
<iframe src="results/figures/model-analysis/ptbdb_roc_auc_comparison.html" width="100%" height="500px"></iframe>

- an interactive version of our model results on PTBDB (tooltips, etc.) is available as an [app on Netlify](https://ptbdb-pred-rocauc.netlify.app/)

## Installation

- for using the repo locally (EDA, analysis) the primary method to install is clone + pip install.
- **NOTE: training notebooks are meant to be run on Google Colab, and the installation of those packages is handled separately inside the notebooks.**

### Steps to install for local use

1. clone the repo `git clone https://github.com/pszemraj/ml4hc-s22-project01.git`
2. set working directory to the root of the repo (`cd ml4hc-s22-project01`)
3. Install required packages with `pip install -r requirements.txt`
4. now can run scripts with `python main.py`

## data

Please make sure you follow the steps in install section to install the required packages first. Data for the project is available in several different ways, and was used in both unaltered as well as cleaned/rearranged ways.

- modified versions of the original data and reformatted pytorch data are available in the repo by default as [Apache feather format](https://arrow.apache.org/docs/python/pandas.html) files to stay under git LFS limit space.
  - to read such a file, simply use `pd.read_feather(file_path)`
- the easiest way to get the data in original formatting is by `python download_data.py` to download the data both as originally provided and in torch format
- Outside of python, another way to get the data is from this open-access dropbox [link](https://www.dropbox.com/sh/hwv3msz2mdfxki1/AACOk6t8z6hNfuc3s7xM9K7-a?dl=0) as the CSV files are large. This contains both the original datafiles (as provided by the course instructors) and the processed datafiles (reformatted by the `data2torchformat.py` script).

---
