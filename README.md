# Machine Learning for Healthcare - Problem 1

Lou Ancillon & Peter Szemraj

## about

this repo contains the code for the first problem of the Machine Learning for Healthcare course. The majority of development is done in Google Colab for the GPU compute power, the _NOTE: access required_ Drive folder is [here](https://drive.google.com/drive/folders/1vZYgrdunRwJBmDzBnQnolPrN9BoagP54?usp=sharing).

This repo primarily aims as a central home for the data processing piece, EDA and analysis, as well as the model training and evaluation.

## data

- the easiest way to get the data is from this open-access dropbox [link](https://www.dropbox.com/sh/hwv3msz2mdfxki1/AACOk6t8z6hNfuc3s7xM9K7-a?dl=0) as the CSV files are large. This contains both the original datafiles (as provided by the course instructors) and the processed datafiles (reformatted by the `data2torchformat.py` script).
- TODO: add data download script for local use

## Installation

- for using the repo locally (EDA, analysis) the primary method to install is clone + pip install.
- **NOTE: training notebooks are meant to be run on Google Colab, and the installation of those packages is handled separately inside the notebooks.**

### Steps to install for local use

1. clone the repo `git clone https://github.com/pszemraj/ml4hc-s22-project01.git`
2. set working directory to the root of the repo (`cd ml4hc-s22-project01`)
3. Install required packages with `pip install -r requirements.txt`
4. now can run scripts with `python main.py`

---
