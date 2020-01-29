xray
==============================

The aim of this project is to classify x-ray images of hands into normal or not normal hands. Because of the high cost of labelling the data the task should be done in an unsupervised way. That means that the labels should not be included while training. 

The data is a subset of the [MURA dataset](http://stanfordmlgroup.github.io/competitions/mura) ([paper](https://arxiv.org/pdf/1712.06957.pdf)) and includes x-ray images of hands. The data seems not to be very clean (see [First look at dataset](https://gitlab.lrz.de/random_state42/xray/wikis/First-look-at-dataset)). To handle this problem a data cleaning pipeline was implemented (see [Data cleaning & preprocessing](https://gitlab.lrz.de/random_state42/xray/wikis/Data-cleaning-&-preprocessing) and [Results of hand center localisation](https://gitlab.lrz.de/random_state42/xray/wikis/Result-of-Hand-Center-Localisation)).

The summary of the research about different unsupervised methods to find anomalies can be found [here](https://gitlab.lrz.de/random_state42/xray/wikis/Unsupervised-Deep-Learning-for-Medical-Image-Analysis). [Different metrics](https://gitlab.lrz.de/random_state42/xray/wikis/Evaluation-strategy-&-metrics) are presented to evaluate different models.

### General

### Data Cleaning

### Research Of Different Methods

### Evaluation of Models

Installation
==============================
`git clone https://github.com/Valentyn1997/xray.git`
`pip3 install -r requirements.txt`


Project Organization
------------

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this projec
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                              the creator's initials, and a short `-` delimited description, e.g.
    │                              `1.0-jqp-initial-data-exploration`.
    │   ├── augmentation        <- Jupyter notebooks. Analysis of augmentation
    │   ├── dataset             <- Jupyter notebooks. Analysis of dataset
    │   ├── edge_detection      <- Jupyter notebooks. Analysis of edge detection
    │   ├── models              <- Jupyter notebooks. Analysis of models
    │   └── preprocessing       <- Jupyter notebooks. Analysis of preprocessing steps
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── data                <- Scripts to split and transform the data
    │   │   └── __init__.py     <- Datagenerator for PyTorch
    │   │   └── transforms.py   <- Different online transformations
    │   ├── features            <- Scripts to turn raw data into features for modeling
    │   │   ├── augmentation    <- augmentation for training
    │   │   ├── crop            <- square detection and cropping
    │   │   ├── grabcut         <- example for grabcut
    │   │   ├── hand_detection  <- SSD hand detection with cropping
    │   │   ├── inversion       <- invert color images
    │   │   ├── pixelwise_loss  <- calculate pixelwise_loss and create heatmap
    │   │   ├── topk            <- calculate top k loss
    │   │   └── unsupervised_anomaly_detection <- different unsupervised non deep learning methods
    │   ├── models_creation     <- Scripts to train hand detection model see wiki for more information
    │   ├── models              <- Scripts to train models and model definitions
    │   │   └── train.py        <- Script to train the models
    │   │   └── alphagan.py     <- Modeldefinition alphagan
    │   │   └── autoencoders.py <- Modeldefinition differnt convolutional autoencoder
    │   │   └── gans.py         <- Modeldefinition gan
    │   │   └── sgan.py         <- Modeldefinition sgan
    │   │   └── torchsummary.py <- get summary of models
    │   │   └── vaetorch.py     <- Modeldefinition variational autoencoder
    │   │
    │   │── visualization       <- Scripts to create exploratory and results oriented visualizations
    │   │   └── plot_loss_label.py <- script to create loss-label plot for convolutional autoencoder
    │   └-- utils.py            <- Scripts with helper function 
    └── tox.ini                 <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
