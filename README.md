# HAR-stacked-residual-bidir-LSTM

Human Activity Recognition (HAR) using stacked residual bidirectional-LSTM cells (RNN) with TensorFlow.

The project is based on [this repository](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) which purpose was to be a tutorial.

Here, we improve accuracy on the dataset used and push the subject further, as well as using another dataset: the [Opportunity dataset](https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition). The neural network has been coded to be easy to adapt to new datasets by using a new config file.

We are using a deeper neural network with stacked LSTM cells as well as residual LSTM cells for every stacked layer, a little bit like in [ResNet](https://research.googleblog.com/2016/08/improving-inception-and-image.html), but for RNNs. Our LSTM cells are also bidirectional in term of how they pass trough the time axis.

The overall architecture of the neural network is modifiable in the config, mostly, the number of stacked and residual layers can be parametrized easily as well as whether or not bidirectional LSTM cells are to be used.

Another dataset also has been tested, see subtitles below.

## Setup

We used TensorFlow 0.11 and Python 2.

The two datasets can be loaded by running `python download_datasets.py` in the `data/` folder.


## Results using the previous UCI HAR dataset

Classifying the type of movement amongst six categories:
`(WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)`. Run config with `python config_dataset_HAR_6_classes.py`.

#### Configs tried:

The bests results for an accuracy of 93.5% are achieved with the last config of the following:

**1** stacked_layers,
**1** residual layers per stacked layer:
- final test accuracy: 0.911435365677
- best epoch's test accuracy: 0.920597195625

**1** stacked_layers,
**1** residual layers per stacked layer,
and bidirectional:
- final test accuracy: 0.889718353748
- best epoch's test accuracy: 0.908381402493

**1** stacked_layers,
**3** residual layers per stacked layer,
and bidirectional:
- final test accuracy: 0.36274176836
- best epoch's test accuracy: 0.36274176836

**2** stacked_layers,
**2** residual layers per stacked layer,
and bidirectional:
- final test accuracy: 0.911435365677
- best epoch's test accuracy: 0.927044451237

**3** stacked_layers,
**1** residual layers per stacked layer,
and bidirectional:
- final test accuracy: 0.872412443161
- best epoch's test accuracy: 0.911774635315

**3** stacked_layers,
**3** residual layers per stacked layer,
and bidirectional:
- final test accuracy: 0.910417318344
- best epoch's test accuracy: 0.93552750349


## Results from the Opportunity dataset

First, don't miss out this [nice video](https://www.youtube.com/watch?v=wzuKjjfYnu8) that offers a nice overview and understanding of the dataset.

We only used a subset of the full dataset, preprocessed as in [this paper](http://www.mdpi.com/1424-8220/16/1/115) by using [their preprocessing scripts](https://github.com/sussexwearlab/DeepConvLSTM) in order to simulate the conditions of the competition. However, we made changes to the windowing of the series for feeding in our neural network to accept longer time series (128 and not 24). In our case the LSTM's inner representation is always reset to 0 between series rather than being kept over the whole dataset.

We yet achieved an **F1 score of 0.86** with the config named `config_dataset_opportunity_18_classes.py`.
