# Masters Thesis: Transfer Learning for Phenotype Prediction from Small Gene Expression Data Sets

This repository contains the code for my masters thesis (published at [Repository of the University of Ljubljana](https://repozitorij.uni-lj.si/IzpisGradiva.php?id=161755&lang=eng)), along with all trained models, used data sets, and figures with results.

## Abstract

Recent advances in biotechnology have enabled researchers to collect huge amounts of data, such as gene expression profiles from patients, which provide a foundation for personalized medicine. Such an approach requires the use of machine learning, however, a significant limitation of many medical studies is the small sample size, typically having only a few hundred patients with tens of thousands of features. In this thesis, we addressed this issue by combining multiple small gene expression data sets into a larger one, regardless of the study type, and training deep learning models capable of producing informative gene expression encodings. We used transfer learning to predict the phenotypes on unseen data sets based on the created encodings. We experimented with two model architectures: autoencoders and multitask models. Although training multi-task models proved challenging, they achieved higher average results on test data sets than autoencoders but never surpassed the results of logistic regression. An examination of the encodings revealed that autoencoders maintained the original data structure whereas the multi-task models mixed samples from different studies, but both proved that the gene expression profile can be reduced to a few informative markers.

## About this repository

The folder [data](data) contains all the Gene Expression Omnibus (GEO) data sets that we used.
The folder [figures](figures) contains all figures, further split into [data](figures/data), [methods](figures/methods), and [results](figures/results).
The folder [models](models) contains all trained models from [autoencoder](models/AutoEncoder) and [multi-task](models/MultiTask) architecture. The first filders indicate the latent layer size, while the folders in those represent the 10 different random trained models.
The folder [src](src) contains the code for data download and parsing, model implementation, training, testing, and drawing final figures.
The folder [thesis](thesis) contains the masters thesis along with presentation and two dissertations.