# The code

## One-time scripts
- [download.py](download.py): downloads the human GEO data sets and their corresponding papers/abstracts.
- [manual_inspect.py](manual_inspect.py) and [manual_inspect_helper.py](manual_inspect_helper.py): manual inspection CLI tool for GEO data sets. For each data set it displays its title, abstract, description, and summary statistics. Then it displays info of different sample annotations/tasks and the correlation matrix between tasks. It then takes user inputs to decide what tasks to accept/reject and how to remap tasks to binary tasks.
- [prepare_train.py](prepare_train.py): keep only data sets with certain gene expression attributes and sufficient sample size, then split the data into training and testing group.
- [sample_distributions.py](sample_distributions.py): scrape the pages of ArrayExpress, GEO, and dbGaP for data set sample size.

## Implementations
- [l1000.py](l1000.py): holds the L1000 genome data.
- [geo_dataset.py](geo_dataset.py): implements the Dataset class, which holds and prepares training, validation, and testing matrices.
- [geo_models.py](geo_models.py): implements different architectures: autoencoder, constraint autoencoder, variational autoencoder, multi-task model, and multi-task model with attention.
- [geo_train.py](geo_train.py): implements the training loop.
- [geo_test.py](geo_test.py): implements the testing loop.

## Experiments
- [Graphics_for_thesis.ipynb](Graphics_for_thesis.ipynb): contains code for analysis of all results and for plotting all figures.
- [run_autoencoder.py](run_autoencoder.py): train and test multiple autoencoder models.
- [run_multitask.py](run_multitask.py): train and test multiple multi-task models.
- [test_baseline.py](test_baseline.py): create the baseline results with logistic regression.
- [test_autoencoder.py](test_autoencoder.py): retest multiple autoencoder models.
- [test_multitask.py](test_multitask.py): retest multiple multi-task models.