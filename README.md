# Alternative Text-Guided Image clustering

Authors: Andreas Stephan, Lukas Miklautz, Collin Leiber, Pedro Henrique Luz de Araujo, Dominik Répás, Claudia Plant and Benjamin Roth

This repo contains the code related to the paper "Alternative Text-Guided Image clustering". If there's questions, please contact us at [andreas.stephan@univie.ac.at](mailto:andreas.stephan@univie.ac.at).

![Overview of the Methodology](docs/methodology_overview.png)

## Abstract

Traditional image clustering techniques only
find a single grouping within visual data. In
particular, they do not provide a possibility to
explicitly define multiple types of clustering.
This work explores the potential of large vision-
language models to facilitate alternative image
clustering. We propose Text-Guided Alterna-
tive Image Consensus Clustering (TGAICC),
a novel approach that leverages user-specified
interests via prompts to guide the discovery of
diverse clusterings. To achieve this, it generates
a clustering for each prompt, groups them us-
ing hierarchical clustering, and then aggregates
them using consensus clustering. TGAICC out-
performs image- and text-based baselines on
four alternative image clustering benchmark
datasets. Furthermore, using count-based word
statistics, we are able to obtain text-based expla-
nations of the alternative clusterings. In conclu-
sion, our research illustrates how contemporary
large vision-language models can transform ex-
planatory data analysis, enabling the generation
of insightful, customizable, and diverse image
clusterings.

## Installation

There are two requirement files. If you want to explicitly work with our versions, run ```pip install fixed_requirements.txt```
In case you want to work with newer version of the used libraries, run ```pip install requirements.txt```. Note that we do not continuously test whether this works.

In addition, we need to libraries:

- [ClustPy](https://github.com/collinleiber/ClustPy): The package provides a simple way to perform clustering in Python. For this purpose it provides a variety of algorithms from different domains. 
- [ClusterEnsembles](https://github.com/GGiecold-zz/Cluster_Ensembles): A package for combining multiple partitions into a consolidated clustering (consensus clustering). 

You can install them via the instructions in the respective Github reporities.

## Data

Untar the datasets.tar.gz file and save it under a folder named ```datasets/```. Now you are ready to run experiments, given the already captioned data. If you want to use your own data, you need to run text-to-image models and image embedding.

### Image-to-Text

```bash
PYTHONPATH=. python alternative_image_clustering/experiments/run_vqa.py
```    

### Embed Images

```bash
PYTHONPATH=. python alternative_image_clustering/experiments/image_embed.py
```         

## Run experiments


### Run TGAICC

You can run TGAICC using the following command. Then automatically all experiments described in Table 3 of the paper are run, i.e. multiple sentence representations, threshhold schemes, and consensus methodologies.

```bash
PYTHONPATH=. python alternative_image_clustering/experiments/tgaicc_run.py --dataset cards
```                                                                                                                               

### Run baselines

In order to run all baselines, e.g. for the cards dataset, run

```bash
PYTHONPATH=. python alternative_image_clustering/experiments/full_baselines_run.py --dataset cards
```

You can find an overview over all of them in Table 2 of the paper.

## Citation