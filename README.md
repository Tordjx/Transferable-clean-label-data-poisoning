<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="images/ENSAE.png" alt="Project logo"></a>
</p>

<h3 align="center">Transferable Adversarial Poisoning of Deep Neural Network Classifiers Using Surrogate Backbones</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> 
In this project, we present an adversarial approach designed to compromise the effectiveness of a deep neural network classifier. Our method entails utilizing surrogate backbones as substitutes for the target network to be compromised.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Prerequisites](#getting_started)
- [Installing](#Installing)
- [Results](#Results)
- [Conclusion](#Conclusion)
- [Authors](#authors)
- [References](#References)

##  About <a name = "about"></a>
As mentioned earlier, our project's primary objective is to explore an adversarial technique with the purpose of diminishing the accuracy of a deep neural network classifier. This involves utilizing surrogate backbones as replacements for the undisclosed model targeted for poisoning. We assess transferability to alternative backbones and evaluate our methodology's performance in both white-box and black-box settings.

Our venture into this field was sparked by the paper titled [Transferable Clean-Label Poisoning Attacks on Deep Neural Nets]((https://doi.org/10.48550/arXiv.1905.05897)), published in 2019. An implementation of their approach can be found in the [Convex Polytope Attack](convex_polytope_attack/Convex_polytope_Attack.py) folder.

To set up and run the project on your local machine for development and testing purposes, follow these instructions. Refer to [deployment](#deployment) for guidance on deploying the project on a live system.

## Prerequisites <a name = "getting_started"></a>

The necessary installations to replicate our experiments, along with their installation instructions, are outlined below. Please note that running this code on macOS may not be possible due to our use of Nvidia CUDA tools.

Additionally, you will need to download the image database from [The German Traffic Sign Recognition Benchmark database](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). Specifically, download the following datasets: GTSRB_Final_Test_GT.zip, GTSRB_Final_Test_Images.zip, GTSRB_Final_Training_Images.zip.

Unzip these datasets in a directory to replicate our experiments. The selection of this dataset is primarily motivated by the intentional design of traffic signs to be easily distinguishable from one another, making it a challenging task to create poisons from this dataset.


## Installing <a name = "Installing"></a>

To ensure you have all the libraries used in our simulations, refer to the [requirements.txt](requirements.txt) file. Use the following command to install any missing libraries:

```
pip install -r requirements.txt
```




##  Running the tests <a name = "tests"></a>

First, execute the [preprocessing](preprocessing.py) script to reorganize your data into a different folder. After adjusting the dataset path and fixing the output path, run the preprocessing using the following command:

```bash
python3 preprocessing.py
```

Now you are prepared to replicate the experiments using our two notebooks. Start by running [GAN.ipynb](GAN.ipynb) to generate poisoned images. Subsequently, use [Classifier.ipynb](Classifiers.ipynb) to conduct experiments.


## Results <a name="Results"></a>

### Examples of Produced Perturbations

Here are a few instances of perturbation as we alter the parameter $\alpha$, which signifies the intensity of L2 penalization.

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">

  <div style="flex-basis: 20%;">
    <img src="images/alpha1.png" alt="Image with $\alpha$ = 1" width="20%">
    <p style="text-align: center;">Figure 1: $\alpha$ = 1</p>
  </div>

  <div style="flex-basis: 20%;">
    <img src="images/alpha100.png" alt="Image with $\alpha$ = 100" width="20%">
    <p style="text-align: center;">Figure 2: $\alpha$ = 100</p>
  </div>

  <div style="flex-basis: 20%;">
    <img src="images/alpha1000.png" alt="Image with $\alpha$ = 1000" width="20%">
    <p style="text-align: center;">Figure 3: $\alpha$ = 1000</p>
  </div>

</div>

Here are a few instances of perturbation as we modify the number of encoders, from 1 to 5.

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">

  <div style="flex-basis: 20%;">
    <img src="images/1bkb.png" alt="Image with $\alpha$ = 1" width="20%">
    <p style="text-align: center;">Figure 1: 1 Backbone (ResNest26D)</p>
  </div>

  <div style="flex-basis: 20%;">
    <img src="images/3bkb.png" alt="Image with $\alpha$ = 100" width="20%">
    <p style="text-align: center;">Figure 2: 3 Backbones (ResNet34, EfficientNetB0, and Resnest26d)</p>
  </div>

  <div style="flex-basis: 20%;">
    <img src="images/5bkb.png" alt="Image with $\alpha$ = 1000" width="20%">
    <p style="text-align: center;">Figure 3: 5 Backbones (ResNet34, EfficientNetB0, Resnest26d, RegNetX_006, and DenseNet121)</p>
  </div>

</div>

With substantial L2 penalization and an increased number of encoders, the perturbation becomes less perceptible.

Here is a figure describing the components of the discriminator losses during training.

![Alt text](<images/disc losses.png>)

Also, here is a plot of the discrimination and generation losses during training.

![Alt text](<images/gen et disc loss.png>)

In a black-box context, where the target network differs from surrogates, we evaluate performance degradation across various backbones—utilizing 3 and 5 surrogate networks. For ResNest101e, akin to ResNest26D, accuracy results are detailed in the table below.

|                 | Clean | 10% Poisoned | 50% Poisoned | 100% Poisoned |
|-----------------|-------|--------------|--------------|---------------|
| 3 Backbones     | 0.5283| 0.6518       | 0.6214       | 0.1988        |
| 5 Backbones     | 0.5283| 0.6515       | 0.6329       | 0.2013        |

*Table 1: Accuracies obtained with ResNest101e, with either 3 or 5 backbones for the discriminator, with varying quantities of poisoned images.*

When the entire training set is poisoned (100%), accuracy decreases from 53% to 20%. Interestingly, when only a portion of the dataset is poisoned, the accuracy exceeds that of the clean dataset—a phenomenon we will delve into further in the Discussion section.

More results and technical details are presented and discussed in our [report](report.pdf).


## Conclusion <a name="Conclusion"></a>

In this project, we introduced an innovative adversarial approach aimed at compromising the effectiveness of a deep neural network classifier by subtly introducing contamination into the training dataset. Surrogate backbones were employed as evaluative entities to measure adaptability to alternative backbones, evaluating performance in both transparent and opaque scenarios.

Our experiments on the German Traffic Sign Recognition dataset successfully showcased the introduced contamination, resulting in a notable decline in classifier accuracy across both scenarios. Interestingly, incorporating a fraction of the contaminant unexpectedly led to an improvement in accuracy, challenging established assumptions and prompting a need for deeper investigation.

Ablation experiments confirmed the importance of integrating both perturbation and counterfeit detection components within the methodology. These unexpected findings suggest a complex interplay between adversarial elements and model performance, emphasizing the need for further investigation and potential applications in various computer vision tasks.







## ✍️ Authors <a name = "authors"></a>

- [@Valentin](https://github.com/Tordjx) 
- [@Ambre](https://github.com/ambree14) 
- [@Ilyes](https://github.com/ilyeshammouda) 



## References <a name = "References"></a>
- C.Zhu,W.R.Huang,A.Shafahi,H.Li,G.Taylor,C.Studer,T.Goldstein. [Transferable Clean-Label Poisoning Attacks on Deep Neural Nets](https://doi.org/10.48550/arXiv.1905.05897): [arXiv:1905.05897v2](https://doi.org/10.48550/arXiv.1905.05897)
- K.He, X.Zhang,S.Ren,J.Sun. [Deep Residual Learning for Image Recognition](https://doi.org/10.48550/arXiv.1512.03385): [	arXiv:1512.03385 ](
https://doi.org/10.48550/arXiv.1512.03385)
- Z.Zhou, M.R.Siddiquee,N.Tajbakhsh, J.Liang. [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://doi.org/10.48550/arXiv.1807.10165): [arXiv:1807.10165v1 ](
https://doi.org/10.48550/arXiv.1807.10165
)
