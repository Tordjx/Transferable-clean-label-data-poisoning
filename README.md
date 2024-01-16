<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="images/ENSAE.png" alt="Project logo"></a>
</p>

<h3 align="center">Transferable adversarial poisoning of deep neural
network classifiers using surrogate backbones</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> 
In this project, we suggest employing an adversarial approach to impair the efficiency of a deep neural network classifier. This involves utilizing surrogate backbones as substitutes for the undisclosed model targeted for poisoning.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Prerequisites](#getting_started)
- [Installing](#Installing)
- [Results](#Results)
- [Conclusions](#Conclusions)
- [Authors](#authors)
- [References](#References)

##  About <a name = "about"></a>
As previously mentioned, the primary goal of this project is to investigate an adversarial technique aimed at reducing the efficacy of a deep neural network classifier. This involves employing surrogate backbones as substitutes for the undisclosed model targeted for poisoning. We assess the transferability to alternative backbones and evaluate the performance of our methodology in both white box and black box settings.

Our initiation into this topic was inspired by the paper [Transferable Clean-Label Poisoning Attacks on Deep Neural Nets]((https://doi.org/10.48550/arXiv.1905.05897)), published in 2019. An implementation of their approach can be found in the [convex polytope attack](convex_polytope_attack/Convex_polytope_Attack.py) folder . While our code draws inspiration mainly from their implementation for simulation, it has been adapted to an object-oriented structure.





These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

## Prerequisites <a name = "getting_started"></a>

The necessary installations to reproduce our experiments and their installation instructions are outlined below. Please be aware that due to our use of Nvidia CUDA tools, running this code on macOS may not be possible.

Additionally, you will need to download the image database from [The German Traffic Sign Recognition Benchmark database](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). Specifically, download the following datasets: GTSRB_Final_Test_GT.zip, GTSRB_Final_Test_Images.zip, GTSRB_Final_Training_Images.zip.

Unzip these datasets in a directory to replicate our experiments. The selection of this dataset is primarily motivated by the fact that traffic signs were intentionally designed to be easily distinguishable from one another, making it a challenging task to create poisons from this dataset.










## Installing <a name = "Installing"></a>

To ensure that you have all the libraries used in our simulations, you can refer to the file  [requirments.txt](requirements.txt).\
You can run the following command to install the missing libraries.
```
pip install -r requirements.txt
```




##  Running the tests <a name = "tests"></a>

You will need first to run the script [preprocessing](preprocessing.py) to get your data structured differently in another folder. \
After modifying the path to your dataset and fix the output path you can run the preprocessing using this command:
```
python3 preprocessing.py
```
Now you are ready to reproduce the experiments thanks to our 2 notebooks. Run [GAN.ipynb](GAN.ipynb) to creat the poisened images and then you use [Classifier.ipynb](Classifiers.ipynb)  to make experiments.


## Results <a name = "Results"></a>
Here is a figure discribing the components of the discriminator losses during training.

![Alt text](<images/disc losses.png>)

And finally here is a plot of the descrimination and generation losses during training. 

![Alt text](<images/gen et disc loss.png>)

More results and technical details are presented and discussed  in our [report](report.pdf). 

## Conclusions <a name="Conclusions"></a>

In this project, we presented an innovative adversarial approach designed to compromise the effectiveness of a deep neural network classifier by subtly introducing contamination into the training dataset. Surrogate backbones were employed as evaluative entities to measure the adaptability to alternative backbones, evaluating performance in both transparent and opaque scenarios. Our experiments on the German Traffic Sign Recognition dataset successfully showcased the introduced contamination, resulting in a notable decline in classifier accuracy across both scenarios. Interestingly, incorporating a fraction of the contaminant unexpectedly led to an improvement in accuracy, challenging established assumptions and prompting a need for deeper investigation. Ablation experiments confirmed the importance of integrating both perturbation and counterfeit detection components within the methodology. These unexpected findings suggest a complex interplay between adversarial elements and model performance, emphasizing the need for further investigation and potential applications in various computer vision tasks.






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

