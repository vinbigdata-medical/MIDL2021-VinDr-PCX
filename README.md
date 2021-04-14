# Classification of Chest X-ray Pathologies in Pediatric Patients Using Deep Convolutional Neural Networks
 

This repository contains the training code for our paper entitled "Classification of Chest X-ray Pathologies in Pediatric Patients Using Deep Convolutional Neural Networks", which was submitted and under review by [Medical Imaging with Deep Learning 2021 (MIDL2021)](https://2021.midl.io/).

## What the code include?
* If you want to train yourself from scratch, we provide training and test the footwork code. In addition, we provide complete training courses.

## Train the model by yourself

* Data preparation
> We gave you the example file, which is in the folder `config/train.csv`

> You can follow it and write its path to `config/example.json`

* If you want to train the model, please run the commands below (you can change the configuration in config file, which is in the folder `config/example.json`):

```shell
pip install -r requirements.txt
python train.py --config ./config/example.json
```

* If you want to test your model, please run the command (you can also change the configuration in config file, which is in the folder `config/test_config.json`):

```shell
python test.py --config ./config/test_config.json
```

## The performance of the proposed method

| Classifier            |      | AUROC                | Sensitivity         | Specificity         | F1 score            |
|-----------------------|------|----------------------|---------------------|---------------------|---------------------|
| DenseNet-121          | val  | 0.748 (0.726-0.768)  | 0.732 (0.699-0.773) | 0.655 (0.637-0.670) | 0.286 (0.269-0.303) |
|                       | test | 0.733 (0.712-0.753)  | 0.689 (0.653-0.730) | 0.631 (0.614-0.645) | 0.268 (0.253-0.283) |
| DenseNet-169          | val  | 0.748 (0.726-0.769)  | 0.761 (0.723-0.796) | 0.634 (0.618-0.650) | 0.285 (0.268-0.300) |
|                       | test | 0.739 (0.719-0.758)  | 0.733 (0.697-0.767) | 0.625 (0.609-0.641) | 0.274 (0.259-0.288) |
| ResNet-101            | val  | 0.746 (0.724-0.767)  | 0.707 (0.667-0.746) | 0.690 (0.674-0.707) | 0.288 (0.271-0.305) |
|                       | test | 0.729 (0.709-0.751)  | 0.672 (0.632-0.709) | 0.669 (0.653-0.687) | 0.273 (0.256-0.287) |
| DenseNet-121+Transfer | val  | 0.781 (0.761-0.800)  | 0.753 (0.718-0.786) | 0.686 (0.670-0.702) | 0.305 (0.287-0.321) |
|                       | test | 0.762 (0.742-0.782)  | 0.742 (0.706-0.776) | 0.652 (0.636-0.668) | 0.287 (0.272-0.301) |
| DenseNet-169+Transfer | val  | 0.762 (0.740-0.783)  | 0.741 (0.701-0.780) | 0.676 (0.658-0.695) | 0.306 (0.290-0.322) |
|                       | test | 0.762 (0.742-0.782), | 0.742 (0.703-0.780) | 0.644 (0.625-0.663) | 0.297 (0.283-0.310) |
| ResNet-101+Transfer   | val  | 0.766 (0.746-0.786)  | 0.729 (0.688-0.769) | 0.690 (0.674-0.706) | 0.307 (0.289-0.324) |
|                       | test | 0.763 (0.743-0.783)  | 0.712 (0.671-0.752) | 0.665 (0.650-0.681) | 0.298 (0.282-0.313) |
| Ensemble              | val  | 0.795 (0.776-0.813)  | 0.752 (0.712-0.790) | 0.711 (0.695-0.726) | 0.321 (0.304-0.338) |
|                       | test | 0.786 (0.767-0.804)  | 0.742 (0.704-0.778) | 0.680 (0.663-0.696) | 0.306 (0.291-0.320) |