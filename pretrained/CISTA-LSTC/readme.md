<!-- The ```cista-lstc.pth.tar``` checkpoint is used in the [CISTA-Flow paper](https://arxiv.org/pdf/2403.11961), while the old version is used in the [TPAMI paper](https://ieeexplore.ieee.org/abstract/document/10130595) and the [V2E2V repository](https://github.com/lsying009/V2E2V). The training datasets and strategies are slightly different for these two checkpoints. Note that the new version is not necessarily better than the old one. -->

```cista-lstc-old.pth.tar```: The original pretrained model used in the [CISTA-LSTC paper](https://ieeexplore.ieee.org/abstract/document/10130595).

```cista-lstc.pth.tar```: checkpoint used in the [CISTA-Flow paper](https://arxiv.org/pdf/2403.11961), trained using fixed number of events per reconstruction, more suitable for ```low_k_events``` evaluation mode. 

```cista-lstc-randNE.pth.tar```: The updated version, trained using various number of events per reconstruction, more suitable for ```std``` evaluation mode. 