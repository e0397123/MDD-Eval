# MDD-Eval

### Installation

```bash
conda env create -f environment.yml
conda activate tf1-nv
```
### Resources
Download data, checkpoints, libray and tools at <br />
https://www.dropbox.com/s/r5eu8tvlmqclyko/resources.zip?dl=0<br />
unzip the file and put everything under the current folder

### Setup Tool
```
cd tools/best_checkpoint_copier
python setup install
```

### Train
```
bash train.sh
```

### Score the Evaluation Data
```
bash eval.sh
```

### Correlation Analysis
see the code in evaluation_notebook.ipynb

### Full Data

Full Machine-annotated Data at https://www.dropbox.com/s/4bnha62u8uuj8ak/mdd_data.zip?dl=0

### Cite the following if use code or resources in this repo

```
@inproceedings{zhang-etal-2021-mdd,
    title = "{MDD}-{E}val: Self-Training on Augmented Data for Multi-Domain Dialogue Evaluation",
    author = "Zhang, Chen  and
      D{'}Haro, Luis Fernando  and
      Friedrichs, Thomas  and
      Li, Haizhou",
    booktitle = "Proceedings of the 36th AAAI Conference on Artificial Intelligence",
    month = March,
    year = "2022",
    address = "Online",
    publisher = "Association for the Advancement of Artificial Intelligence",
}
```
