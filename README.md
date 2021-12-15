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

###

Full Machine-annotated Data Under Preparatio

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

