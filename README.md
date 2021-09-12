# MDD-Eval

## Installation

```bash
conda env create -f environment.yml
conda activate tf1-nv
```
## Resources
Download data, checkpoints, libray and tools at <br />
<br />
unzip the file and put everything under the current folder

## Setup Tool
```
cd tools/best_checkpoint_copier
python setup install
```

### train
```
bash train.sh
```

### score the evaluation data
```
bash eval.sh
```

### correlation analysis
see the code in evaluation_notebook.ipynb

