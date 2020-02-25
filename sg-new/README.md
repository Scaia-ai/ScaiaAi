# SG-Ne

This is our reboot of SG using the Huggingface transformers library 


## Environment

Recommend to create a new conda environment called huggingface


```bash

conda create --name huggingface tensorflow==2.0 pytorch
conda activate huggingface
pip install transformers
```


You may need to do a source install of huggingface:

```bash
conda activate huggingface
git clone git@github.com:huggingface/transformers.git
cd transformers
pip install . 
```

## Example

I have installed the huggingface example in the [example](./example) folder.

To run:

```bash
export SCAIA_HOME=/path/to/your/scaia/home
cd $SCAIA_HOME/example
cd example
./run_example.sh
```

