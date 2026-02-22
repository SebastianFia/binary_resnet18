# binary_resnet18

In this repository we provide the code to gradually binarize the weights of a Resnet18 pretrained
on Imagenet, loaded from timm. For our experiments, we finetune on CIFAR100.

## Methodology 

We only binarize the weights of the Linear and Conv2d layers, containing almost all the weights
of the model. We also gradually quantize to 8 bits the inputs to those layers.

The gradual quantization/binarization is done by scheduling a bin_ratio param from 0 to 1, 
and using it to interpolate  the full precision and binarized weights. 
We also schedule a p param from 0 to 1: each entry of a weight has probability p of being binarized 
(we are "sprinkling" the weight tensors with binarized values).
This means that at the start of training the weights are in full precision, and at the end they are fully binary.

The weight binarization is simulated (the weights are still in full precision, but constrained to two values, -1 and +1);
same thing for the activation binarization (which are constrained to the int8 range of values).

## Reproducibility 
Running the training script requires python3 and having a wandb account for logging.

Install python requirements:
```
pip install -r requirements.txt
```

Then run the following command to login to your wandb accout (for logging):
```
wandb login
```

Then run the experiment:
```
python3 src/train.py
```
