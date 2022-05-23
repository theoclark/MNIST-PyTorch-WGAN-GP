# MNIST PyTorch WGAN-GP

Pytorch implementation of [WGAN-GP](https://arxiv.org/abs/1704.00028) trained on MNIST.

## Examples

#### Samples
![samples](https://github.com/theoclark/MNIST-PyTorch-WGAN-GP/samples.png)

## Usage

### Inference

Use the inference.py script to generate digits. The parser takes one positional argument: the number of images you want to generate. Images will be saved to the 'Saved_Images' folder.

To create 50 images:

```bash
python inference.py 50
```

### Training

Use the train.ipynb notebook to train your model. I used Google Colab and it took 1-2 hours on a free GPU to train to the level seen in the samples.

Change the directory paths as required to save image samples and model weights.

This is an implementation of the [WGAN-GP paper](https://arxiv.org/abs/1704.00028) which itself is based on the original [WGAN paper](https://arxiv.org/abs/1701.07875).

The Frechet Inception Distance (FID) score can be calculated as a performance metric. FID module was imported from [here](https://github.com/mseitzer/pytorch-fid) and the original paper is [here](https://arxiv.org/abs/1706.08500).
