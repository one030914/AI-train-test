## Hardware

* CPU: I7-13700
* RAM: 32GB
* GPU: ARC A770

## Environment

* Python 3.10
* Pytorch 2.6.0-xpu

**Windows:**
   
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

**Remember to disable iGPU (內顯) in device manager.**

If you want to train in cuda, install Pytorch-cuda version and change this line

```python
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
```

into this

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Details

These codes are generate by ChatGPT 4o.

### CNN.py

Using the MNIST dataset to train the simple CNN model in 1 epoch, and see the performance.

### ResNet-18.py

Using the CIFAR-10 dataset to train the built-in model ResNet-18 in 10 epochs, and see the performance.

### ResNet-50.py

Using the same dataset to train the built-in model ResNet-50 in 10 epochs, and see the performance.

### ResNet-101.py

Using the same dataset to train the built-in model ResNet-101 in 10 epochs, and see the performance.