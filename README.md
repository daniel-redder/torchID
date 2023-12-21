# torchID
A tool which allows for easy identification of leaf tensors along the backpropogation pass in PyTorch
----

Have you spent days trying to fix the ever so frustrating: 
```py
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).
```
The goal of this package is to allow for fast and easy identification of leaves in your computational graph. I built this for optimzing across diffusion because the large computational graphs generated make it very frustrating to see what is actually being optimized. 
