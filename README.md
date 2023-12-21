# torchID
A tool which allows for easy identification of leaf tensors along the backpropogation pass in PyTorch
----
By: Daniel Redder

Have you spent days trying to fix the ever so frustrating: 
```py
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).
```
**The goal of this package is to allow for fast and easy identification of leaves in your computational graph.** I built this for optimzing across diffusion because the large computational graphs generated make it very frustrating to see what is actually being optimized. 

----

This is comprised of 3 functions:

`find_leaves( grad_fn )`: this loops over `grad_fn.next_functions` to find `grad_fn` objects which have `.variable` attributes. It also records the path of `grad_fn's` taken to get to it **(used in our approximate search method)**

`tag_model( torch.nn.Module )`: this is the straightforward solution it checks `model.named_parameters()`, and assigns `leaf.nmm = name` from `named_parameters` 

`identify_tensors( [tensor] )`: this approach finds tensors which are not parameters, but are leaves in the computational graph. This works by searching `sys.modules` i.e. it **searches all references defined in all loaded modules**. We explain why we choose to do it this way on the ReadME. To mitigate the overhead this includes we have a `limited_system_search` parameter which will look for whether a specific variable exists in each module before checking it for the tensor. 
