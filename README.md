# torchID
A tool which allows for easy identification of leaf tensors along the backpropogation pass in PyTorch
----
By: Daniel Redder

----
TODOS:
- Test on non module instances (may require the use of `locals()` will be addressed soon
- Add full method signatures


Have you spent days trying to fix the ever so frustrating: 
```py
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).
```
**The goal of this package is to allow for fast and easy identification of leaves in your computational graph.** We built this for optimzing across diffusion because the large computational graphs generated make it very frustrating to see what is actually being optimized. 

----

This is comprised of 3 functions:

`find_leaves( grad_fn )`: this loops over `grad_fn.next_functions` to find `grad_fn` objects which have `.variable` attributes. It also records the path of `grad_fn's` taken to get to it **(used in our approximate search method)**

`tag_model( torch.nn.Module )`: this is the straightforward solution it checks `model.named_parameters()`, and assigns `leaf.nmm = name` from `named_parameters` 

see [test_simple.ipynb](https://github.com/daniel-redder/torchID/blob/main/test_simple.ipynb) for an example.

----

`identify_tensors( [tensor] )`: this approach finds tensors which are not parameters, but are leaves in the computational graph. This works by searching `sys.modules` i.e. it **searches all references defined in all loaded modules**.  To mitigate the overhead this includes a `limited_system_search` parameter which looks for whether a specific variable exists in each module before checking it for the tensor. 

see [test_approximate.ipynb](https://github.com/daniel-redder/torchID/blob/main/test_approximation.ipynb)https://github.com/daniel-redder/torchID/blob/main/test_approximation.ipynb for a example.

We decided to use reference searching rather than a "monkey patching" (wrapping all tensors so they save a name on initialization) approach because "monkey patching" makes it more difficult to work with state dictionaries, and thus breaks our goal of **identifying tensors leaves in arbitrary "mostly" unmodified packages**, here mostly meaning the minimal ammount possible. 

The only modification that this project can benefit from is the `limited_system_search` parameter of `identify_tensor` which makes memory searching faster. Otherwise this should work in most all cases. 
