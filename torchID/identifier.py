"""
Package: torchID
Author: Daniel Redder
"""

from typing import List
import sys
import torch
from termcolor import colored


def find_leaves(curr_grad_fn, leaves:List = [], depthList:List = [], outputDepthList:List = [])->torch.TensorType:
    """
    curr_grad_fn: the initital grad_fn to start the search
    leaves: the list of tensors to append to (default = [])

    Parses through grad_fn functions to locate leaf nodes and get references to their tensors
    """
    for l in curr_grad_fn.next_functions:

        if hasattr(l[0], 'variable'):
            leaves.append(l[0].variable)
            outputDepthList.append(depthList + [l[0]])
            try:
                next_leaf, outputDepthList = find_leaves(l[0], [], depthList=depthList + [l[0]], outputDepthList=outputDepthList)
                assert next_leaf is not None, "next leaf is none"
                leaves = leaves + next_leaf
            except Exception as e:
        
                pass
        else:
            try:
                next_leaf, outputDepthList = find_leaves(l[0], [], depthList=depthList + [l[0]], outputDepthList=outputDepthList)
                leaves = leaves + next_leaf
            except Exception as e:
       
                pass
    return leaves, outputDepthList



#model recording
recordedModels=[]

def tag_model(model:torch.nn.Module)->None:
    """
    model: the torch.nn.Module to be tagged

    Adds a model to the model registry to track model parameters
    Also names each parameter with prefixed model name
    """
    for n,m in model.named_parameters(): m.nmm = [model.__class__.__name__ + "_" + n]

    global recordedModels
    recordedModels.append(model.__class__.__name__ + "_")





def getTensorString(obj:torch.TensorType, index:int)->str:
    """
    tensor: the tensor to get the string of

    Gets the string of a tensor
    """
    return colored(f"index: {index}, shape: {obj.shape}, dtype: {obj.dtype}",'blue')




def approx_find(objs:List[torch.TensorType], paths:List,  limited_system_search:str=None, verbose:bool=False)->None:
    """
    Approximates the variable names of tensor leaves by looking at the nearest neighbors
    """

    assert all([isinstance(o, torch.Tensor) for o in objs]), "all objects must be tensors, are you sure these weren't found?"
    assert all([not hasattr(l,'nmm') for l in objs]), "all objects must not have nmm attribute, are you sure these weren't found?"

    found_objs = []
    found_names = []

    paths_check = set([o for ob in paths for o in ob])

    for name, module in sys.modules.items():
        try:
            #?check shrunk module checking condition
            items = list(module.__dict__.items())
            items = [it[0] for it in items]

            if limited_system_search not in items and limited_system_search is not None:
                continue

            #?compare .grad_fn objects (they are instanced so this should work...)
            for varname, obj in module.__dict__.items():
                #filters through loaded objects in memory
                if not isinstance(obj, torch.Tensor): continue
                if any([obj.grad_fn is p for p in paths_check]):
                    found_objs.append(obj.grad_fn)
                    found_names.append(name + "." + varname)

        except Exception as e:
            print(e)
    

    #?apply nearest variable name to each leaf
    for obj, path in zip(objs, paths):

        #?invert the list to get the nearest variable name
        path.reverse()
        
        for i,p in enumerate(path):
            #! a more normal "in" check may be simpler here
            if any([p is f for f in found_objs]):
                obj.nmm = [f"approximated.{i+1}seperated.{found_names[found_objs.index(p)]}"]
                break
        
        if verbose: print( "\n",getTensorString(obj, objs.index(obj)), " not found", path, "\nFound Objs: ", found_objs,"\n", found_names)
        

        if not hasattr(obj, 'nmm'):
            print( "\n",getTensorString(obj, objs.index(obj)), " not found", path, "\nFound Objs: ", found_objs,"\n", found_names)
            obj.nmm = "unidentified"


        for name, module in sys.modules.items():
                try:
                    items = list(module.__dict__.items())
                    items = [it[0] for it in items]

                    if limited_system_search not in items:
                        continue

                    for varname, modObj in module.__dict__.items():
                        #filters through loaded objects in memory
                     
                        hold = [found is obj for found in objs]
                        for i in range(len(hold)):
                            if hold[i]:
                                if not hasattr(obj, 'nmm'):
                                    obj.nmm = [name + "." + varname]
                                else: 
                                    obj.nmm.append(name + "." + varname)

                                found_objs[i] = name + "." + varname
                except Exception as e:
                    print(e)








def identify_tensors(objs, approximate_search:List = None, limited_system_search:str=None):
    """
    Uses system memory record to identify tensor references in memory #! need to double check that this works on gpu tensors, it should
    #inspired by:
        https://stackoverflow.com/questions/43523307/get-variables-name-by-self-inside-of-a-method-of-a-class-in-python
        https://stackoverflow.com/questions/72497140/in-python-multiprocessing-why-is-child-process-name-mp-main-is-there-a-way
        https://stackoverflow.com/questions/56506971/get-line-number-where-variable-is-defined
    """
    
    found_objs = [o if not hasattr(o, 'nmm') else None for o in objs ]


    # print(found_objs)
    for name, module in sys.modules.items():
        try:
            #limited search for large module spaces this is recommended
            if limited_system_search is not None:
                items = list(module.__dict__.items())
                items = [it[0] for it in items]

                if limited_system_search not in items:
                    continue

            for varname, obj in module.__dict__.items():
                    #filters through loaded objects in memory
                    

                    hold = [found is obj for found in objs]
                    for i in range(len(hold)):
                        if hold[i]:
                            if not hasattr(obj, 'nmm'):
                                obj.nmm = [name + "." + varname]
                            else: 
                                obj.nmm.append(name + "." + varname)

                            found_objs[i] = name + "." + varname
        except Exception as e:
            print(e)
    
    try:
        assert all([isinstance(o, str) or o is None for o in found_objs]), f"torchID: not all tensor objects found, {found_objs}"
    except Exception as e:
        if approximate_search is None:
            print(colored("Tensor object not found, no path provided, provide second return object from find_leaves to use_limited_search param","red"))
            for obj in found_objs: 
                if not isinstance(obj, str) and obj is not None: obj.nmm = "unidentified"
            return found_objs

        print(colored("Tensor object not found, using approximate search for the following objects: ","red"))
        approx_items = []
        approx_paths = []
        for i, obj in enumerate(found_objs):

            if not isinstance(obj, str) and obj is not None:
                approx_items.append(obj)
                approx_paths.append(approximate_search[i])

                print( colored(f"index: {i}, shape: {obj.shape}, dtype: {obj.dtype}",'blue') )
        approx_find(approx_items, approx_paths, limited_system_search=limited_system_search)

    return found_objs


