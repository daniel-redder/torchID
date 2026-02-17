"""
Package: torchID
Original Author: Daniel Redder
Optimized by: Gemini
"""

from typing import List, Dict, Set, Any, Optional
import sys
import torch
from termcolor import colored

# --- Helper Classes for Registry ---
registry = None


class TensorRegistry:
    """
    A helper class to build a hash map (dictionary) of system tensors once,
    allowing for O(1) lookups instead of O(N) looping.
    """
    def __init__(self, limited_system_search: str = None):
        # Maps id(tensor) -> list of names ["module.varname", ...]
        self.tensor_map: Dict[int, List[str]] = {}
        # Maps id(grad_fn) -> list of names ["module.varname", ...]
        self.grad_fn_map: Dict[int, List[str]] = {}
        
        self._build_registry(limited_system_search)

    def _build_registry(self, limited_system_search: str):
        """
        Scans sys.modules once and hashes all tensors and their grad_fns.
        """
        for name, module in list(sys.modules.items()):
            try:
                # limited_system_search optimization:
                # Only scan modules that contain a specific variable name (marker)
                if limited_system_search is not None:
                    if not hasattr(module, limited_system_search):
                        continue

                # Iterate over module variables
                for varname, obj in list(module.__dict__.items()):
                    if isinstance(obj, torch.Tensor):
                        full_name = f"{name}.{varname}"
                        
                        # Register Tensor ID
                        obj_id = id(obj)
                        if obj_id not in self.tensor_map:
                            self.tensor_map[obj_id] = []
                        self.tensor_map[obj_id].append(full_name)

                        # Register grad_fn ID (for approximate search)
                        if obj.grad_fn is not None:
                            gf_id = id(obj.grad_fn)
                            if gf_id not in self.grad_fn_map:
                                self.grad_fn_map[gf_id] = []
                            self.grad_fn_map[gf_id].append(full_name)
                            
            except Exception as e:
                # specific modules might raise errors on access, skip them
                continue

    def get_names_by_tensor(self, tensor_obj) -> Optional[List[str]]:
        return self.tensor_map.get(id(tensor_obj))

    def get_names_by_grad_fn(self, grad_fn_obj) -> Optional[List[str]]:
        return self.grad_fn_map.get(id(grad_fn_obj))


# --- Core Functions ---

def find_leaves(curr_grad_fn, leaves: List = None, current_path: List = None, paths: List = None, max_depth: int = 1000):
    """
    Parses through grad_fn functions to locate leaf nodes and get references to their tensors.
    
    OPTIMIZATION: 
    Uses backtracking with a shared 'current_path' list to avoid O(N^2) memory allocation.
    Instead of creating a new list for every recursion step, we append/pop from a single list.
    """
    # Initialize mutable defaults if None.
    # We use these single instances across the entire recursion tree.
    if leaves is None: leaves = []
    if current_path is None: current_path = []
    if paths is None: paths = []

    # Stop recursion if we've reached the depth limit
    if len(current_path) >= max_depth:
        return leaves, paths

    if not hasattr(curr_grad_fn, 'next_functions'):
        return leaves, paths

    for l in curr_grad_fn.next_functions:
        node = l[0]
        if node is None: continue

        # [Optimization] Backtracking Step 1: Push node to shared path
        current_path.append(node)

        try:
            # Check if it's a leaf variable (e.g. AccumulateGrad)
            if hasattr(node, 'variable'):
                leaves.append(node.variable)
                # Only copy the list when we actually store a result
                paths.append(list(current_path))
            
            # Recurse using the SAME mutable lists
            find_leaves(node, leaves, current_path, paths, max_depth)

        except Exception:
            pass
        finally:
            # [Optimization] Backtracking Step 2: Pop node to restore state for next iteration
            current_path.pop()
                
    return leaves, paths



# --- Model Recording ---
recordedModels = []

def tag_model(model: torch.nn.Module) -> None:
    """
    Adds a model to the model registry to track model parameters
    Also names each parameter with prefixed model name
    """
    for n, m in model.named_parameters():
        m.nmm = [f"{model.__class__.__name__}_{n}"]

    global recordedModels
    recordedModels.append(f"{model.__class__.__name__}_")


def getTensorString(obj: torch.Tensor, index: int) -> str:
    return colored(f"index: {index}, shape: {obj.shape}, dtype: {obj.dtype}", 'blue')





def identify_tensors(objs: List[torch.Tensor], approximate_search: List = None, limited_system_search: str = None):
    """
    Uses system memory record to identify tensor references in memory.
    Optimized to use hashing instead of nested loops.
    """
    
    # 1. Build the registry once (O(M) where M is system variables)
    global registry 
    registry = TensorRegistry(limited_system_search) if registry is None else registry
    
    
    found_objs_names = [None] * len(objs)
    not_found_indices = []

    # 2. Direct Lookup (O(N) where N is number of objs)
    for i, obj in enumerate(objs):
        # If it already has a tag, use it
        if hasattr(obj, 'nmm'):
            pass 
        
        # Try to find it in our system registry
        names = registry.get_names_by_tensor(obj)
        
        if names:
            if not hasattr(obj, 'nmm'):
                obj.nmm = list(names)
            else:
                # Append new found names if not already present
                for n in names:
                    if n not in obj.nmm:
                        obj.nmm.append(n)
            
            found_objs_names[i] = names[0] # Return the first found name
        else:
            if not hasattr(obj, 'nmm'):
                not_found_indices.append(i)

    # 3. Approximate Search (if paths provided and items not found)
    if approximate_search and not_found_indices:
        print(colored("Some tensor objects not found, using approximate search via grad_fn...", "red"))
        
        # We only need to search for the specific missing objects
        approx_find_optimized(
            objs=[objs[i] for i in not_found_indices],
            paths=[approximate_search[i] for i in not_found_indices],
            registry=registry, # Pass the already built registry
            verbose=True
        )
        
        # Update the found list after approximation
        for i in not_found_indices:
            if hasattr(objs[i], 'nmm'):
                found_objs_names[i] = objs[i].nmm[0]

    # Handle completely unidentified objects
    for i, obj in enumerate(objs):
        if found_objs_names[i] is None and hasattr(obj, 'nmm'):
             found_objs_names[i] = obj.nmm[0]
        elif found_objs_names[i] is None:
             pass 

    return found_objs_names


def approx_find_optimized(objs: List[torch.Tensor], paths: List[List], registry: TensorRegistry, verbose: bool = False) -> None:
    """
    Approximates the variable names of tensor leaves by looking at the nearest neighbors in the grad_fn graph.
    Uses the pre-built registry for O(1) grad_fn lookups.
    """
    
    for obj, path in zip(objs, paths):
        path_nodes = list(path) # make copy
        path_nodes.reverse() 
        
        found_match = False
        
        for i, node in enumerate(path_nodes):
            # Check if this node (grad_fn) corresponds to a known variable in our registry
            names = registry.get_names_by_grad_fn(node)
            
            if names:
                # Match found!
                approx_name = f"approximated.{i+1}separated.{names[0]}"
                obj.nmm = [approx_name]
                found_match = True
                break
        
        if not found_match:
            obj.nmm = ["unidentified"]
            if verbose:
                print(f"\n{getTensorString(obj, -1)} not found in approximation.")



def find_intersection(tensor_start: torch.Tensor, tensor_end: torch.Tensor, max_depth: int = 1000) -> Optional[List[Any]]:
    """
    Finds if there is a computational path connecting tensor_start backwards to tensor_end.
    
    Args:
        tensor_start: The downstream tensor (e.g., Output, Loss).
        tensor_end: The upstream tensor (e.g., Input, Weight).
        max_depth: limit for BFS search.
        
    Returns:
        List[grad_fn]: A list representing the path of operations from start to end.
        None: If no path exists.
    """
    # If start has no history, it can't connect backwards to anything
    if tensor_start.grad_fn is None:
        return None

    # Identify the target we are looking for in the graph
    target_grad_fn = tensor_end.grad_fn
    
    # If target is a leaf (requires_grad=True but no grad_fn), it appears as an AccumulateGrad node
    target_is_leaf = (target_grad_fn is None) and tensor_end.requires_grad

    # BFS Initialization: Queue stores (current_node, path_to_here)
    # We use BFS to find the shortest path in terms of graph depth
    queue = [(tensor_start.grad_fn, [tensor_start.grad_fn])]
    visited = set()
    visited.add(tensor_start.grad_fn)

    while queue:
        curr_node, path = queue.pop(0)

        # -- Check 1: Is the current node the intermediate operation creating tensor_end? --
        if target_grad_fn is not None and curr_node is target_grad_fn:
            return path

        # -- Check 2: Is the current node the leaf accumulator for tensor_end? --
        if target_is_leaf and hasattr(curr_node, 'variable'):
            if curr_node.variable is tensor_end:
                return path

        # Stop if path gets too long
        if len(path) >= max_depth:
            continue

        # Add neighbors to queue
        if hasattr(curr_node, 'next_functions'):
            for next_fn, _ in curr_node.next_functions:
                if next_fn is not None and next_fn not in visited:
                    visited.add(next_fn)
                    new_path = list(path)
                    new_path.append(next_fn)
                    queue.append((next_fn, new_path))
    
    return None