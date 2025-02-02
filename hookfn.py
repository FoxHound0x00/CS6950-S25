from functools import partial
import torch

class Probe:
    def __init__(self, model):
        self.model = model
        self.layer_outs = {}

    def register_hook(self, layers_=None):
        """
        Registers forward hooks on the specified layers.

        Args:
        - layers_ (list or None): List of layer names to hook. If None, all layers will be hooked.
        """
        hooks = []
        
        # Iterate through model modules, not parameters
        for name, module in self.model.named_modules():
            if layers_ is None or name in layers_:
                hook = module.register_forward_hook(partial(self.hook_fn, name=name))
                hooks.append(hook)
        
        return hooks

    def hook_fn(self, module, input, output, name):
        """
        Hook function to store layer outputs.

        Args:AttributeError: 'Parameter' object has no attribute 'register_forward_hook'e.
        - name (str): The name of the layer.
        """
        if isinstance(output, (list, tuple)):
            # tuple -> layers return multiple tensors
            self.layer_outs[name] = [out.detach().cpu() for out in output]
        elif isinstance(output, torch.Tensor):
            # single-tensor outputs
            self.layer_outs[name] = output.detach().cpu()
        else:
            print(f"Unsupported output type at layer {name}: {type(output)}")
        
    # def remove_hooks(self):
    #     """
    #     Removes all the registered hooks.
    #     """
    #     for hook in getattr(self, "hooks", []):
    #         hook.remove()
    #     self.hooks = []


