# blendmytorch
Produces a list of Blender API commands to make a 3d model of a pytorch model.

Not quite ready yet. 

tesblender.py shows how to run with a model. This produces a list of blender commands. If you put them into the blender python console it'll create the boxes for you. 
Layout still needs a lot of work.

This currently makes heavy presumptions about the module you want to display in blender.
The top level module probably needs to define the submodules it invokes in order and in the forward method definition it should just call these submodules and nothing else. Sequential submodules are expanded recursively and non-sequential submodules, the gradient graph is searched to find all intermediate tensors to display.

