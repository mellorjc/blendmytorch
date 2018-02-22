import torch
import torch.nn as nn
import uuid
import inspect
import numpy as np


class BlenderModel(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.submodule = model
    def forward(self, x):
        self.submodule.train()
        offx = 0.
        offy = 0.
        offz = 0.
        writer = BlenderWriter()
        _, channels, basew, baseh = x.data.numpy().shape
        oldx = x
        try:
            x, offx, offy, offz = eval_module(self.submodule, x, offx, offy, offz, basew, baseh, True, writer)
        except:
           print('Something went wrong')
           #x = oldx
           writer.reset()
           #if isinstance(x, tuple):
           # x, _ = x
           #start_fn = x.grad_fn
           #x = self.submodule(x)
           #if isinstance(x, tuple):
           # x, _ = x
           #end_fn = x.grad_fn
           #funcs = {}
           #nodes = recurse_grad(end_fn, start_fn, Node(None, end_fn), funcs)
           #for n in nodes:
           #    if n is not None:
           #        _, offy = print_grad_graph(n, offx, offy, offz, basew, baseh, writer)
           #        offy += 3.
        print(writer.out)
        return x


class Node:
    def __init__(self, parent, fn):
        self.parents = []
        self.children = []
        self.ct = 1
        self.fn = fn
        self.marked = False
        if parent is not None:
            self.parents = [parent]
            for p in self.parents:
                if p is not None:
                    if self not in p.children:
                        p.children.append(self)
            self.ct = parent.ct + 1
        self.tensorshape = None
        if hasattr(fn, 'saved_tensors'):
            self.tensorshape = []
            for t in fn.saved_tensors:
                self.tensorshape.append(t.numpy().shape)
    def add_parent(self, parent):
        if parent not in self.parents and parent is not self:
            self.parents.append(parent)
        self.ct = max(self.ct, parent.ct + 1)


def recurse_grad(fn, stopfn, parent, funcs):
    begin_nodes = {}
    for child_fn in fn.next_functions:
        if isinstance(child_fn, tuple):
            child_fn, _ = child_fn
        if child_fn is None:
            continue
        # now we can add to the graph
        if id(child_fn) not in funcs:
            child = Node(parent, child_fn)
            funcs[id(child_fn)] = child
        else:
            child = funcs[id(child_fn)]
            if parent is not None:
                child.add_parent(parent)
        if type(child_fn).__name__ == 'AccumulateGrad':
            pass
        elif child_fn != stopfn:
            extra_nodes = recurse_grad(child_fn, stopfn, child, funcs)
            for node in extra_nodes:
                begin_nodes[node] = 1
        else:
            begin_nodes[child] = 1
    return begin_nodes


def print_grad_graph(graph, offx, offy, offz, basew, baseh, writer):
    initoffx, initoffy, initoffz = offx, offy, offz
    if graph.tensorshape is not None:
        maxscalex = 0.
        maxscaley = 0.
        maxscalez = 0.
        for shapes in graph.tensorshape:
            if len(shapes) == 4:
                _, channels, w, h = shapes
            elif len(shapes) == 3:
                channels, w, h = shapes
            else:
                w, channels = shapes
                h = 1
            scalex = float(channels)/100.
            scaley = float(w)/float(basew)
            scalez = float(h)/float(baseh)
            if not graph.marked:
                writer.make_node((offx, offy, offz), (scalex, scaley, scalez))
            maxscalex = max(maxscalex, scalex)
            maxscaley = max(maxscaley, scaley)
            maxscalez = max(maxscalez, scalez)
            offx += 2.*scalex + 1.
        offy += 2*maxscaley + 1.
    accscalex = offx - initoffx
    if graph.marked:
        return accscalex, offy
    offx = initoffx
    maxoffy_ = offy
    for parents in sorted(graph.parents, key = lambda x: x.ct)[::-1]:
        if parents is not None:
            accscx, maxoffy = print_grad_graph(parents, offx, offy, offz, basew, baseh, writer)
            offx += accscx + 1.
            maxoffy_ = max(offy, maxoffy_)
    graph.marked = True
    return max(offx-initoffx, accscalex), maxoffy_ + 1.


def eval_module(submodule, x, offx, offy, offz, basew, baseh, showinput, writer):
        if showinput:
            _, channels, w, h = x.data.numpy().shape
            scalex = float(channels)/100.
            scaley = float(w)/float(basew)
            scalez = float(h)/float(baseh)
            writer.make_node((offx, offy, offz), (scalex, scaley, scalez))
            offy += 1.5*scaley + 1.
        for name, module in submodule._modules.items():
            layer_type = type(module).__name__
            if 'Sequential' in layer_type:
                x, offx, offy, offz = eval_module(module, x, offx, offy, offz, basew, baseh, False, writer)
            else:
                if True: # for now just hardcode path. If you want to just show the pytorch module structure rather than intermediate tensors with module change to False.
                    start_fn = x.grad_fn
                    try:
                        x = module(x)
                    except:
                        try:
                            x = x.view(x.size(0), -1)
                            x = module(x)
                        except:
                            x = x.view(x.size(0), 256 * 6 * 6)
                    end_fn = x.grad_fn
                    funcs = {}
                    nodes = recurse_grad(end_fn, start_fn, Node(None, end_fn), funcs)
                    for n in nodes:
                        if n is not None:
                            _, offy = print_grad_graph(n, offx, offy, offz, basew, baseh, writer)
                            offy += 3.

                else:
                    _, channels, w, h = x.data.numpy().shape
                    offy += 1.
                    scalex = float(channels)/100.
                    scaley = float(w)/float(basew)
                    scalez = float(h)/float(baseh)
                    writer.make_node((offx, offy, offz), (scalex, scaley, scalez))
                    offy += 1.
        return x, offx, offy, offz


def model2blender(model, img):
    blendmod = BlenderModel(model)
    blendmod(img)


class BlenderWriter:
    def __init__(self):
        self.out = ''
    def make_node(self, loc, shape):
        x, y, c = shape
        matid = uuid.uuid1()
        self.out += 'bpy.ops.mesh.primitive_cube_add(location=(%s, %s, %s))\n' % loc
        self.out +='bpy.context.scene.objects.active.scale = (%s, %s, %s)\n' % shape
        self.out +='mat = bpy.data.materials.new("mat%s")\n' % matid
        self.out +='mat.emit = 1.0\n'
        self.out +='bpy.context.scene.objects.active.data.materials.append(mat)\n'
    def reset(self):
        self.out = ''


def make_node(loc, shape):
    x, y, c = shape
    matid = uuid.uuid1()
    print('bpy.ops.mesh.primitive_cube_add(location=(%s, %s, %s))' % loc)
    print('bpy.context.scene.objects.active.scale = (%s, %s, %s)' % shape)
    print('mat = bpy.data.materials.new("mat%s")' % matid)
    print('mat.emit = 1.0')
    print('bpy.context.scene.objects.active.data.materials.append(mat)')

