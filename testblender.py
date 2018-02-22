from model2blender import model2blender
import torch
from torch.autograd import Variable
from torchvision.models.vgg import vgg11
from torchvision.models.alexnet import alexnet


inputs = Variable(torch.randn(1, 3, 224, 224), requires_grad=True)
model = alexnet()
#model = vgg11()
model2blender(model, inputs)
