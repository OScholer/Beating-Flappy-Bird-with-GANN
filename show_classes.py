import copy
import turtle
import random
import torch
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class Ball:
    def __init__(self,net):
        self.ball = turtle.Turtle()
        self.ball.speed(0)
        self.ball.shape("square")
        self.ball.color("white")
        self.ball.penup()
        self.ball.goto(-300,0)
        self.gravity = 1
        self.velocity = 0
        self.x = -300

        self.score = 0
        self.alive = True

        self.d_vec = torch.tensor([0,0,0,0])

        self.brain = net

    def move(self):
        self.velocity += self.gravity
        self.velocity *=0.9
        y = self.ball.ycor() - self.velocity
        self.ball.sety(y)

    def brain_prediction(self,data):
        return self.brain(data)
    
    def Up(self):
        self.velocity -= 8
    def Up_2(self):
        self.velocity -= 40


    def gety(self):
        return self.ball.ycor()

    def sety(self,d):
        self.ball.sety(d)

    def update_color(self):
        self.ball.color("green")

    def check_for_hit(self, pipe):
        if (pipe.getx() == self.x and pipe.y1>self.gety()) or (pipe.getx() == self.x and pipe.y2<self.gety()) or (self.gety() < -300):
            self.alive = False

    def resets(self):
        self.ball.goto(-300,0)
        self.gravity = 1
        self.velocity = 0
        self.score = 0
        self.alive = True


class Pipe:
    def __init__(self,xinit,yinit):

        self.stretch1 = random.randint(1,44)
        self.stretch2 = 45-self.stretch1

        self.y1 = -300+self.stretch1*10
        self.y2 =  300-self.stretch2*10

        self.pipe = turtle.Turtle()
        self.pipe.speed(0)
        self.pipe.shape("square")
        self.pipe.color("white")
        self.pipe.shapesize(stretch_wid=self.stretch1,stretch_len=1)
        self.pipe.penup()
        self.pipe.goto(xinit,yinit)

        self.pipe1 = turtle.Turtle()
        self.pipe1.speed(0)
        self.pipe1.shape("square")
        self.pipe1.color("white")
        self.pipe1.shapesize(stretch_wid=self.stretch2,stretch_len=1)
        self.pipe1.penup()
        self.pipe1.goto(xinit,-yinit)

        self.velocity = 5

    def move(self):
        x = self.pipe.xcor() - self.velocity
        self.pipe.setx(x)
        self.pipe1.setx(x)

    def getx(self):
        return self.pipe.xcor()


class Net(nn.Module):
    def __init__(self, N_depth=2, N_width=6):
        super(Net,self).__init__()

        self.N_depth = N_depth
        self.N_width =N_width
        self.MLP = nn.Sequential()

        start = "start"
        self.MLP.add_module(name=f"L{start}", module=nn.Linear(6, N_width))
        for i in range(N_depth):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(N_width, N_width))
            #self.MLP.add_module(name=f"ReLU{i}", module=nn.ReLU())

        self.MLP.add_module(name="LOL",module=nn.Linear(N_width,1))

        self.add_tensors = {}
        for name,tensor in self.named_parameters():
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())


    def evolve(self, sigma, rng_state=0):
        #torch.manual_seed(rng_state)
        #self.evolve_states.append((sigma, rng_state))

        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            to_add.bernoulli(0.4)
            tensor.data.add_(to_add)

    def dropout_test(self,X,p_drop):
        binomial = torch.bernoulli(X,p_drop)
        X[binomial==0] = 0
        return X/p_drop

    def forward(self, z):
        z = self.dropout_test(z.view(1,6).float(),1)
        return torch.sigmoid(self.MLP(z))


def closest_pipe(pipes):
    closet_pipe = pipes[0]
    if pipes[0].getx()< -300:
        closet_pipe = pipes[1]
    return closest_pipe
