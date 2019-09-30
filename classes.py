'''
File contains the Class for the neuronal nets, the genetic Algorithm, and the objects that are involved in the game
'''
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

import functions


'''Networks for Part 1 - Networks to put into the Genetic Algoritm'''
#Make a net without non-linear activation functions
class Net(nn.Module):
    def __init__(self, N_depth=2, N_width=6, N_input=6):
        #create variable neuronal nets that can have depths between 1 and 2 and widths between 1 and 6
        super(Net,self).__init__()
        #set up net parameters
        self.N_input = N_input
        self.N_depth = N_depth
        self.N_width = N_width
        self.MLP = nn.Sequential()
        #build net
        start = "start"
        self.MLP.add_module(name=f"L{start}", module=nn.Linear(self.N_input, N_width))
        for i in range(N_depth):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(N_width, N_width))

        #output layer
        self.MLP.add_module(name="LOL",module=nn.Linear(N_width,1))

        #save weight attributes in variable.
        self.add_tensors = {}
        for name,tensor in self.named_parameters():
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())



    def evolve(self, sigma, binomial=0.4, rng_state=0):
        #torch.manual_seed(rng_state)
        #self.evolve_states.append((sigma, rng_state))

        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            to_add.mul_(tensor.data)
            to_add.bernoulli(binomial)
            tensor.data.add_(to_add)

    def dropout_test(self,X,p_drop):
        #throws away random inputs, the probability to keep an input is p_drop
        binomial = torch.bernoulli(X,p_drop)
        X[binomial==0] = 0
        return X/p_drop

    def forward(self, z):
        z = self.dropout_test(z.view(1,self.N_input).float(),1)
        return torch.sigmoid(self.MLP(z))



#Make a Network that uses only ReLU activation functions
class ReLU_Net(nn.Module):
    def __init__(self, N_depth=2, N_width=6, N_input=6):
        #create variable neuronal nets that can have depths between 1 and 2 and widths between 1 and 6
        super(ReLU_Net,self).__init__()
        #set up net parameters
        self.N_input = N_input
        self.N_depth = N_depth
        self.N_width = N_width
        self.MLP = nn.Sequential()
        #build net
        start = "start"
        self.MLP.add_module(name=f"L{start}", module=nn.Linear(self.N_input, N_width))
        self.MLP.add_module(name=f"ReLU{start}", module=nn.ReLU())
        for i in range(N_depth):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(N_width, N_width))
            self.MLP.add_module(name=f"ReLU{i}", module=nn.ReLU())

        #output layer
        self.MLP.add_module(name="LOL",module=nn.Linear(N_width,1))

        #save weight attributes in variable.
        self.add_tensors = {}
        for name,tensor in self.named_parameters():
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())



    def evolve(self, sigma, binomial=0.4, rng_state=0):
        #torch.manual_seed(rng_state)
        #self.evolve_states.append((sigma, rng_state))

        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            to_add.mul_(tensor.data)
            to_add.bernoulli(binomial)
            tensor.data.add_(to_add)

    def dropout_test(self,X,p_drop):
        #throws away random inputs, the probability to keep an input is p_drop
        binomial = torch.bernoulli(X,p_drop)
        X[binomial==0] = 0
        return X/p_drop

    def forward(self, z):
        z = self.dropout_test(z.view(1,self.N_input).float(),1)
        return torch.sigmoid(self.MLP(z))



#Make a Net filled with random non-linear activation functions
class Rand_Net(nn.Module):
    def __init__(self, N_depth=2, N_width=6, N_input=6):
        #create variable neuronal nets that can have depths between 1 and 2 and widths between 1 and 6
        super(Rand_Net,self).__init__()

        #define a list of non-linear activation functions to find the optimal network
        self.neurons = [[nn.ReLU(), "ReLU"],
                   [nn.SELU(), "SELU"],
                   [nn.Tanh(), "Tanh"],
                   [nn.Softplus(), "Softplus"],
                   [nn.CELU(), "CELU"],
                   [nn.Sigmoid(), "Sigmoid"],
                   [nn.Softmax(dim=1), "Softmax"]]
        #set up net parameters
        self.N_input = N_input
        self.N_depth = N_depth
        self.N_width = N_width
        self.MLP = nn.Sequential()
        #build net
        start = "start"
        self.MLP.add_module(name=f"L{start}", module=nn.Linear(self.N_input, N_width))
        #self.MLP.add_module(name=f"ReLU{i}", module=nn.ReLU())
        for i in range(N_depth):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(N_width, N_width))
            #Choose a random activation function to add to the layer
            neuron = random.choice(self.neurons)
            self.MLP.add_module(name=neuron[1]+str(i), module=neuron[0])

        #output layer
        self.MLP.add_module(name="LOL",module=nn.Linear(N_width,1))

        #save weight attributes in variable.
        self.add_tensors = {}
        for name,tensor in self.named_parameters():
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())



    def evolve(self, sigma, binomial=0.4):
        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            to_add.mul_(tensor.data)
            to_add.bernoulli(binomial)
            tensor.data.add_(to_add)

    def dropout_test(self,X,p_drop):
        #throws away random inputs, the probability to keep an input is p_drop
        binomial = torch.bernoulli(X,p_drop)
        X[binomial==0] = 0
        return X/p_drop

    def forward(self, z):
        z = self.dropout_test(z.view(1,self.N_input).float(),1)
        return torch.sigmoid(self.MLP(z))

'''_____________________________________________________________________________________________________________________________________________________'''

'''Networks for Part 2 - Networks to train the VAE'''
class g_Net(nn.Module):
    def __init__(self, N_depth=1, N_width=1, N_input=1):
        super(g_Net,self).__init__()
        
        self.MLP = nn.Sequential()
       
        #build net
        self.MLP.add_module(name="l1", module=nn.Linear(5084, 2500))
        self.MLP.add_module(name="r1", module=nn.ReLU())
        self.MLP.add_module(name="l2",module=nn.Linear(2500,1700))
        self.MLP.add_module(name="r2", module=nn.ReLU())
        self.MLP.add_module(name="l3",module=nn.Linear(1700,850))
        self.MLP.add_module(name="r3", module=nn.ReLU())
        self.MLP.add_module(name="l4",module=nn.Linear(850,200))
        self.MLP.add_module(name="r4", module=nn.ReLU())
        self.MLP.add_module(name="l5",module=nn.Linear(200,50))
        self.MLP.add_module(name="r5", module=nn.ReLU())
        self.MLP.add_module(name="l6",module=nn.Linear(50,10))
        self.MLP.add_module(name="r6", module=nn.ReLU())
     
        #output layer   
        self.MLP.add_module(name="l7",module=nn.Linear(10,1))
        self.MLP.add_module(name="r7", module=nn.Sigmoid())
        

        
        #save weight attributes in variable.
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

    def forward(self, z):
        z = z.reshape(1,82*62).float()
        return self.MLP(z)
    

class conv_Net(nn.Module):
    def __init__(self, N_depth=1, N_width=1, N_input=1):
        super(conv_Net,self).__init__()
        
        self.l1 = nn.Conv1d(62, 33, 3, stride=2)
        self.l2 = nn.MaxPool1d(3, stride=2)

        self.l3 = nn.Conv1d(33, 10, 3, stride=2)
        self.l4 = nn.MaxPool1d(3, stride=2)

        self.l5 = torch.nn.Linear(40, 10)
        self.l6 = torch.nn.Linear(10, 1)

        #save weight attributes in variable.
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

    def forward(self, x):
        x = x.reshape(1,62,82).float()
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = x.view(1,10*4)
        x = self.l5(x)
        x = self.l6(x)
        return torch.sigmoid(x)


'''_____________________________________________________________________________________________________________________________________________________'''

'''Classes to simulate and run the Game'''


#define the properties of the bird/ball
class Ball:
    def __init__(self,net):
        #initialize a new Ball and give it a neuronal net as a brain
        self.x = -300   #x-coordinate
        self.y = 0      #y-coordinate
        self.v = 0      #velocity negative velocity = upward movement!
        self.g = 1      #gravity
        self.s = 0      #score

        self.alive = True
        self.net = net

    def g_update(self):
        #when g_update is called, we numerically solve the equation d^2x/dt^2=g and set a new x position.
        #we set dt=1 for each iteration. To make the game playable one can put a delay in front of each iteration
        #In order to slow down the game we multiply the velocity by 0.9. This gives the game a more natural feel.
        self.v += self.g
        self.v *= 0.9
        self.y -= self.v

    def net_update(self,input):
        #net_update calls the foreward function of the neuronal net and makes an
        #action depending on the input
        if self.net(input)<0.5:
            self.v -= 40

    def get_parameters(self):
        #this function can be used to transfer Ball states to a drawing tool
        return self.x,self.y,self.v,self.g,self.s,self.alive

    def collision(self,pipe):
        #this function checks, whether the balls position overlaps with the position of a pipe or if the ball crashes on the floor
        if (pipe.x == self.x and pipe.y1>=self.y) or (pipe.x == self.x and pipe.y2<=self.y) or (self.y < -300):
            self.alive = False

    def enclose(self):
        #pass
        #Ball cannot escape window to the top and will bounce off
        if self.y>300:
            self.y = 300
            self.v = 8
        #if self.y<-300:
        #    self.y = -290

    def reset(self):
        #all parameters are resetted, in order to use it (with its net) for e new run.
        self.x = -300
        self.y = 0
        self.v = 0
        self.g = 1
        self.s = 0
        self.alive = True


#define the properties of the pipes
class Pipes:
    def __init__(self,xinit):
        #initialize a pipe object

        self.height1 = random.randint(1,44)
        self.height2 = 45-(self.height1)

        self.y1 = -300+self.height1*10
        self.y2 = 300-self.height2*10

        self.len1 = self.height1*10
        self.len2 = self.height2*10

        self.x = xinit
        self.v = 5

        self.x = xinit

    def update(self):
        #when update is called the pipes x-position changes by -v.
        self.x -= self.v

    def get_parameters(self):
        #this function can be used to draw the pipe object
        return self.x1,self.y1,self.y2




'''_____________________________________________________________________________________________________________________________________________________'''

'''The Genetic Algorithm'''
class GA:
    def __init__(self,nGenerations,nPopulation,Sigma_mut, binomial,max_net_depth,max_net_width,N_best, Net_type,train,dec_input, fix_shape):
        #initialize the parameters of the evolution. E.g number of epoches, nuumber of individuals in a generation
        self.nGenerations = nGenerations                    #number of generations used to evolve the net
        self.nPopulation = nPopulation                      #population per generation
        self.Sigma_mut = Sigma_mut                          #standard deviation used to evolve the nets
        self.binomial = binomial                            #percentage of parameters that are evolved
        self.max_net_depth = max_net_depth                  #maximum net depth = number of layers
        self.max_net_width = max_net_width                  #maximum net width = neurons per layer
        self.N_best = N_best                                #Number of surviving nets
        self.Net_type = Net_type

        #define variables that save the essential results of the evolution
        self.best_individual_generation = []                #saves the best individual net in each generation
        self.best_scores_generations = []                   #saves the best score of the above net
        self.number_of_best_individuals_generation = []     #saves the scores of all nets per generation
        self.last_generation = []                           #saves the last generations networks

        #initialize arrays that save training images
        if train==True:
            self.train = True
            self.xArray = torch.randn(1,62,82)
            self.yArray = torch.randn(1,1)
        else:
            self.train = False

        #initialize encoder to transform graphical input into dimension reducted data.
        if dec_input == True:
            self.dec_input = dec_input
            self.vae = torch.load('./vae.pt')
            self.N_input = 14
        else:
            self.dec_input = False
            self.N_input = 6

        #set up the initial population. One can create random variables for depth and width
        #in order to get a wider population->evolution may find optimal net depth and width.
        self.Balls = []
        if fix_shape == True:
            for i in range(self.nPopulation):
                N_depth = self.max_net_depth
                N_width = self.max_net_width
                self.Balls.append(Ball(Net_type(N_depth, N_width, self.N_input)))
        else:
            for i in range(self.nPopulation):
                N_depth = random.randint(1, self.max_net_depth)
                N_width = random.randint(1, self.max_net_width)
                self.Balls.append(Ball(Net_type(N_depth, N_width, self.N_input)))

    def main(self):
        self.convergence_count = 0
        #start the evolition
        for i in range(self.nGenerations):
            if self.convergence_count<5:
                if i%10 == 0:
                    print("Generation={}".format(i))

                #play the game
                if self.train == True:
                    print("train")
                    self.Balls,self.xArray,self.yArray = functions.game(self.Balls,self.train,None,self.xArray,self.yArray)
                elif self.dec_input == True:
                    print("dec_input")
                    self.Balls = functions.game(self.Balls,self.train,self.vae)
                else:
                    print("normal_Input")
                    self.Balls = functions.game(self.Balls,self.train)


                #get the scores of the balls
                scores = np.zeros(self.nPopulation)
                for i in range(self.nPopulation):
                    scores[i] = self.Balls[i].s

                #sort the Balls by their score
                ind = [x for _,x in sorted(zip(scores,np.arange(self.nPopulation)))]
                newBalls = []
                for i in ind:
                    newBalls.append(self.Balls[i])
                self.Balls = newBalls

                #save the essential results of this run
                self.best_scores_generations.append(max(scores))
                self.number_of_best_individuals_generation.append(scores)
                self.best_individual_generation.append(self.Balls[-1])

                #create new poputlation
                #copy N_best Balls and evolve the copies. Keep the previous best balls.
                for i in range(self.N_best):
                    self.Balls[-(self.N_best+i)].net = copy.deepcopy(self.Balls[-i]).net
                    self.Balls[-(self.N_best+i)].net.evolve(self.Sigma_mut, self.binomial)


                #Generate completely new Balls
                for i in range(self.nPopulation-2*self.N_best):
                    self.Balls[i].net = self.Net_type(self.max_net_depth,self.max_net_width, self.N_input)

                #reset the parameters to the init variables
                for i in self.Balls:
                    i.reset()

                #print some numbers
                print("Score of best Individual={}".format(self.best_scores_generations[-1]))
                print("standard-deviation={}".format(self.evolution_statistics()[1][-1]))

                if max(scores)>=7000:
                    self.convergence_count+=1
                else:
                    self.convergence_count=0

        #Save the last generation
        self.last_generation = self.Balls
        
    def evolution_parameters(self):
        #get the essential results of the evolution
        return [self.best_scores_generations,
                self.best_individual_generation,
                self.number_of_best_individuals_generation,
                self.last_generation]

    def evolution_statistics(self):
        #get variance and mean of the score in each generation
        mean_score_in_generation = [np.mean(x) for x in self.number_of_best_individuals_generation]
        stdv_score_in_generation = [np.var(x) for x in self.number_of_best_individuals_generation]
        return mean_score_in_generation, stdv_score_in_generation



'''_____________________________________________________________________________________________________________________________________________________'''

class Data(Dataset):
    #Class that stores the picture input. Can later be passed to the dataLoader in oder to train
    #the neuronal net of the VAE.
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    #class methods that are required by the dataloader.
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        sample = (self.X[idx,:], self.Y[idx])
        return sample, idx

    
#class to read out the data for each method (algorithm)
class method():
    def __init__(self, name):
        print(name)
        self.name = name
        self.scores, self.best_scores, self.mean_scores, self.std_scores = functions.load_data(self.name) 
        self.N_population = len(self.scores[0])
        self.last_gen = []
        for i in range(self.N_population):
            self.last_gen.append(torch.load(self.name+"/TorchSaves/Last_Gen"+str(i)+".pt"))

        #self.best_ball = self.last_gen[-1]