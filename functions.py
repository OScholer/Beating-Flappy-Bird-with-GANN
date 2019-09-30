'''
define the essential functions that are required: the game function play a whole game until all Balls have failed.
'''

import torch
import classes
import random
import time
#from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pickle

#game takes a list of Balls as input that each paly the game independentely of each other.
def game(Balls,train,vae=None,xArray=None,yArray=None):

    '''
    function game playes the game with all Balls in the List Balls until all have failed. During the game we
    keep track of the score and wether they have collided with the ground or with a pipe.
    Arguments:
        Balls: A list of Ball-classes that each represent one Ball in the game.
        train: if train == True, we fill xArray and yArray with the picture input from the game. Later these
               arrays can be passed to a encoder-decoder network in order to find the relevant features of the images input
               and use this as inputs to our decision neuronal network
               if train == False, we take the internal game input which consits of the speed and position of the pipes and balls.
        xArray: saves the picture input
        yArray: we save random input in this array. It will not be used, however the dataloader requires this as an argument.
    '''

    #initialite the first pipe
    pipes = []
    pipes.append(classes.Pipes(50))

    #set up game parameters: Game_count right now is the score of the balls
    Game_count = 0
    Game = True

    #after certain time we like to interupt the while loop with a time condition

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    start = time.time()

    #randomly generate and double the first input to give the network 2 pictures to start with
    last_input = torch.randn(1,7).double()


    while Game == True:
        ######################################################
        #update the Balls position and status (alive, failed)#
        ######################################################
        for i in Balls:
            #first we get the closest pipe to the balls
            closest_Pipe = closest_pipe(pipes)
            #we check wheter the balls have hit a pipe
            i.collision(pipes[0])
            #we check wheter the balls has escaped the screen
            i.enclose()
            #we update the y position by the effect of gravity
            i.g_update()
            #every 5 game loop we ask the "net" of the ball for a decision (for the balls that are alive)
            if i.alive == True:# and Game_count % 5 == 0:
                #get_NN_input gets the distance to the closest pipe, the height of the pipe, the speed of the pipe
                #and ghe ball and returns is as a tensor that can be passed to the net.

                """
                input to neuronal net is either game input, or encoded graphical input
                """
                if vae==None:
                    #print("not vae!!!")
                    input = get_NN_input(i,closest_Pipe)
                    #input = pixel_input(i,pipes)
                    i.net_update(input)
                else:
                    img = pixel_input(i,pipes)
                    #print("vae!!!")
                    means, log_var = vae.encoder((img.reshape(1,62*82)).to(device))
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([1, 7])        
                    eps = eps.to(device)        
                    z = eps * std + means
                    
                    z = z.detach().cpu().double()
                
                    i.net_update(torch.cat((z,last_input)))
                
                    last_Input = z.clone()

                #update score of alive balls
                i.s+=1

        ##################################################################################################
        #if training !=False, game images are stored in Array in order to train a encoder-decoder network#
        ##################################################################################################
        if train == True:
            if Game_count % 1 == 0 and Balls[-1].alive == True:
                img = pixel_input(Balls[-1],pipes)
                #X is of size (N,62,82) where N is the count of current training instances. The input has to
                #be of size (1,62,82).
                xArray = torch.cat((xArray,img))
                yArray = torch.cat((yArray,torch.randn(1,1)))


        #######################################
        #update the pipes positions and status#
        #######################################
        for i in pipes:
            i.update()
            #if the pipe has left the screen: remove it from the list
            if i.x < -400:
                #del i
                del pipes[0]
        #create new pipe
        if Game_count % 70 ==0:
            pipes.append(classes.Pipes(400))

        #############################################################################
        #update game_count and check wheter all balls have failed -> break game loop#
        #############################################################################
        Game_count +=1
        Game = check_Game(Balls)
        #stop after crossing 100 pipes
        if Game_count/70 > 101:
            Game = False
        #stop time and break game-loop if time exceeds 5 sec
        #stop = time.time()
        #if stop-start>100:
        #    Game = False

    #return Balls that have played the game
    if train == True:
        return Balls,xArray,yArray
    else:
        return Balls


def get_NN_input(Ball,closest_pipe):
    #gets all input parameters for the neuronal net
    v1 = Ball.y
    v2 = Ball.y-(closest_pipe.y1)
    v3 = Ball.y-(closest_pipe.y2)
    v4 = closest_pipe.x
    v5 = closest_pipe.v
    v6 = Ball.v
    return torch.tensor([v1,v2,v3,v4,v5,v6]).double()

def closest_pipe(pipes):
    #gets closest pipe
    closest_pipe = pipes[0]
    if pipes[0].x < -300:
        closest_pipe = pipes[1]
    return closest_pipe

def check_Game(Balls):
    #checkts wheter all balls are dead
    count = 0
    for i in Balls:
        if i.alive == False:
            count +=1
    if count == len(Balls):
        return False
    else:
        return True

def get_NN_input(Ball,closest_pipe):
    #gets all input parameters for the neuronal net
    v1 = Ball.y
    v2 = Ball.y-(closest_pipe.y1)
    v3 = Ball.y-(closest_pipe.y2)
    v4 = closest_pipe.x
    v5 = closest_pipe.v
    v6 = Ball.v
    return torch.tensor([v1,v2,v3,v4,v5,v6]).double()

def closest_pipe(pipes):
    #gets closest pipe
    closest_pipe = pipes[0]
    if pipes[0].x < -300:
        closest_pipe = pipes[1]
    return closest_pipe

def check_Game(Balls):
    #checkts wheter all balls are dead
    count = 0
    for i in Balls:
        if i.alive == False:
            count +=1
    if count == len(Balls):
        return False
    else:
        return True

def parallel_ball_update(Ball):
    pass

def parallel_pipe_update(Pipes):
    pass

def pixel_input(ball,pipes):
    #convert pipes and ball positions into a picture
    Pic = torch.zeros(62,82)
    for i in pipes :
        y1 = int(i.len1/10)
        y2 = int(i.len2/10)
        x  = int((i.x + 410)/10)
        Pic[61-y2:,(x-1):(x+1)] = 1
        Pic[:y1,(x-1):(x+1)] = 1
    xb = int((ball.x+410)/10)
    yb = int((ball.y+310)/10)
    Pic[(yb-1):(yb+1),(xb-1):(xb+1)]=1
    return Pic.view(1,62,-1)

def show_image(mat):
    #show picture
    mat = mat.numpy()
    mat.reshape(62,82)
    plt.imshow(mat, cmap="gray")
    plt.show()



#save classes
def save_class(some_class, savename):
    file = open("SavedClasses/"+savename+".dictionary", "wb")
    pickle.dump(some_class, file)
    file.close()
    
def load_class(some_class, savename):
    file = open("SavedClasses/"+savename+".dictionary", "r")
    some_class = pickle.load(file)
    file.close()
    

def savetxts(data, filenames):
    for i in range(len(data)):
        np.savetxt("SavedScores/"+filenames[i]+"txt", data[i])


#load the saved data from the algorithms runs
def load_data(method):
    scores = np.loadtxt(method+"/SavedScores/all_scorestxt")
    scores.sort()
    best_scores = np.zeros(len(scores))
    mean_scores = np.zeros(len(scores))
    std_scores = np.zeros(len(scores))
    for idx in range(len(scores)):
        best_scores[idx] = scores[idx][-1]
        #take the mean score and the std of the best 450 balls
        mean_scores[idx] = np.mean(scores[idx][550:])
        std_scores[idx] = np.std(scores[idx][550:])
        
    return(scores, best_scores, mean_scores, std_scores)