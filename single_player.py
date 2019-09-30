import turtle
import random
import torch
import show_classes as c
import numpy as np
import copy
import time
    
def jump(Ball, v):
    Ball.velocity += v

def single_main():
    print("Please chose a level from 1, 2 and 3")
    level = int(input())
    if level == 1:
        v = 8
        limit = True
    elif level == 2:
        v = 40
        limit = True
    elif level == 3:
        v = 40
        limit = False
    window = turtle.Screen()
    turtle.getscreen()._root.attributes('-topmost', 1)
    turtle.getscreen()._root.attributes('-topmost', 0)
    window.title("game")
    window.bgcolor("black")
    window.setup(width=800,height=600)
    window.tracer(0)

    Pipes = []


    Balls =  []
    Balls.append(c.Ball(1))

    window.listen()
    if level == 1:
        window.onkeypress(Balls[0].Up,"w")
        window.onkeypress(Balls[0].Up,"w")
    else:
        window.onkeypress(Balls[0].Up_2,"w")

    count = 0
    
    Pipes.append(c.Pipe(50,-300))

    Game = True
    while Game == True:
        time.sleep(1/30)
        window.update()
       
        if count % 70 == 0:
            Pipes.append(c.Pipe(400,-300))
       
        counter = 0
        for i in Balls:
            i.move()
            
            if i.alive == True:
                i.score = count
                
            if limit == True:
                if i.gety()>300:
                    i.sety(300)
                    i.velocity = 8
            if i.gety()<-300:
                print("I Fell Down")
                Game = False
              
            i.check_for_hit(Pipes[0])
            
            if i.alive == False:
                counter +=1
                i.update_color()
            
            if counter == 1:
                print("You fucked up")
                Game = False
                
        for i in Pipes:
            i.move()
               
            if i.getx() < -400:
                i.pipe.reset()
                i.pipe1.reset()
                del i
                del Pipes[0]
                
        count = count+1
        
    window.clear()
    window.bye()
    del Balls
    return True

if __name__=="__main__":
    single_main()