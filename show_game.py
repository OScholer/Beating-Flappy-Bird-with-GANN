import turtle
import random
import torch
import show_classes as c
import numpy as np
import copy
import time


def main(nBalls,kBalls, time_limit, limit, jump):
    start=time.time()
    delay = 1/30
    window = turtle.Screen()
    window.title("game")
    window.bgcolor("black")
    window.setup(width=800,height=600)
    window.tracer(0)

    Pipes = []

    Balls = kBalls

    window.listen()
    #window.onkeypress(Balls[0].Up,"w")

    count = 0

    Pipes.append(c.Pipe(50,-300))

    Game = True
    while Game == True:
        now = time.time()
        if now-start>time_limit:
            Game = False
        time.sleep(delay)

        window.update()


        counter = 0
        for i in Balls:


            closet_pipe = Pipes[0]
            if Pipes[0].getx()< -300:
                closet_pipe = Pipes[1]

            i.check_for_hit(Pipes[0])
            i.move()

            r = i.gety()
            f = i.gety()-(closet_pipe.y1)
            g = i.gety()-(closet_pipe.y2)
            h = closet_pipe.getx()
            j = closet_pipe.velocity
            k = i.velocity

            input=torch.tensor([r,f,g,h,j,k]).double()

            '''
            v1 = Ball.y
            v2 = Ball.y-(closest_pipe.y1)
            v3 = Ball.y-(closest_pipe.y2)
            v4 = closest_pipe.x
            v5 = closest_pipe.v
            v6 = Ball.v
            '''

            if count%1==0:
                a = i.brain_prediction(input)
                if a<0.5:
                    i.velocity -= jump

            if i.alive == True:
                i.score = count

            #if i.gety()<-300:
            #    i.sety(-290)
            if limit == True:
                if i.gety()>300:
                    i.sety(300)
                    i.velocity = 8


            if i.alive == False:
                counter +=1
                i.update_color()

            if counter == nBalls:
                Game = False

        if count % 70 == 0:
            Pipes.append(c.Pipe(400,-300))

        for i in Pipes:
            i.move()

            if i.getx() < -400:
                i.pipe.reset()
                i.pipe1.reset()
                del i
                del Pipes[0]

        count = count+1
    for i in Balls:
        i.ball.reset()
        del i
    del Balls
    window.clear()
    window.bye()
    return True

