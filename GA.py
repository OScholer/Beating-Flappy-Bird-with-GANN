import classes
import show_game
import show_classes
import functions as f
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import Encoder_Decoder as ed
from torch.autograd import Variable
from torchvision.utils import save_image
import torch

#initialize GA
N_generations = 1000
N_population = 1000
var_of_mutation = 1
binomial = 0.4
max_net_width = 5
max_net_lenght = 10
balls_to_duplicate = 450
training = False
dec_input = False
fix_shape = True
Net_type = classes.Net


print("Start Training:")
#populate __init__ method with starting parameters
new = classes.GA(N_generations, N_population, var_of_mutation, binomial, max_net_width, max_net_lenght, balls_to_duplicate,
                 Net_type, training, dec_input, fix_shape)
#start GA
new.main()
print("Training Finished")

score, ind, all_scores, last_gen  = new.evolution_parameters()

print("Saving Data")
#save important data
f.savetxts([score, all_scores], ["best_scores_per_generation", "all_scores"])
for i in range(len(ind)):
    torch.save(ind[i].net, "TorchSaves/Best_Ball_Gen"+str(i)+".pt")
for i in range(len(last_gen)):
    torch.save(last_gen[i].net, "TorchSaves/Last_Gen"+str(i)+".pt")
#f.save_class(last_gen, "last_gen")
best_Ball = last_gen[-1]
torch.save(best_Ball.net, "TorchSaves/Best_Ball.pt")


#get parameters of Evolution
#score, ind, all_scores, last_gen  = new.evolution_parameters()
#np.savetxt("best_scores.txt", score)
#np.savetxt("best_individual.txt", ind)
#np.savetxt("all_scores.txt", all_scores)
#np.savetxt("last_gen.txt", last_gen)
#np.savetxt("best_ball.txt", [show_classes.Ball(ind[-1].net)])
index = score.index(max(score))
best_ind = ind[index]


#print(new.evolution_parameters())
#print(new.evolution_statistics())

#play game 20 times with best Ball of evolution
for i in range(200):
    best_Ball = [show_classes.Ball(ind[-1].net)]
    show_game.main(1,best_Ball)
