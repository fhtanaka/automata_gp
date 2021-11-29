import numpy as np
from cgp.graph import Graph
from cgp.population import Population
from automata import CA_2D_model
from bw_images import *
from arg_parser import parse_args
from time import sleep, time
from ipycanvas import Canvas, hold_canvas
import dill

POPULATION = 32
APPLY_SOBEL_FILTER = False
VISION = 1
N_TOTAL_STEPS = 25
GENS = 100000
SAVETO = None
RENDER = False
EXTRA_LAYERS = 1
GEN_CHAMPIONS = 4
FITNESS_GOAL = 0

img_size = 15
TARGET_IMG = plus_img(img_size)

input_size = (VISION+2)**2
if APPLY_SOBEL_FILTER:
    input_size *= 3

seed = 2002
Graph.rng = np.random.RandomState(seed)


BLACK = 255
WHITE = 0

def op_and(x, y):
    if y == BLACK and x == BLACK:
        return BLACK
    else:
        return WHITE

def op_or(x, y):
    if y == BLACK or x == BLACK:
        return BLACK
    else:
        return WHITE

def op_xor(x, y):
    if (y == BLACK or x == BLACK) and x != y:
        return BLACK
    else:
        return WHITE

def op_nand(x, y):
    if y == BLACK and x == BLACK:
        return WHITE
    else:
        return BLACK

def op_nor(x, y):
    if y == BLACK or x == BLACK:
        return WHITE
    else:
        return BLACK

def op_not(x):
    if x == WHITE:
        return BLACK
    else:
        return WHITE

def cte(x):
    return x

Population.add_operation(arity=2, func=op_and, string="and")
Population.add_operation(arity=2, func=op_or, string="or")
Population.add_operation(arity=2, func=op_xor, string="xor")
Population.add_operation(arity=2, func=op_nand, string="nand")
Population.add_operation(arity=2, func=op_nor, string="nor")
Population.add_operation(arity=1, func=op_not, string="not")
Population.add_operation(arity=1, func=cte, string="cte")

def eval_individual(individual: Graph, target_image, layers, render=False):
    shape = target_image.shape
    ca = CA_2D_model(shape[0], shape[1], individual.operate, layers)
    
    total_fitness = 0.0
    ca.reset_ca()

    for _ in range(N_TOTAL_STEPS):
        if render:
            test_images.print_img(ca.remove_pad())
            print(ca.fitness(target_image))
        update = ca.update()
        if not update: # the automata got stable
            break
    if ca.ca[0, :, :].sum() >= ca.len*ca.len*255:
        return 1000
    total_fitness += ca.fitness(target_image)/(255*255)

    fitness = (total_fitness) 
    return fitness

def fit_func(x):
    return eval_individual(x, TARGET_IMG, total_layers)

# test_images.print_img(TARGET_IMG)
total_layers = 1 + EXTRA_LAYERS
population = Population(
    population_size=POPULATION,
    n_in=total_layers*input_size,
    n_out=total_layers,
    n_row=8,
    n_col=4,
    levels_back=5,
    mutation_strategy="prob",
    fitness_func=fit_func,
    minimize_fitness=True,
    point_mut_qnt=10,
    prob_mut_chance=.2,
    mutate_active_only=False
)


start = time()
population.one_plus_lamda(20000, GEN_CHAMPIONS, FITNESS_GOAL, report = 100)
end = time()
print("Execution time: ", end - start)

population.save_pop("plus.pkl")