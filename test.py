import numpy as np
from cgp.graph import Graph
from cgp.population import Population
from automata import CA_2D_model
from bw_images import *
from time import time



def test_image(target_img, test_name, fit_func, report_interval = 10):
    print("******************************** ", test_name," ****************************")
    # def fit_func(x):
    #     return eval_individual(x, target_img, total_layers)

    # test_images.print_img(TARGET_IMG)
    total_layers = 1 + EXTRA_LAYERS
    input_size = (VISION+2)**2

    population = Population(
        population_size=POPULATION,
        n_in=total_layers*input_size,
        n_out=total_layers,
        n_row=2,
        n_col=4,
        levels_back=5,
        mutation_strategy="point",
        fitness_func=fit_func,
        minimize_fitness=True,
        point_mut_qnt=8,
        prob_mut_chance=.25,
        mutate_active_only=False
    )

    start = time()
    population.one_plus_lamda(GENS, GEN_CHAMPIONS, FITNESS_GOAL, report=report_interval)
    end = time()
    print("Execution time: ", end - start)
    population.save_pop(test_name+"3.pkl")

POPULATION = 256
APPLY_SOBEL_FILTER = False
VISION = 1
N_TOTAL_STEPS = 20
GENS = 500
SAVETO = None
RENDER = False
EXTRA_LAYERS = 0
GEN_CHAMPIONS = 128
FITNESS_GOAL = 0

img_size = 15
report_interval=10

input_size = (VISION+2)**2
if APPLY_SOBEL_FILTER:
    input_size *= 3

seed = 1997
Graph.rng = np.random.RandomState(seed)

BLACK = 255
WHITE = 0

def op_and(*args):
    for inp in args:
        if inp == WHITE:
            return WHITE
    return BLACK

def op_or(*args):
    for inp in args:
        if inp == BLACK:
            return BLACK
    return WHITE

def op_xor(*args):
    cont = 0
    for inp in args:
        if inp == BLACK:
            cont += 1
    if cont == 0 or cont == len(args):
        return WHITE
    return BLACK

def op_nand(*args):
    for inp in args:
        if inp == WHITE:
            return BLACK
    return WHITE

def op_nor(*args):
    for inp in args:
        if inp == BLACK:
            return WHITE
    return BLACK

def op_not(x):
    if x == WHITE:
        return BLACK
    else:
        return WHITE

def cte(x):
    return x

for i in range(2, 9):
    Population.add_operation(arity=i, func=op_and, string=("and"+str(i)))
    Population.add_operation(arity=i, func=op_or, string=("or"+str(i)))
    Population.add_operation(arity=i, func=op_xor, string=("xor"+str(i)))
    Population.add_operation(arity=i, func=op_nand, string=("nand"+str(i)))
    Population.add_operation(arity=i, func=op_nor, string=("nor"+str(i)))
Population.add_operation(arity=1, func=op_not, string="not")
Population.add_operation(arity=1, func=cte, string="cte")

def eval_individual(individual: Graph, target_image, layers, render=False):
    shape = target_image.shape
    ca = CA_2D_model(shape[0], shape[1], individual.operate, layers)
    
    total_fitness = 0.0
    ca.reset_ca()

    for _ in range(N_TOTAL_STEPS):
        update = ca.update()
        if not update: # the automata got stable
            break
    if ca.ca[0, :, :].sum() >= ca.len*ca.len*255:
        return 1000
    total_fitness += ca.fitness(target_image)/(255*255)

    fitness = (total_fitness) 
    return fitness

total_layers = 1 + EXTRA_LAYERS

def fit_func_one_color_img(x):
    return eval_individual(x, one_color_img(img_size, 0), total_layers)
test_image(one_color_img(img_size, 0), "one_color", fit_func_one_color_img, report_interval)

def fit_func_checkboard_img(x):
    return eval_individual(x, checkboard_img(img_size), total_layers)
test_image(checkboard_img(img_size), "checkboard", fit_func_checkboard_img, report_interval)

def fit_func_plus_img(x):
    return eval_individual(x, plus_img(img_size), total_layers)
test_image(plus_img(img_size), "plus", fit_func_plus_img, report_interval)

def fit_func_fat_plus_img(x):
    return eval_individual(x, fat_plus_img(img_size), total_layers)
test_image(fat_plus_img(img_size), "fat_plus", fit_func_fat_plus_img, report_interval)

def fit_func_x_img(x):
    return eval_individual(x, x_img(img_size), total_layers)
test_image(x_img(img_size), "x", fit_func_x_img, report_interval)
