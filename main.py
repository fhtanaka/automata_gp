import numpy as np
from cgp.graph import Graph
from cgp.population import Population
from automata import CA_2D_model
from config import *
import test_images
from arg_parser import parse_args

input_size = (VISION+2)**2
if APPLY_SOBEL_FILTER:
    input_size *= 3

############################################ Node Custom Operations ################################################

addition = lambda x, y: x+y
multiplication = lambda x, y: x*y
subtraction = lambda x, y: x-y
constant = lambda x: x
protected_div = lambda x, y: 1 if y == 0 else x/y
increment = lambda x: x+1
invert = lambda x: -x

seed = 2002
Graph.rng = np.random.RandomState(seed)

Population.add_operation(arity=1, func=lambda x: 1, string="1")
Population.add_operation(arity=1, func=lambda x: .5, string="0.5")
Population.add_operation(arity=1, func=lambda x: .1, string="0.1")
Population.add_operation(arity=1, func=constant, string="x")
Population.add_operation(arity=1, func=increment, string="x+1")
Population.add_operation(arity=1, func=invert, string="-x")
Population.add_operation(arity=2, func=addition, string="x+y")
Population.add_operation(arity=2, func=multiplication, string="x*y")
Population.add_operation(arity=2, func=subtraction, string="x-y")
Population.add_operation(arity=2, func=protected_div, string="*x/y")

def eval_individual(individual: Graph, target_image, layers, render=False):
    shape = target_image.shape
    ca = CA_2D_model(shape[0], shape[1], individual.operate, layers)
    
    total_fitness = 0.0
    for i in range(TESTS_FOR_EACH_TREE):
        ca.reset_ca()

        for _ in range(N_TOTAL_STEPS):
            if render:
                test_images.print_img(ca.remove_pad())
                print(ca.fitness(target_image))
            update = ca.update()
            if not update: # the automata got stable
                break

        total_fitness += ca.fitness(target_image)  

    fitness = (total_fitness / TESTS_FOR_EACH_TREE) 
    return fitness


def main():

    TARGET_IMG = test_images.degrade_img()
    # test_images.print_img(TARGET_IMG)
    total_layers = 1 + EXTRA_LAYERS
    population = Population(
        population_size=POPULATION,
        n_in=total_layers*input_size,
        n_out=total_layers,
        n_row=10,
        n_col=4,
        levels_back=5,
        mutation_strategy="prob",
        fitness_func=lambda x: eval_individual(x, TARGET_IMG, total_layers),
        minimize_fitness=False,
        point_mut_qnt=10,
        prob_mut_chance=.15,
        mutate_active_only=False
    )
    population.one_plus_lamda(100, 2, 0, True)
    # return pop, log, hof
    for ind in population.indvs:
        eval_individual(ind, TARGET_IMG, total_layers, True)
if __name__ == "__main__":
    main()

