import numpy as np
import pygraphviz as pgv
import matplotlib.pyplot as plt
import imageio
import pickle
import argparse
from cgp.graph import Graph
from cgp.population import Population
# import warnings
# warnings.simplefilter('ignore') 
############################################ Parameters ################################################
TARGET_EMOJI = 0 #@param "🦎"
MAX_HEIGHT = 15
POPULATION = 400
APPLY_SOBEL_FILTER = False
VISION = 1
TESTS_FOR_EACH_TREE = 1
N_TOTAL_STEPS = 100
GENS = 30
TARGET_IMG = np.full((25,25), .5)
SAVETO = None
RENDER = False
LIMIT = 10000
GPS_ON_CELL = True
EXTRA_LAYERS = 3

############################################ Test Images ################################################
def degrade_img():
    img = np.ones((25,25))
    for i in range(25):
        img[i][:] = 1 - (i*4)/100
    return img

def column_img():
    img = np.ones((25,25))
    for i in range(25):
        for j in range(25):
            if j == 12:
                img[i][j] = 1 - (i*4)/100
    return img

def plus_img():
    img = np.ones((25,25))
    for i in range(25):
        for j in range(25):
            if j == 12 or i == 12:
                img[i][j] = 0
    return img

def x_img():
    img = np.ones((25,25))
    for i in range(25):
        for j in range(25):
            if j == i or i + j == 24:
                img[i][j] = 0
    return img

def diagonal_img():
    img = np.ones((25,25))
    for i in range(25):
        for j in range(25):
            if i + j == 24:
                img[i][j] = 0
    return img

def inv_diagonal_img():
    img = np.ones((25,25))
    for i in range(25):
        for j in range(25):
            if i == j:
                img[i][j] = 0
    return img

def print_img(img):
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()

############################################ Image Convertion functions ################################################
def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)

def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)

def to_gray(x):
    rgb = to_rgb(x)
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def load_emoji(index, path="data/emoji.png", size=40):
    im = imageio.imread(path)
    emoji = np.array(im[:, index*size:(index+1)*size].astype(np.float32))
    emoji /= 255.0
    gray_emoji = to_gray(emoji)
    return gray_emoji

def plot_loss(loss_log):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.1)
    plt.show()

def create_base_env(target_image):
    env = np.copy(target_image)
    env = np.pad(np.copy(target_image), 1)
    env[:] = 1
    a, b = env.shape
    env[int(a/2)][int(b/2)] = 0

############################################ Parser args ################################################
parser = argparse.ArgumentParser()

parser.add_argument('-s', nargs='?', default=None,
                    help='The filename for a checkpoint file to restart from')

parser.add_argument('-g', nargs='?', type=int, default=GENS, help='number of generations')

parser.add_argument('--sobel', type=str, nargs='?', default="false", help='')

parser.add_argument('--img', nargs='?', default=None, help='')

parser.add_argument('--steps', nargs='?', type=int, default=N_TOTAL_STEPS, help='')

parser.add_argument('--render', nargs='?', type=str, default="false", help='')

parser.add_argument('--pop', nargs='?', type=int, default=POPULATION, help='')

command_line_args = parser.parse_args()

GENS = command_line_args.g
SAVETO = command_line_args.s
N_TOTAL_STEPS = command_line_args.steps
POPULATION = command_line_args.pop

if command_line_args.render == "true" or command_line_args.render == "True":
    RENDER = True

if command_line_args.sobel == "true" or command_line_args.sobel == "True":
    APPLY_SOBEL_FILTER = True

if command_line_args.img is not None:
    if command_line_args.img == "stick":
        TARGET_IMG = load_emoji(0, "data/stick.png", 25)
    elif command_line_args.img == "brazil":
        TARGET_IMG = load_emoji(0, "data/brazil.png", 25)
    elif command_line_args.img == "column":
        TARGET_IMG = column_img()
    elif command_line_args.img == "plus":
        TARGET_IMG = plus_img()
    elif command_line_args.img == "degrade":
        TARGET_IMG = degrade_img()
    elif command_line_args.img == "x":
        TARGET_IMG = x_img()
    elif command_line_args.img == "diagonal":
        TARGET_IMG = diagonal_img()
    elif command_line_args.img == "inv_diagonal":
        TARGET_IMG = inv_diagonal_img()
    else:
        TARGET_IMG = load_emoji(command_line_args.img)
else:
    command_line_args.img = "Gray"

############################################ Automata  ################################################
sobel_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
sobel_y = np.transpose(sobel_x)
def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input
class CA_2D_model:
    def __init__(self, length, width, update_function, layers = 1, *, vision=VISION):
        # self.action = toolbox.compile(individual)
        self.step = update_function
        self.len = length + 2*vision
        self.wid = width + 2*vision
        self.layers = layers 
        self.vision = vision
        self.vision_size = (1+VISION*2)**2

        # The size of the pad is dependent on how far each cell sees to updates its valus
        self.original = np.pad(np.zeros((layers, length, width)), vision)
        self.original[:] = 1 # make all cells white
        self.original[int(self.len/2)][int(self.wid/2)] = 0 # make the center cell black

        self.ca = np.copy(self.original)
    
    def reset_ca(self):
        self.ca = np.copy(self.original)

    def get_observation(self, i, j):
        observation = self.ca[:, i-self.vision:i+self.vision+1, j-self.vision:j+self.vision+1]
        return observation.reshape(-1)

    def new_cell_value(self, i, j):
        # checking ig it is a pad
        if i-self.vision < 0 or j-self.vision < 0:
            return 1
        if i+self.vision >= self.len or j + self.vision >= self.wid:
            return 1

        observation = self.get_observation(i, j)
        if observation[0:self.vision_size].sum() >= 1 * self.vision_size: # checking if the cell is alive
            return 1
        if GPS_ON_CELL:
            observation = np.append(observation, [i,j])
        value = self.step(observation)
        for i in range(len(value)):  
            value[i] = round(value[i], 5)
            value[i] = limit(value[i], -1*LIMIT, LIMIT)
        return value

    def update(self):
        new_ca = np.copy(self.ca)
        for i in range(self.vision, self.len - self.vision): # skipping pad
            for j in range(self.vision, self.wid - self.vision): # skipping pad
                new_values = self.new_cell_value(i, j)
                for l in self.layers:
                    new_ca[l, i, j] = new_values[l]
        if (new_ca == self.ca).all(): # checking if the cell updated or not
            return False
        self.ca = new_ca
        return True

    def remove_pad(self):
        return self.ca[self.vision:self.len - self.vision, self.vision:self.wid - self.vision]

    def fitness(self, target_image):
        ca = self.remove_pad()
        if target_image.shape != ca.shape:
            raise
        loss = 0
        for i in range(target_image.shape[0]):
            for j in range(target_image.shape[1]):
                if ca[i,j] > 1 or ca[i,j] < 0: # Checking if the cell is in the right interval
                    return -1000
                l = ca[i,j] - target_image[i,j]
                loss += l**2
        return 1 - loss

############################################ Creating GP and image ################################################
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
                print_img(ca.remove_pad())
                print(ca.fitness(target_image))
            update = ca.update()
            if not update: # the automata got stable
                break

        total_fitness += ca.fitness(target_image)  

    fitness = (total_fitness / TESTS_FOR_EACH_TREE) 
    return fitness

# def fitness_func(individual: Graph, target_image):

#     fitness = 0
#     for t in tests:
#         pred1, pred2 = individual.operate([t[0][0], t[0][1]])
#         fitness += (t[1][0] - pred1)**2 + (t[1][1] - pred2)**2

#         # pred1 = individual.operate(t[0])[0]
#         # fitness += (t[1][0] - pred1)**2
#     return  fitness

############################################ Main ################################################

def avg(aux):
    return round(np.mean(aux), 1)

def main():

    
    print("TARGET_IMG: ",  command_line_args.img)
    print("POPULATION: ", POPULATION)
    print("GENS: ", GENS)
    print("MAX_HEIGHT: ", MAX_HEIGHT)
    print("TESTS_FOR_EACH_TREE: ", TESTS_FOR_EACH_TREE)
    print("N_TOTAL_STEPS: ", N_TOTAL_STEPS)
    print("APPLY_SOBEL_FILTER: ", APPLY_SOBEL_FILTER)
    print("VISION: ", VISION)
    print("SAVETO: ", SAVETO)
    print("RENDER: ", RENDER)
    print()

    population = Population (
        population_size = Population,
        n_in = EXTRA_LAYERS*input_size,
        n_out = 1+EXTRA_LAYERS,
        n_row = 8,
        n_col = 8,
        levels_back = 3,
        mutation_strategy = "prob",
        fitness_func = lambda x: eval_individual(x, TARGET_IMG, 1+EXTRA_LAYERS),
        minimize_fitness = False,
        point_mut_qnt = 10,
        prob_mut_chance = .1,
        mutate_active_only = False
    )

    # pop = toolbox.population(n=POPULATION)
    # hof = tools.HallOfFame(5)

    # stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    # # stats_size = tools.Statistics(len)
    # mstats = tools.MultiStatistics(fitness=stats_fit)
    # mstats.register("avg", avg)
    # mstats.register("std", np.std)
    # mstats.register("min", np.min)
    # mstats.register("max", np.max)

    # if RENDER:
    #     print_img(TARGET_IMG)

    # pop, log = algorithms.eaSimple(pop, toolbox, 0.9, 0.3, GENS, stats=mstats,
    #                                halloffame=hof, verbose=True)
    # if RENDER:
    #     print(hof[0].fitness)
    #     save_graph(hof[0], command_line_args.img)
    #     fit = eval_individual(hof[0], True)
    #     print(fit)

    # if SAVETO is not None:
    #     with open(SAVETO, "wb") as cp_file:
    #         cp = dict(
    #             population=pop, 
    #             halloffame=hof,
    #             logbook=log)
    #         pickle.dump(cp, cp_file)

    # return pop, log, hof

if __name__ == "__main__":
    main()

