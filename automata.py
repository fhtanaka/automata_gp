import numpy as np

BLACK = 0
WHITE = 255

def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input


class CA_2D_model:
    def __init__(self, length, width, update_function, layers=1, vision = 1, gps = False, value_range = [0,255], integer_only=True):
        # self.action = toolbox.compile(individual)
        self.step = update_function
        self.len = length + 2*vision
        self.wid = width + 2*vision
        self.layers = layers
        self.vision = vision
        self.value_range = value_range
        self.gps = gps
        self.vision_size = (1+vision*2)**2

        # The size of the pad is dependent on how far each cell sees to updates its valus
        self.original = np.full((layers, length+2*vision, width+2*vision), WHITE)

        # make the center cell black
        self.original[0][int(self.len/2)][int(self.wid/2)] = BLACK

        self.ca = np.copy(self.original)

    def reset_ca(self):
        self.ca = np.copy(self.original)

    def get_observation(self, i, j):
        observation = self.ca[:, i-self.vision:i + self.vision+1, j-self.vision:j+self.vision+1]
        return observation.reshape(-1)

    def new_cell_value(self, i, j):
        # checking if it is a pad
        if i-self.vision < 0 or j-self.vision < 0:
            return np.full(self.layers, WHITE)
        if i+self.vision >= self.len or j + self.vision >= self.wid:
            return np.full(self.layers, WHITE)

        observation = self.get_observation(i, j)

        # checking if the cell is alive
        if observation[:].sum() >= WHITE * self.vision_size:
            return np.full(self.layers, WHITE)

        if self.gps:
            observation = np.append(observation, [i, j])

        value = self.step(observation)
        
        for i in range(len(value)):
            value[i] = limit(value[i], self.value_range[0], self.value_range[1])
        
        return value

    def update(self):
        new_ca = np.copy(self.ca)
        for i in range(self.vision, self.len - self.vision):  # skipping pad
            for j in range(self.vision, self.wid - self.vision):  # skipping pad
                new_values = self.new_cell_value(i, j)
                for l in range(self.layers):
                    new_ca[l, i, j] = new_values[l]
        
        if (new_ca == self.ca).all():  # checking if the cell updated or not
            return False
        
        self.ca = new_ca
        return True

    def remove_pad(self, layer = 0):
        return self.ca[layer, self.vision:self.len - self.vision, self.vision:self.wid - self.vision]

    def fitness(self, target_image):
        ca = self.remove_pad()
        if target_image.shape != ca.shape:
            raise
        loss = 0
        for i in range(target_image.shape[0]):
            for j in range(target_image.shape[1]):
                # Checking if the cell is in the right interval
                if ca[i, j] < self.value_range[0] or ca[i, j] > self.value_range[1]:
                    return 1000
                l = ca[i, j] - target_image[i, j]
                loss += l**2
        return loss
