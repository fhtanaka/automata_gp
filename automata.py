import numpy as np

def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input


class CA_2D_model:
    def __init__(self, length, width, update_function, layers=1, vision = 1, gps = False, value_limit = 1000):
        # self.action = toolbox.compile(individual)
        self.step = update_function
        self.len = length + 2*vision
        self.wid = width + 2*vision
        self.layers = layers
        self.vision = vision
        self.value_limit = value_limit
        self.gps = gps
        self.vision_size = (1+vision*2)**2

        # The size of the pad is dependent on how far each cell sees to updates its valus
        self.original = np.ones((layers, length+2*vision, width+2*vision))
        # make the center cell black
        self.original[0][int(self.len/2)][int(self.wid/2)] = 0

        self.ca = np.copy(self.original)

    def reset_ca(self):
        self.ca = np.copy(self.original)

    def get_observation(self, i, j):
        observation = self.ca[:, i-self.vision:i +
                              self.vision+1, j-self.vision:j+self.vision+1]
        return observation.reshape(-1)

    def new_cell_value(self, i, j):
        # checking ig it is a pad
        if i-self.vision < 0 or j-self.vision < 0:
            return np.ones(self.layers)
        if i+self.vision >= self.len or j + self.vision >= self.wid:
            return np.ones(self.layers)

        observation = self.get_observation(i, j)
        # checking if the cell is alive
        if observation[0:self.vision_size].sum() >= 1 * self.vision_size:
            return np.ones(self.layers)
        if self.gps:
            observation = np.append(observation, [i, j])
        value = self.step(observation)
        for i in range(len(value)):
            value[i] = round(value[i], 5)
            value[i] = limit(value[i], -1*self.value_limit, self.value_limit)
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

    def remove_pad(self):
        return self.ca[0, self.vision:self.len - self.vision, self.vision:self.wid - self.vision]

    def fitness(self, target_image):
        ca = self.remove_pad()
        if target_image.shape != ca.shape:
            raise
        loss = 0
        for i in range(target_image.shape[0]):
            for j in range(target_image.shape[1]):
                if ca[i, j] > 1 or ca[i, j] < 0:  # Checking if the cell is in the right interval
                    return -1000
                l = ca[i, j] - target_image[i, j]
                loss += l**2
        return 1 - loss
