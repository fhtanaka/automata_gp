from test_images import *
from config import *
import argparse

def parse_args():
    global GENS, N_TOTAL_STEPS, POPULATION, SAVETO, RENDER, APPLY_SOBEL_FILTER, TARGET_IMG

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', nargs='?', default=None,
                        help='The filename for a checkpoint file to restart from')

    parser.add_argument('-g', nargs='?', type=int, default=GENS,
                        help='number of generations')

    parser.add_argument('--sobel', type=str, nargs='?', default="false", help='')

    parser.add_argument('--img', nargs='?', default=None, help='')

    parser.add_argument('--steps', nargs='?', type=int,
                        default=N_TOTAL_STEPS, help='')

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
        # command_line_args.img = "Gray"
        TARGET_IMG = degrade_img()
