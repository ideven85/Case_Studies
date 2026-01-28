##### IMAGE REPRESENTATION WITH SIMILAR ABSTRACTIONS TO LAB 1 AND 2


def get_width(image):
    return image.get_width() // SCALE


def get_height(image):
    return image.get_height() // SCALE


def get_pixel(image, row, col):
    print(row, col)
    color = image.get_at((col * SCALE, row * SCALE))
    return (color.r, color.g, color.b)


def flood_fill(image, location, new_color):
    """
    Given an image, replace the same-colored region around a given location
    with a given color.  Returns None but mutates the original image to
    reflect the change.

    Parameters:
      * image: the image to operate on
      * location: an (row, col) tuple representing the starting location of the
                  flood-fill process
      * new_color: the replacement color, as an (r, g, b) tuple where all values
                   are between 0 and 255, inclusive
    """
    print(f"You clicked at row {location[0]} col {location[1]}")
    original_color = get_pixel(image, location[0], location[1])
    print(original_color)

    def set_pixel(image, row, col, color, visited=None):
        """
        Been a year, still don't know how this works!!
        """
        loc = row * SCALE, col * SCALE
        c = pygame.Color(*color)
        for i in range(SCALE):
            for j in range(SCALE):
                image.set_at((loc[1] + i, loc[0] + j), c)
                # visited.add((loc[0]+j,loc[1]+i))
                # image.set_at((loc[1] - i, loc[0] + j), c)
                # image.set_at((loc[1] - i, loc[0] - j), c)
                # image.set_at((loc[1] + i, loc[0] - j), c)
        ## comment out the two lines below to avoid redrawing the image every time
        ## we set a pixel
        screen.blit(image, (0, 0))
        pygame.display.flip()

    def get_neighbors(cell):
        row, col = cell

        potential_neighbors = [
            (row + 1, col),  # up
            # (row+1,col+1), # diagonal right down
            (row - 1, col),  # left
            # (row-1,col-1), # diagonal left up
            # (row-1,col+1), # diagonal right up
            (row, col + 1),  # right
            (row, col - 1),  # down
            # (row+1,col-1) # diagonal left down
        ]
        #     return [
        #     (nr, nc)
        #     for nr, nc in potential_neighbors
        #     if 0 < nr < get_height(image) and 0 <= nc < get_width(image)
        # ]
        return [
            (dx, dy)
            for dx, dy in potential_neighbors
            if 0 < dx < get_height(image) and 0 <= dy < get_width(image)
        ]

    print(get_neighbors((location[0], location[1])))
    to_color = [location]  # agenda: all of the cells we need to color in
    visited = {location}  # all pixels ever added to the agenda
    while to_color:
        # if len(to_color) == 1:
        # print(f"Length 1: {to_color}")
        this_cell = to_color.pop(0)
        set_pixel(image, *this_cell, new_color, visited)
        for neighbor in get_neighbors(this_cell):
            if (
                neighbor not in visited
                and get_pixel(image, *neighbor) == original_color
            ):
                to_color.append(neighbor)
                # print(len(to_color),end=' ')
                visited.add(neighbor)
            # print(to_color)
    # while to_color:
    #     this_cell = to_color.pop()
    print(visited)
    print(len(visited))
    # screen.blit(image, (, 0))
    # pygame.display.flip()


##### USER INTERFACE CODE
##### DISPLAY AN IMAGE AND CALL flood_fill WHEN THE IMAGE IS CLICKED

import os
import sys

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

from pygame.locals import *

COLORS = {
    pygame.K_r: (255, 0, 0),
    pygame.K_w: (255, 255, 255),
    pygame.K_k: (0, 0, 0),
    pygame.K_g: (0, 255, 0),
    pygame.K_b: (0, 0, 255),
    pygame.K_c: (0, 255, 255),
    pygame.K_y: (255, 230, 0),
    pygame.K_p: (179, 0, 199),
    pygame.K_o: (255, 77, 0),
    pygame.K_n: (66, 52, 0),
    pygame.K_e: (152, 152, 152),
}

COLOR_NAMES = {
    pygame.K_r: "red",
    pygame.K_w: "white",
    pygame.K_k: "black",
    pygame.K_g: "green",
    pygame.K_b: "blue",
    pygame.K_c: "cyan",
    pygame.K_y: "yellow",
    pygame.K_p: "purple",
    pygame.K_o: "orange",
    pygame.K_n: "brown",
    pygame.K_e: "grey",
}

SCALE = 7
IMAGE = "flood_input.png"

pygame.init()
image = pygame.image.load(IMAGE)
print(image.get_width(), image.get_height())
dims = (image.get_width() * SCALE, image.get_height() * SCALE)
screen = pygame.display.set_mode(dims)
image = pygame.transform.scale(image, dims)
screen.blit(image, (0, 0))
pygame.display.flip()
initial_color = pygame.K_b
cur_color = COLORS[initial_color]
print("current color:", COLOR_NAMES[initial_color])
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key in COLORS:
                cur_color = COLORS[event.key]
                print("current color:", COLOR_NAMES[event.key])
            elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            flood_fill(image, (event.pos[1] // SCALE, event.pos[0] // SCALE), cur_color)
            screen.blit(image, (0, 0))
            pygame.display.flip()
