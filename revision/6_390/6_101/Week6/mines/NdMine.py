import random
import numpy as np


class CoordND1:
    def __init__(self, dimensions, coordinates):
        self.dimensions = dimensions
        self.coordinates = coordinates


# def index_in_nD(n_d, n, coordinates):
#     return np.sum(np.prod(np.indices(n)[np.indices(n)[:, :, np.newaxis]
# == coordinates], axis=-1), axis=-1)


class Cell:
    def __init__(self, value=0):
        self.value = value
        self.is_mined = False
        self.adjacent_mines = 0


class Minefield:
    def __init__(self, width, height, mines):
        self.width = width
        self.height = height
        self.cells = [[Cell() for _ in range(height)] for _ in range(width)]
        self.mines = mines
        self.place_mines()
        self.reveal_adjacent_mined_cells()

    def place_mines(self):
        for i in range(self.mines):
            x, y = random.sample(range(self.width * self.height), 2)
            self.cells[x][y].is_mined = True

    def reveal_adjacent_mined_cells(self):
        for i in range(self.width):
            for j in range(self.height):
                if not self.cells[i][j].is_mined:
                    self.reveal_adjacent_mines(i, j)

    def is_valid_cell(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def reveal_adjacent_mines(self, x, y):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (
                    self.is_valid_cell(x + dx, y + dy)
                    and self.cells[x + dx][y + dy].value > 0
                ):
                    self.cells[x + dx][y + dy].adjacent_mines += 1
                    if self.cells[x + dx][y + dy].value == self.mines:
                        self.cells[x][y].is_revealed = True


class MinefieldND1:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.mine_count = np.prod(dimensions)
        self.cells = np.zeros((np.prod(dimensions), len(dimensions)))
        self.place_mines()
        self.reveal_adjacent_mined_cells()

    # def place_mines(self):
    #     mines = set()
    #     for mine in self.revealable_cells():
    #         for d in range(-1, 2):
    #             for dimension in reversed(self.dimensions):
    #                 c = coordinates[np.where([i == dim for i in
    #                                           np.arange(len(coordinates))])][0] + d
    #                 if (d == 0 or abs(c) >= len(dimension)):
    #                     adjacently_mined_cell = index_in_nD(*([coordinates] +
    #                                                           [(c,)]), *self.dimensions)
    #                     if np.any(self.cells[adjacently_mined_cell]):
    #                         self.reveal_cell(*coordinates)
    #                         break
    #                 if (d >= 0 and c < 0 or d < 0 and c >=
    #                         dim[dim.index(len(dim) - 1)]):
    #                     continue
    #
    #                 adjacently_mined_cell = index_in_nD(*([coordinates] + [(c,)]),
    #                                                     *self.dimensions)
    #                 if np.any(self.cells[adjacently_mined_cell]):
    #                     self.reveal_cell(*coordinates)
    #                     break

    def reveal_adjacent_mined_cells(self):
        for mine in self.revealable_cells():
            self.reveal_adjacent_mines(*mine)

    def reveal_adjacent_mines(self, *coordinates):
        cell_index = index_in_nD(*coordinates)
        cell = self.cells[cell_index]
        if np.all(cell == 0):
            for d in range(-1, 2):
                for dimension in reversed(self.dimensions):
                    c = (
                        coordinates[
                            np.where(
                                [i == dimension for i in np.arange(len(coordinates))]
                            )
                        ][0]
                        + d
                    )
                    if (
                        d == 0 or abs(c) >= len(dimension)
                    ) and self.is_valid_coordinate(*(coordinates + [c])):
                        adjacently_mined_cell = index_in_nD(
                            *([coordinates] + [(c,)]), *self.dimensions
                        )
                        if np.any(self.cells[adjacently_mined_cell]):
                            self.reveal_cell(*coordinates)
                            break

    def reveal_cell(self, *coordinates):
        cell = self.cells[index_in_nD(*coordinates)]
        if np.all(cell == 0):
            self.cells[index_in_nD(*coordinates)] = 1

    def is_valid_coordinate(self, *coordinates):
        for d, coord in enumerate(coordinates):
            if coord < 0 or coord >= self.dimensions[d]:
                return False
        return True

    def revealable_cells(self):
        valid_coords = np.indices(np.prod(self.dimensions)).reshape(
            -1, len(self.dimensions)
        )
        result = []
        for coord in valid_coords:
            if self.is_valid_coordinate(*coord):
                result.append(coord + (index_in_nD(*coord),))
        return result


import random


class CoordND:
    def __init__(self, dimensions, coordinates):
        self.dimensions = dimensions
        self.coordinates = coordinates


def index_in_nD(n_d, n, coordinates):
    dimensions = [*list(n_d)]  # Unpacking list for easier access in this local scope
    return sum(
        [
            int(np.floor(i / d)) * np.prod(dimensions[: i + 1]) + j
            for i, (d, j) in enumerate(zip(dimensions, dimensions[-len(coordinates) :]))
        ]
    )


class MinefieldND:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.mine_count = 1
        for dimension in dimensions:
            self.mine_count *= dimension
        self.cells = [[0] * len(dimensions) for _ in range(self.mine_count)]
        self.place_mines()
        self.reveal_adjacent_mined_cells()

    def place_mines(self):
        mines = set()
        while len(mines) < self.mine_count:
            mine_coordinates = [np.random.randint(0, d) for d in dimensions]
            index = index_in_nD(*dimensions, *mine_coordinates)
            cell = self.cells[index]
            if all([c == 0 for c in cell]):
                mines.add((*mine_coordinates, index))
                self.cells[index] = [1] + [0] * (len(dimensions) - 1)

    def reveal_adjacent_mined_cells(self):
        for mine in self.revealable_cells():
            self.reveal_adjacent_mines(*mine)

    def reveal_adjacent_mines(self, *coordinates):
        index = index_in_nD(*dimensions, *coordinates)
        cell = self.cells[index]
        if all([c == 0 for c in cell]):
            for d in range(-1, 2):
                for dimension, dimension_index in enumerate(dimensions):
                    coord = coordinates[dimension_index] + d
                    if 0 <= coord < dimension:
                        x, y = np.divmod(
                            index,
                            [
                                np.prod(dimensions[:dimension]),
                                np.prod(dimensions[dimension:]),
                            ],
                        )
                        adjacent_index = index_in_nD(
                            *dimensions, *np.array([x, y, coord])
                        )
                        if self.cells[adjacent_index][dimension_index] != 0:
                            self.cells[index] = [1] + [int(c) for c in cell[1:]]
                            self.reveal_adjacent_mines(
                                *np.split(np.array(coordinates), len(dimensions))
                            )
                            break

    def revealable_cells(self):
        valid_coords = np.indices(np.prod(self.dimensions)).reshape(
            -1, len(self.dimensions)
        )
        result = []
        for coord in valid_coords:
            if self.is_valid_coordinate(*coord):
                result.append(coord + (index_in_nD(*coord),))
        return result

    def is_valid_coordinate(self, *coordinates):
        for d, coord in enumerate(coordinates):
            if coord < 0 or coord >= self.dimensions[d]:
                return False
        return True

    def is_game_won(self):
        return (
            all([all([c != 0 for c in cell]) for cell in self.cells])
            and sum([1 for cell in self.cells if all([c == -1 for c in cell])])
            + sum([1 for cell in self.cells if all([c == 1 for c in cell])])
            == self.mine_count
        )


dimensions = [5, 3]
minefield = MinefieldND(dimensions)
print(minefield.is_game_won())
