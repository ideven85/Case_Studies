"""
6.101 Lab 7:
Six Double-Oh Mines
"""

import pickle

#!/usr/bin/env python3

from typing import List
import doctest

# NO ADDITIONAL IMPORTS ALLOWED!
"""
At least 10 hours per lab
"""

# For 2d Mine Sweeper 8 neighbours
directions = ((-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1), (1, 0), (1, -1), (1, 1))

# For n dimensional MineSweeper => 3^n-1 neighbours


def dump(game):
    """
    Prints a human-readable version of a game (provided as a dictionary)
    """
    keys = ("board", "dimensions", "state", "visible")
    # ^ Uses only default game keys. If you modify this you will need
    # to update the docstrings in other functions!
    for key in keys:
        val = game[key]
        if isinstance(val, list) and val and isinstance(val[0], list):
            print(f"{key}:")
            for inner in val:
                print(f"    {inner}")
        else:
            print(f"{key}:", val)


# 2-D IMPLEMENTATION


def new_game_2d(nrows, ncolumns, mines):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'visible' fields adequately initialized.

    Parameters:
       nrows (int): Number of rows
       ncolumns (int): Number of columns
       mines (list): List of mines, given in (row, column) pairs, which are
                     tuples

    Returns:
       A game state dictionary

    >>> dump(new_game_2d(2, 4, [(0, 0), (1, 0), (1, 1)]))
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: (2, 4)
    state: ongoing
    visible:
        [False, False, False, False]
        [False, False, False, False]
    """
    board = [[0 for _ in range(ncolumns)] for _ in range(nrows)]
    for x in mines:
        board[x[0]][x[1]] = "."

    visible = [[False for _ in range(ncolumns)] for _ in range(nrows)]

    for r in range(nrows):
        for c in range(ncolumns):
            if board[r][c] == 0:
                board[r][c] = len(
                    [
                        ((r + dx), (c + dy))
                        for dx, dy in directions
                        if 0 <= r + dx < nrows
                        and 0 <= c + dy < ncolumns
                        and board[r + dx][c + dy] == "."
                    ]
                )

    return {
        "dimensions": (nrows, ncolumns),
        "board": board,
        "state": "ongoing",
        "visible": visible,
    }


def dig_2d(game, row, col):
    """
    Reveal the cell at (row, col), and, in some cases, recursively reveal its
    neighboring squares.

    Update game['visible'] to reveal (row, col).  Then, if (row, col) has no
    adjacent mines (including diagonally), then recursively reveal (dig up) its
    eight neighbors.  Return an integer indicating how many new squares were
    revealed in total, including neighbors, and neighbors of neighbors, and so
    on.

    The state of the game should be changed to 'defeat' when at least one mine
    is visible on the board after digging (i.e. game['visible'][mine_location]
    == True), 'victory' when all safe squares (squares that do not contain a
    mine) and no mines are visible, and 'ongoing' otherwise.

    Parameters:
       game (dict): Game state
       row (int): Where to start digging (row)
       col (int): Where to start digging (col)

    Returns:
       int: the number of new squares revealed

    >>> game = {'dimensions': (2, 4),
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'visible': [[False, True, False, False],
    ...                  [False, False, False, False]],
    ...         'state': 'ongoing'

                }
    >>> dig_2d(game, 0, 3)
    4
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: (2, 4)
    state: victory
    visible:
        [False, True, True, True]
        [False, False, True, True]

    >>> game = {'dimensions': [2, 4],
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'visible': [[False, True, False, False],
    ...                  [False, False, False, False]],
    ...         'state': 'ongoing'


    }
    >>> dig_2d(game, 0, 0)
    1
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: [2, 4]
    state: defeat
    visible:
        [True, True, False, False]
        [False, False, False, False]
    'ascii_art': '.310\n..10'
    """
    if game["state"] == "defeat" or game["state"] == "victory":
        game["state"] = game["state"]  # keep the state the same
        return 0

    if game["board"][row][col] == ".":
        game["visible"][row][col] = True
        game["state"] = "defeat"
        return 1

    rows, columns = game["dimensions"]
    num_revealed_squares = [
        ((row + dx), (col + dy))
        for dx, dy in directions
        if 0 <= row + dx < rows
        and 0 <= col + dy < columns
        and not game["visible"][row + dx][col + dy]
    ]

    if num_revealed_squares == 0:
        game["state"] = "victory"
        return 0

    if not game["visible"][row][col]:
        game["visible"][row][col] = True
        revealed = 1
    else:
        return 0

    if game["board"][row][col] == 0:

        neighbours = (
            ((row + dx), (col + dy))
            for dx, dy in directions
            if 0 <= row + dx < rows
            and 0 <= col + dy < columns
            and not game["visible"][row + dx][col + dy]
            and game["board"][row + dx][col + dy] != "."
        )
        for r, c in neighbours:
            revealed += dig_2d(game, r, c)

    num_revealed_squares = 0
    for r in range(rows):
        # for each r,
        for c in range(columns):
            # for each c,

            if not game["visible"][r][c] and not game["board"][r][c] == ".":
                num_revealed_squares += 1

    game["state"] = "victory" if num_revealed_squares == 0 else "ongoing"
    return revealed


def render_2d_locations(game, all_visible=False):
    """
    Prepare a game for display.

    Returns a two-dimensional array (list of lists) of '_' (hidden squares),
    '.' (mines), ' ' (empty squares), or '1', '2', etc. (squares neighboring
    mines).  game['visible'] indicates which squares should be visible.  If
    all_visible is True (the default is False), game['visible'] is ignored
    and all cells are shown.

    Parameters:
       game (dict): Game state
       all_visible (bool): Whether to reveal all tiles or just the ones allowed
                    by game['visible']

    Returns:
       A 2D array (list of lists)

    >>> game = {'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'visible':  [[False, True, True, False],
    ...                   [False, False, True, False]]}
    >>> render_2d_locations(game, False)
    [['_', '3', '1', '_'], ['_', '_', '1', '_']]

    >>> render_2d_locations(game, True)
    [['.', '3', '1', ' '], ['.', '.', '1', ' ']]
    """
    # raise NotImplementedError
    board = game["board"]
    visible = game["visible"]
    rows, columns = game["dimensions"]
    if all_visible:
        out = [
            [str(board[r][c]) if board[r][c] != 0 else " " for c in range(columns)]
            for r in range(rows)
        ]

    else:
        out = [
            [
                "_" if not visible[r][c] else str(board[r][c]) if board[r][c] else " "
                for c in range(columns)
            ]
            for r in range(rows)
        ]
    return out


def render_2d_board(game, all_visible=False):
    """
    Render a game as ASCII art.

    Returns a string-based representation of argument 'game'.  Each tile of the
    game board should be rendered as in the function
        render_2d_locations(game)

    Parameters:
       game (dict): Game state
       all_visible (bool): Whether to reveal all tiles or just the ones allowed
                           by game['visible']

    Returns:
       A string-based representation of game

    >>> render_2d_board({'dimensions': (2, 4),
    ...                  'state': 'ongoing',
    ...                  'board': [['.', 3, 1, 0],
    ...                            ['.', '.', 1, 0]],
    ...                  'visible':  [[True, True, True, False],
    ...                            [False, False, True, False]]})
    '.31_\\n__1_'
    """
    board = game["board"]
    visible = game["visible"]
    rows, columns = game["dimensions"]
    if all_visible:
        out = [
            [str(board[r][c]) if board[r][c] != 0 else " " for c in range(columns)]
            for r in range(rows)
        ]
        return "\n".join(
            "".join(out[r][c] for c in range(columns)) for r in range(rows)
        )

    else:
        out = [
            [
                "_" if not visible[r][c] else str(board[r][c]) if board[r][c] else " "
                for c in range(columns)
            ]
            for r in range(rows)
        ]
        return "\n".join(
            "".join(out[r][c] for c in range(columns)) for r in range(rows)
        )


# N-D IMPLEMENTATION


def vector(dimensions: int) -> List[int]:
    """
    Creates a vector in len(dimensions)
    Args:
        dimensions:

    Returns:

    """
    pass


def all_coords(dimensions):
    """
    Modify the code below to make this function into an efficient generator.
    A function that generates all possible coordinates in a given board.
    """
    if len(dimensions) == 1:
        return [(x,) for x in range(dimensions[0])]
    first = all_coords(dimensions[:1])
    rest = all_coords(dimensions[1:])
    result = []
    for start in first:
        for end in rest:
            result.append(start + end)
    return result


def new_game_nd(dimensions, mines):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'visible' fields adequately initialized.

    Args:
       dimensions (tuple): Dimensions of the board a 3d (cube like) hypermine would like rows, columns and number of fields in each row, column i think what would a 4d look like?
       mines (list): mine locations as a list of tuples, each an
                     N-dimensional coordinate Consider a vector position as a mine

    Returns:
       A game state dictionary

    >>> g = new_game_nd((2, 4, 2), [(0, 0, 1), (1, 0, 0), (1, 1, 1)])
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    state: ongoing
    visible:
        [[False, False], [False, False], [False, False], [False, False]]
        [[False, False], [False, False], [False, False], [False, False]]
    >>> game = new_game_nd((3,3,2),[(1,2,0)]) # 3 rows 3 columns 2 cells in each cell
    >>> dump(game)
    board:
        [[0, 0], [1, 1], [1, 1]]
        [[0, 0], [1, 1], ['.', 1]]
        [[0, 0], [1, 1], [1, 1]]
    dimensions: (3, 3, 2)
    state: ongoing
    visible:
        [[False, False], [False, False], [False, False]]
        [[False, False], [False, False], [False, False]]
        [[False, False], [False, False], [False, False]]
    """

    n = len(dimensions)
    board = all_coords(dimensions)

    return {
        "board": board,
        "dimensions": dimensions,
        "state": "ongoing",
        # "visible": visible
    }


def dig_nd(game, coordinates):
    """
    Recursively dig up square at coords and neighboring squares.

    Update the visible to reveal square at coords; then recursively reveal its
    neighbors, as long as coords does not contain and is not adjacent to a
    mine.  Return a number indicating how many squares were revealed.  No
    action should be taken and 0 returned if the incoming state of the game
    is not 'ongoing'.

    The updated state is 'defeat' when at least one mine is visible on the
    board after digging, 'victory' when all safe squares (squares that do
    not contain a mine) and no mines are visible, and 'ongoing' otherwise.

    Args:
       coordinates (tuple): Where to start digging

    Returns:
       int: number of squares revealed

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'visible': [[[False, False], [False, True], [False, False],
    ...                [False, False]],
    ...               [[False, False], [False, False], [False, False],
    ...                [False, False]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 3, 0))
    8
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    state: ongoing
    visible:
        [[False, False], [False, True], [True, True], [True, True]]
        [[False, False], [False, False], [True, True], [True, True]]
    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'visible': [[[False, False], [False, True], [False, False],
    ...                [False, False]],
    ...               [[False, False], [False, False], [False, False],
    ...                [False, False]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 0, 1))
    1
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    state: defeat
    visible:
        [[False, True], [False, True], [False, False], [False, False]]
        [[False, False], [False, False], [False, False], [False, False]]
    """
    raise NotImplementedError


def render_nd(game, all_visible=False):
    """
    Prepare the game for display.

    Returns an N-dimensional array (nested lists) of '_' (hidden squares), '.'
    (mines), ' ' (empty squares), or '1', '2', etc. (squares neighboring
    mines).  The game['visible'] array indicates which squares should be
    visible.  If all_visible is True (the default is False), the game['visible']
    array is ignored and all cells are shown.

    Args:
       all_visible (bool): Whether to reveal all tiles or just the ones allowed
                           by game['visible']

    Returns:
       An n-dimensional array of strings (nested lists)

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'visible': [[[False, False], [False, True], [True, True],
    ...                [True, True]],
    ...               [[False, False], [False, False], [True, True],
    ...                [True, True]]],
    ...      'state': 'ongoing'}
    >>> render_nd(g, False)
    [[['_', '_'], ['_', '3'], ['1', '1'], [' ', ' ']],
     [['_', '_'], ['_', '_'], ['1', '1'], [' ', ' ']]]

    >>> render_nd(g, True)
    [[['3', '.'], ['3', '3'], ['1', '1'], [' ', ' ']],
     [['.', '3'], ['3', '.'], ['1', '1'], [' ', ' ']]]
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Test with doctests. Helpful to debug individual autocomplete_autocorrect.py functions.
    # _doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    # doctest.testmod(optionflags=_doctest_flags)  # runs ALL doctests

    # Alternatively, can run the doctests JUST for specified function/methods,
    # e.g., for render_2d_locations or any other function you might want.  To
    # do so, comment out the above line, and uncomment the below line of code.
    # This may be useful as you write/debug individual doctests or functions.
    # Also, the verbose flag can be set to True to see all test results,
    # including those that pass.
    #
    # doctest.run_docstring_examples(
    #    render_2d_locations,
    #    globals(),
    #    optionflags=_doctest_flags,
    #    verbose=False
    # )
    with open("test_inputs/testnd_integration3.pickle", "rb") as f:
        smallnd = pickle.load(f)
    print(smallnd["dimensions"])
    # print(len(all_coords(small2d['dimensions'])))
    dim = (3, 3, 2)
    print(all_coords((dim)))  # 3^n-1 neighbours
    print(len(all_coords(dim)))
