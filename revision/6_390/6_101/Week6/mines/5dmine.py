import random

# Define the size of the game board
N = 10

# Initialize the game board with all cells set to unrevealed
game_board = [[0] * N for _ in range(N)]

# Place mines randomly on the game board
for i in range(N):
    for j in range(N):
        if random.random() < 0.1:
            game_board[i][j] = -1
print(game_board)


# Define a function to check if a cell is safe or not
def is_safe(row, col):
    # Check if the cell is within the bounds of the game board
    if row < 0 or col < 0 or row >= N or col >= N:
        return False

    # Check if the cell has already been revealed
    if game_board[row][col] != 0:
        return False

    # Check if the cell is a mine
    if game_board[row][col] == -1:
        return False

    return True


# Define a function to reveal a cell and its adjacent cells
def reveal(row, col):
    # Check if the cell is safe
    if not is_safe(row, col):
        return

    # Reveal the cell and its adjacent cells
    game_board[row][col] = 1
    for i in range(max(0, row - 1), min(N - 1, row + 1)):
        for j in range(max(0, col - 1), min(N - 1, col + 1)):
            if is_safe(i, j):
                game_board[i][j] = 1


# Define a function to print the current state of the game board
def print_game_board():
    for row in range(N):
        for col in range(N):
            if game_board[row][col] == -1:
                print("M", end=" ")
            elif game_board[row][col] == 0:
                print(".", end=" ")
            else:
                print(" ", end=" ")
        print()


# Start the game
print_game_board()
reveal(5, 5)
print_game_board()
