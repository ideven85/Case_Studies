# 6.101 recitation: lab 1 wrap-up

# how to represent a kernel?
# (a kernel is a square, odd-sized matrix of float values)

# like an image?
kernel = {
    "height": 3,
    "width": 3,
    "pixels": [0, 0, 0, 0, 0, 0, 0, 1, 0],
}

# OR:
kernel = [0, 0, 0, 0, 0, 0, 0, 1, 0]

# OR:
kernel = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 1, 0],
]


# OR:
kernel = {
    (-1, -1): 0,
    (-1, 0): 0,
    (-1, 1): 0,
    (0, -1): 0,
    (0, 0): 0,
    (0, 1): 0,
    (1, -1): 0,
    (1, 0): 1,
    (1, 1): 0,
}


# OR:
kernel = {
    (1, 0): 1,
}


# on to base correlate


def empty_image(height, width):
    return {
        "height": height,
        "width": width,
        "pixels": [0] * (height * width),
    }


def correlate(image, kernel):
    result = empty_image(image["height"], image["width"])
    for row in range(image["height"]):
        for col in range(image["width"]):
            pass  # TODO: what goes here?
    return result


# core of the correlation algorithm...


# ... when kernel represented as list of lists, making a square 2D matrix
def correlate_single_pixel(image, kernel, row, col):
    """
    returns output pixel at row, col
    """
    pixel_value = 0
    # TODO: how to compute the value at this spot?
    center = len(kernel) // 2  # kernel is square, so center row is also center column
    for kernel_row in len(kernel):
        for kernel_col in len(kernel):  # kernel is square
            dr = kernel_row - center
            dc = kernel_col - center
            pixel_value += kernel[kernel_row][kernel_col] * get_pixel(
                image, row + dr, col + dc
            )
    return pixel_value


# ... when kernel represented as dictionary, with (dr,dc) => kernel value at that offset in the kernel
def correlate_single_pixel(image, kernel, row, col):
    pass  # TODO: what goes here?


# create the coordinate-dictionary representation from a 2D array representation
kernel = make_kernel(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ]
)


def make_kernel(matrix):
    """
    Takes an odd square 2D matrix (a list of rows, where each row is a list of floats)
    and returns a coordinate dictionary (dr,dc) => float,
                 where the origin (0,0) is the center cell of the matrix
    """
    raise NotImplementedError
