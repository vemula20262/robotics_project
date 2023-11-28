import numpy as np
import matplotlib.pyplot as plt
import random

def create_maze(size=10, obstacle_density=0.2):
    """
    Create a maze with given size and obstacle density.
    """
    maze = np.random.choice([0, 1], (size, size), p=[obstacle_density, 1-obstacle_density])
    return maze

def find_target(maze):
    """
    Find a random position for the target in the maze.
    """
    while True:
        x, y = random.randint(0, maze.shape[0]-1), random.randint(0, maze.shape[1]-1)
        if maze[x, y] == 1:  # Ensure the target is not on an obstacle
            return x, y

def compute_attraction_field(maze, target):
    """
    Compute the attraction field for the maze given a target position.
    """
    field = np.full(maze.shape, -1)  # Initialize with -1
    queue = [target]
    field[target] = 100  # Starting value for the target

    while queue:
        x, y = queue.pop(0)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # Check 4 directions
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
                if maze[nx, ny] == 1 and field[nx, ny] == -1:  # If not an obstacle and not visited
                    field[nx, ny] = field[x, y] - 10  # Decrease the value
                    queue.append((nx, ny))

    return field

def plot_maze_and_field(maze, field):
    """
    Plot the maze and the attraction field.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot maze
    ax[0].imshow(maze, cmap='gray')
    ax[0].set_title('Maze')

    # Plot attraction field
    im = ax[1].imshow(field, cmap='viridis')
    ax[1].set_title('Attraction Field')
    fig.colorbar(im, ax=ax[1])

    plt.show()
    
def parse_maze(maze_str):
    """
    Parse the maze string representation into a numerical array.
    """
    return np.array([[0 if char == '#' else 1 for char in row] for row in maze_str])

def find_target_in_provided_maze(maze_str, target_char='S'):
    """
    Find the position of the target in the provided maze.
    """
    for x, row in enumerate(maze_str):
        for y, char in enumerate(row):
            if char == target_char:
                return x, y
    return None  # If the target is not found

def plot_maze_and_field_with_target(maze, field, target):
    """
    Plot the maze and the attraction field, highlighting the target position.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot maze with a blue color map
    ax[0].imshow(maze, cmap='Blues')
    ax[0].set_title('Maze')

    # Plot maze with target
    maze_with_target = maze.copy()
    maze_with_target[target] = 2  # Special value for target
    cmap_maze = plt.cm.gray
    cmap_maze.set_over('red')  # Color for the target
    ax[0].imshow(maze_with_target, cmap=cmap_maze, clim=[0, 1])
    ax[0].set_title('Maze with Target')

    # Plot attraction field
    im = ax[1].imshow(field, cmap='hot')
    ax[1].set_title('Attraction Field')
    fig.colorbar(im, ax=ax[1])

    plt.show()

# Provided maze string

maze_str_with_target = [
    "#######",
    "#     #",
    "# ##  #",
    "# S   #",
    "#  ## #",
    "#     #",
    "#######",
]

# maze_str_with_target = [
#     "#####################",
#     "#     #   S         #",
#     "# ### # ###  #####  #",
#     "# #   #   #    #    #",
#     "# ######  ###  #  ###",
#     "#      #    #  #  # #",
#     "####  ## #  ####  # #",
#     "#     #  #  #       #",
#     "#  ####  ####   #####",
#     "#           #       #",
#     "#####################"

# ]

# Parsing the updated maze and finding the target
maze_with_target = parse_maze(maze_str_with_target)
target_in_maze = find_target_in_provided_maze(maze_str_with_target, target_char='S')

# Computing the attraction field for the updated maze
field_with_target = compute_attraction_field(maze_with_target, target_in_maze)

# Plotting the updated maze and field with the target
plot_maze_and_field_with_target(maze_with_target, field_with_target, target_in_maze)

print(field_with_target)