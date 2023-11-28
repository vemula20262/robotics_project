import numpy as np
import pygame
import sys
from collections import deque
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 650, 450
GRID_SIZE = 30
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Maze definition
maze = [
    "#####################",
    "#     #             #",
    "# ### # ###  #####  #",
    "# #   #   #    #    #",
    "# ######  ###  #  ###",
    "#      #S   #  #  # #",
    "####  ## #  ####  # #",
    "#     #  #  #       #",
    "#  ####  ####   #####",
    "#           #       #",
    "#####################"

]
# maze = [
#     "#######",
#     "#     #",
#     "# ##  #",
#     "# S   #",
#     "#  ## #",
#     "#     #",
#     "#######",
# ]

# Initialize Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rat Maze")

# Clock to control the frame rate
clock = pygame.time.Clock()


def draw_maze():
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == '#':
                color = BLACK
            elif maze[row][col] == 'S':
                color = (0, 255, 0)  # Green for starting point
            elif maze[row][col] == 'E':
                color = (255, 0, 0)  # Red for ending point
            else:
                color = WHITE
            pygame.draw.rect(screen, color, (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE), 0)


def find_start(maze):
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == 'S':
                return (row, col)


class VisibilityMap:
    def __init__(self, maze):
        self.maze = maze
        self.grid_size = (len(maze), len(maze[0]))
        self.visibility_map = np.full(self.grid_size, False, dtype=bool)
        self.boundaries = []

    def ray_casting(self, target_position, next_cell, maze):

        def line_intersects_line(p1, p2, p3, p4):
            # Check if line segment p1p2 intersects with line segment p3p4
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            A, B, C, D = map(np.array, [p1, p2, p3, p4])
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        def get_square_corners(center, half_size):
            cx, cy = center
            return [(cx - half_size, cy - half_size), (cx + half_size, cy - half_size),
                    (cx + half_size, cy + half_size), (cx - half_size, cy + half_size)]

        def line_square_collision(p1, p2, center, half_size):
            # Get the corners of the square
            square_corners = get_square_corners(center, half_size)
            # Check each of the four sides of the square
            for i in range(4):
                if line_intersects_line(p1, p2, square_corners[i], square_corners[(i + 1) % 4]):
                    return True
            return False

        half_size = 0.5  # Half the size of the grid cell
        for row_idx, row in enumerate(maze):
            for col_idx, cell in enumerate(row):
                if cell == '#':  # Obstacle found
                    # Convert grid coordinates to center of cell
                    obstacle_center = (row_idx, col_idx)
                    if line_square_collision(target_position, next_cell, obstacle_center, half_size):
                        return False  # Collision detected
        return True  # No collision with any obstacle
    def compute_boundary_cells(self):
        self.boundary_cells = []  # List to store boundary cell positions

        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                if not self.visibility_map[row, col]:  # Invisible cell
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check adjacent cells
                        nx, ny = row + dx, col + dy
                        if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                            if self.visibility_map[nx, ny]:  # Adjacent to a visible cell
                                self.boundary_cells.append((row, col))
                                break  # No need to check other neighbors
                                


    def update_visibility(self, target_position):
        self.visibility_map.fill(False)
        queue = deque([target_position])
        self.visibility_map[target_position] = True



        def chebyshev_distance(cell_a, cell_b):
            return max(abs(cell_a[0] - cell_b[0]), abs(cell_a[1] - cell_b[1]))


        done = set()
        while queue:
            # print(queue)
            for i in range(len(queue)):

                current = queue.popleft()
                # print(queue)
                current_distance_to_target = chebyshev_distance(current, target_position)
                # print(current_distance_to_target)

                # Inside your while loop, after dequeuing the current cell
                for direction in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1),
                                  (1, 1)]:  # 4 orthogonal and 4 diagonal directions
                    next_cell = (current[0] + direction[0], current[1] + direction[1])

                    # Check boundaries and whether the cell has been processed
                    if (0 <= next_cell[0] < self.grid_size[0] and
                            0 <= next_cell[1] < self.grid_size[1] and
                            not self.visibility_map[next_cell] and
                            next_cell not in done):

                        # Consider all eight surrounding cells ('canalysis' cells)
                        canalysis = []
                        for d in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            neighbor = (next_cell[0] + d[0], next_cell[1] + d[1])
                            if (0 <= neighbor[0] < self.grid_size[0] and
                                    0 <= neighbor[1] < self.grid_size[1] and
                                    chebyshev_distance(neighbor, target_position) < chebyshev_distance(next_cell,
                                                                                                       target_position)):
                                canalysis.append(neighbor)
                        # print(canalysis, next_cell)
                        if maze[next_cell[0]][next_cell[1]] != '#':
                            
                            if any(self.visibility_map[(c[0], c[1])] for c in canalysis):
                                # raycasting
                                if all(self.visibility_map[(c[0], c[1])] for c in canalysis):
                                    # All canalysis cells are visible, set the current cell as visible
                                    self.visibility_map[next_cell] = True

                                elif self.ray_casting(target_position, next_cell, maze):

                                    # Ray casting is successful, set the current cell as visible
                                    self.visibility_map[next_cell] = True

                            queue.append(next_cell)
                        done.add(next_cell)
        # print(target_position)
        
    def generate_repulsion_field(self):
        # Initialize distance field with infinity
        distance_field = np.full(self.grid_size, np.inf)
        # Set the boundary cells to distance 0
        for cell in self.boundary_cells:
            distance_field[cell] = 0

        # Queue for BFS
        queue = deque(self.boundary_cells)

        while queue:
            current_cell = queue.popleft()
            current_distance = distance_field[current_cell]

            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check 4 directions
                next_cell = (current_cell[0] + direction[0], current_cell[1] + direction[1])

                if (0 <= next_cell[0] < self.grid_size[0] and
                        0 <= next_cell[1] < self.grid_size[1] and
                        distance_field[next_cell] == np.inf):  # Unvisited cell

                    if not self.visibility_map[next_cell]:  # If it's part of the invisible region
                        distance_field[next_cell] = current_distance - 1
                    else:  # If it's part of the visible region
                        distance_field[next_cell] = current_distance + 1

                    queue.append(next_cell)

        # The distance_field now contains the repulsion field
        return distance_field


def draw_visibility_map(visibility_map):
    for row in range(len(visibility_map.maze)):
        for col in range(len(visibility_map.maze[0])):
            if visibility_map.visibility_map[row, col]:
                pygame.draw.circle(screen, (0, 0, 255),
                                   (col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 4)

def draw_boundary_cells(boundary_cells):
    for row, col in boundary_cells:
        pygame.draw.rect(screen, (255, 0, 0), (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Fill red for boundary cells

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

def parse_maze(maze_str):
    """
    Parse the maze string representation into a numerical array.
    """
    return np.array([[0 if char == '#' else 1 for char in row] for row in maze_str])


def main():
    start_position = find_start(maze)
    visibility_map = VisibilityMap(maze)
    visibility_map.update_visibility(start_position)
    visibility_map.compute_boundary_cells() 
    draw_boundary_cells(visibility_map.boundary_cells)
    rep = visibility_map.generate_repulsion_field()
    repulsion_field = rep
    print(rep)
    midpoint = 1. - np.max(repulsion_field) / (np.max(repulsion_field) - np.min(repulsion_field))
    
    # Create a custom colormap
    # Colors will range from blue (less intense) for negative values to red (more intense) for positive values
    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', ['red', 'white', 'blue'])
    
    # Create a normalize object that will map the repulsion values to the [0, 1] range of the colormap
    norm = mcolors.TwoSlopeNorm(vmin=np.min(repulsion_field), vcenter=0, vmax=np.max(repulsion_field))
    
    # Plot the heatmap
    plt.imshow(repulsion_field, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.title('Repulsion Field Heatmap')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(WHITE)
        draw_maze()
        draw_visibility_map(visibility_map)
        visibility_map.compute_boundary_cells() 
        draw_boundary_cells(visibility_map.boundary_cells)
        rep = visibility_map.generate_repulsion_field()

        

        pygame.display.flip()
        
        clock.tick(FPS)
        

if __name__ == "__main__":
    main()
