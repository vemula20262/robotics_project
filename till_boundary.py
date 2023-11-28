import numpy as np
import pygame
import sys
from collections import deque
import matplotlib.pyplot as plt
import math

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
    "#      #    #  #  # #",
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
#     "#     #",
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
        self.lVertices = self.convert_maze_to_vertices()
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

        print(self.boundary_cells)

    def calculate_distance_field(self, resolution=1):
        # Initialize the distance field with None
        distance_field = [[None for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]

        # Initialize the queue and sets
        queue = deque()
        close_set = set()
        temp_set = set()

        # Add boundary cells to the queue and set their distance to 0
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                if self.is_edge_cell(row, col):
                    queue.append((row, col))
                    distance_field[row][col] = 0

        # While the queue is not empty, process cells
        while queue:
            close_set.update(queue)
            while queue:
                q = queue.popleft()
                neighbors = self.find_neighbors(q)
                for y in neighbors:
                    if y not in close_set and not self.is_obstacle(y):
                        temp_set.add(y)
            for y in temp_set:
                if not self.visibility_map[y[0]][y[1]]:  # Inside the invisible region
                    distance_field[y[0]][y[1]] = distance_field[q[0]][q[1]] + resolution
                else:  # Outside the invisible region
                    distance_field[y[0]][y[1]] = distance_field[q[0]][q[1]] - resolution
                queue.append(y)
            temp_set.clear()
        print(distance_field)
        self.distance_field = distance_field

    def is_edge_cell(self, row, col):
        if not self.visibility_map[row][col]:  # Check for invisible cells
            # Check only direct neighbors (up, down, left, right)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = row + dx, col + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if self.visibility_map[nx, ny]:  # Adjacent cell is visible
                        return True
        return False

    def find_neighbors(self, cell):
        row, col = cell
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-neighborhood
            nx, ny = row + dx, col + dy
            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                neighbors.append((nx, ny))
        return neighbors

    def is_obstacle(self, cell):
        row, col = cell
        return self.maze[row][col] == '#'

    def convert_maze_to_vertices(self):
        vertices = []
        for row_index, row in enumerate(self.maze):
            for col_index, cell in enumerate(row):
                if cell == '#':  # Wall
                    # Add the wall's vertices (corners)
                    vertices.append((col_index, row_index))
        return vertices

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
                            print(next_cell)
                            print(canalysis)
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


def draw_visibility_map(visibility_map):
    for row in range(len(visibility_map.maze)):
        for col in range(len(visibility_map.maze[0])):
            if visibility_map.visibility_map[row, col]:
                pygame.draw.circle(screen, (0, 0, 255),
                                   (col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 4)


def generate_wavefront(grid, start_pos):
    wavefront = np.full(grid.shape, -1)  # -1 indicates unvisited cells
    q = deque()
    q.append(start_pos)
    wavefront[start_pos] = 0

    while q:
        x, y = q.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-neighborhood
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if wavefront[nx, ny] == -1 and grid[nx, ny] == 0:
                    wavefront[nx, ny] = wavefront[x, y] + 1
                    q.append((nx, ny))

    return wavefront


def draw_boundary_cells(boundary_cells):
    for row, col in boundary_cells:
        pygame.draw.rect(screen, (255, 0, 0),
                         (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Fill red for boundary cells


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


def generate_repulsion_field(grid, repulsion_range=3):
    repulsion_field = np.zeros(grid.shape)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                for dx in range(-repulsion_range, repulsion_range + 1):
                    for dy in range(-repulsion_range, repulsion_range + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                            distance = np.sqrt(dx ** 2 + dy ** 2)
                            repulsion_field[nx, ny] += max(0, repulsion_range - distance)

    return repulsion_field


def main():
    start_position = find_start(maze)
    visibility_map = VisibilityMap(maze)

    # Define the path as a list of tuples (y, x)
    path = [(9, 19), (9, 18), (9, 17), (9, 16), (9, 15), (9, 14), (8, 14), (8, 15), (7, 15), (7, 16), (7, 17), (6, 17),
            (5, 17), (4, 17), (3, 17), (3, 18), (2, 18), (1, 18), (1, 17), (1, 16), (1, 15), (1, 14), (1, 13), (1, 12),
            (1, 11), (1, 10), (1, 9), (1, 8), (1, 7), (2, 7), (3, 7), (3, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8),
            (9, 7), (9, 6), (9, 5), (9, 4), (9, 3), (9, 2), (8, 2), (7, 2), (7, 3), (7, 4), (6, 4), (5, 4), (5, 3),
            (5, 2), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1)]

    path_index = 0  # Start with the first position in the path

    move_delay = 200  # milliseconds
    last_move_time = pygame.time.get_ticks()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        draw_maze()

        # Update the visibility map based on the current position of the robot
        if path_index < len(path):
            current_pos = path[path_index]
            visibility_map.update_visibility(current_pos)
            visibility_map.compute_boundary_cells()  # Compute boundary cells
            visibility_map.calculate_distance_field()  # Calculate the distance field

            # Draw the maze, visibility map, and boundary cells
            draw_maze()
            draw_visibility_map(visibility_map)
            draw_boundary_cells(visibility_map.boundary_cells)
        # Draw the visibility map
        draw_visibility_map(visibility_map)

        # Draw the path and the robot
        for i in range(path_index):
            if i + 1 < len(path):
                start_point = (path[i][1] * GRID_SIZE + GRID_SIZE // 2, path[i][0] * GRID_SIZE + GRID_SIZE // 2)
                end_point = (path[i + 1][1] * GRID_SIZE + GRID_SIZE // 2, path[i + 1][0] * GRID_SIZE + GRID_SIZE // 2)
                pygame.draw.line(screen, (128, 128, 128), start_point, end_point, 5)

        # Draw the robot at the current path position
        if path_index < len(path):
            robot_pos = path[path_index]
            robot_screen_x = robot_pos[1] * GRID_SIZE + GRID_SIZE // 2
            robot_screen_y = robot_pos[0] * GRID_SIZE + GRID_SIZE // 2
            pygame.draw.circle(screen, (0, 128, 255), (robot_screen_x, robot_screen_y), GRID_SIZE // 2)

        np_maze = parse_maze(maze)
        # generate the attraction field accroding to the target position
        # attraction_field = compute_attraction_field(np_maze, path[path_index])
        # render the attraction_field on the screen based on the value as gradient color of yellow
        # for row in range(len(attraction_field)):
        #     for col in range(len(attraction_field[0])):
        #         if attraction_field[row, col] != -1:
        #             # pygame.draw.rect(screen, (255, 2 * (attraction_field[row, col]), 0),
        #             #                  (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        # #             with 90% transparency
        #             pygame.draw.rect(screen, (255, 2 * (attraction_field[row, col]), 0),
        #                              (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        # plot the attraction_field
        # plot_maze_and_field(np_maze, attraction_field)

        pygame.display.flip()

        # Update the path index to move to the next position after a delay
        current_time = pygame.time.get_ticks()
        if current_time - last_move_time > move_delay and path_index < len(path):
            path_index += 1
            last_move_time = current_time

        clock.tick(FPS)

    grid_size = 20
    grid = np.zeros((grid_size, grid_size))
    grid[5:15, 10] = 1  # Adding an obstacle
    target_pos = (0, 0)  # Target position at the top left corner

    # Generate fields
    attraction_field = generate_wavefront(grid, target_pos)
    repulsion_field = generate_repulsion_field(grid)

    plt.show()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()