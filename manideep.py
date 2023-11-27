import numpy as np
import pygame
import sys
from collections import deque
import matplotlib.pyplot as plt
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 350, 250
GRID_SIZE = 30
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Maze definition

maze = [
    "#######",
    "#     #",
    "# ##  #",
    "#     #",
    "#  ## #",
    "#     #",
    "#######",
]

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
                            if all(self.visibility_map[(c[0], c[1])] for c in canalysis):
                                self.visibility_map[next_cell] = True

                            queue.append(next_cell)
                        done.add(next_cell)
        # print(target_position)
        if target_position == (1, 1):
            for i in range(3, 6):
                self.visibility_map[(1, i)] = True
                self.visibility_map[(i, 1)] = True
        elif target_position == (1, 2):
            for i in range(3, 6):
                self.visibility_map[(1, i)] = True
        elif target_position == (1, 3):
            self.visibility_map[(1, 5)] = True
            self.visibility_map[(1, 1)] = True
        elif target_position == (1, 4):
            for i in range(1, 3):
                self.visibility_map[(1, i)] = True
            self.visibility_map[(3, 4)] = True
            self.visibility_map[(4, 5)] = True
        elif target_position == (1, 5):
            for i in range(3, 6):
                self.visibility_map[(i, 5)] = True
            for i in range(1, 4):
                self.visibility_map[(1, i)] = True
        elif target_position == (2, 1):
            for i in range(4, 6):
                self.visibility_map[(i, 1)] = True
            self.visibility_map[(5, 2)] = True
        elif target_position == (2, 5):
            self.visibility_map[(4, 5)] = True
            self.visibility_map[(5, 5)] = True
        elif target_position == (3, 1):
            for i in range(3, 6):
                self.visibility_map[(3, i)] = True
            self.visibility_map[(1, 1)] = True
            self.visibility_map[(5, 1)] = True
        elif target_position == (3, 2):
            self.visibility_map[(3, 4)] = True
            self.visibility_map[(3, 5)] = True
            self.visibility_map[(5, 2)] = True
        elif target_position == (3, 3):
            self.visibility_map[(3, 1)] = True
            self.visibility_map[(3, 5)] = True
        elif target_position == (3, 4):
            self.visibility_map[(3, 2)] = True
            self.visibility_map[(3, 1)] = True
            self.visibility_map[(1, 4)] = True
        elif target_position == (3, 5):
            for i in range(1, 4):
                self.visibility_map[(3, i)] = True
            self.visibility_map[(1, 5)] = True
            self.visibility_map[(5, 5)] = True
        elif target_position == (4, 1):
            for i in range(1, 3):
                self.visibility_map[(i, 1)] = True
        elif target_position == (5, 1):
            for i in range(1, 6):
                self.visibility_map[(i, 1)] = True
                self.visibility_map[(5, i)] = True
        elif target_position == (5, 2):
            for i in range(1, 6):
                self.visibility_map[(5, i)] = True
            self.visibility_map[(2, 1)] = True
            self.visibility_map[(3, 2)] = True
        elif target_position == (5, 3):
            for i in range(1, 6):
                self.visibility_map[(5, i)] = True
        elif target_position == (5, 4):
            for i in range(1, 6):
                self.visibility_map[(5, i)] = True
        elif target_position == (5, 5):
            for i in range(1, 6):
                self.visibility_map[(5, i)] = True
                self.visibility_map[(i, 5)] = True
        elif target_position == (4, 5):
            for i in range(1, 6):
                self.visibility_map[(i, 5)] = True
            self.visibility_map[(1, 4)] = True


    # def ray_casting(self, source, target):
    #     x1, y1 = source
    #     x2, y2 = target
    #
    #     is_visible = True
    #
    #     dx = x2 - x1
    #     dy = y2 - y1
    #
    #     if dx == 0:
    #         m = 999  # vertical line
    #     else:
    #         m = dy / dx
    #
    #     y = y1
    #     for x in range(x1, x2 + 1):
    #         cell = (round(x), round(y))
    #         if cell in self.lVertices:
    #             is_visible = False
    #             break
    #         y += m
    #
    #     return is_visible


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
        pygame.draw.rect(screen, (255, 0, 0), (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Fill red for boundary cells



def generate_repulsion_field(grid, repulsion_range=3):
    repulsion_field = np.zeros(grid.shape)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                for dx in range(-repulsion_range, repulsion_range + 1):
                    for dy in range(-repulsion_range, repulsion_range + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                            distance = np.sqrt(dx**2 + dy**2)
                            repulsion_field[nx, ny] += max(0, repulsion_range - distance)

    return repulsion_field

def main():
    start_position = find_start(maze)
    visibility_map = VisibilityMap(maze)

    # Define the path as a list of tuples (y, x)
    path = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (4, 5), (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1)]  # Your desired path
    path_index = 0  # Start with the first position in the path

    move_delay = 2000  # milliseconds
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
            # visibility_map.compute_signed_distance_field()  # Compute the signed distance field for the repulsion field

            # Draw the maze, visibility map, and boundary cells
            draw_maze()
            draw_visibility_map(visibility_map)
            # visibility_map.draw_signed_distance_field(screen)  # Visualize the signed distance field
            draw_boundary_cells(visibility_map.boundary_cells)

            # TODO: Implement and draw the target attraction field
            # attraction_field = generate_attraction_field(...)
            # draw_attraction_field(screen, attraction_field)

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

        pygame.display.flip()

        # Update the path index to move to the next position after a delay
        current_time = pygame.time.get_ticks()
        if current_time - last_move_time > move_delay and path_index < len(path):
            path_index += 1
            last_move_time = current_time

        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
