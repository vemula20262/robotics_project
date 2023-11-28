import numpy as np
import pygame
import sys
from collections import deque
import matplotlib.pyplot as plt
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1900, 1000
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
#
# maze = [
# "###############################################################",
# "###############################################################",
# "###############################################################",
# "###               ###                                       ###",
# "###               ###                                       ###",
# "###               ###                                       ###",
# "###   #########   ###   #########      ###############      ###",
# "###   #########   ###   #########      ###############      ###",
# "###   #########   ###   #########      ###############      ###",
# "###   ###         ###         ###            ###            ###",
# "###   ###         ###         ###            ###            ###",
# "###   ###         ###         ###            ###            ###",
# "###   ##################      #########      ###      #########",
# "###   ##################      #########      ###      #########",
# "###   ##################      #########      ###      #########",
# "###                  ###            ###      ###      ###   ###",
# "###                  ###            ###      ###      ###   ###",
# "###                  ###            ###      ###      ###   ###",
# "############      ######   ###      ############      ###   ###",
# "############      ######   ###      ############      ###   ###",
# "############      ######   ###      ############      ###   ###",
# "###               ###      ###      ###                     ###",
# "###               ###      ###      ###                     ###",
# "###               ###      ###      ###                     ###",
# "###      ############      ############         ###############",
# "###      ############      ############         ###############",
# "###      ############      ############         ###############",
# "###                                 ###                     ###",
# "###                                 ###                     ###",
# "###                                 ###                     ###",
# "###############################################################",
# "###############################################################",
# "###############################################################",
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

        # print(self.boundary_cells)

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
        # print(distance_field)
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
        print(distance_field)
        return distance_field
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
                            # print(next_cell)
                            # print(canalysis)
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

def draw_boundary_cells(boundary_cells):
    for row, col in boundary_cells:
        pygame.draw.rect(screen, (255, 0, 0), (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Fill red for boundary cells

def compute_attraction_field(maze, target):
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

    # make target cell with attraction field to 0
    field[target] = 0

    # make all 8 cells around the target cell with attraction field to 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = target[0] + dx, target[1] + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
            field[nx, ny] = 0
    return field




def parse_maze(maze_str):
    """
    Parse the maze string representation into a numerical array.
    """
    return np.array([[0 if char == '#' else 1 for char in row] for row in maze_str])



def main():
    visibility_map = VisibilityMap(maze)

    # Define the path as a list of tuples (y, x)
#     path = [
#     (27, 51), (27, 48), (27, 45), (27, 42), (24, 42), (24, 45), (21, 45), (21, 48),
#     (21, 51), (18, 51), (15, 51), (12, 51), (9, 51), (9, 54), (6, 54), (3, 54),
#     (3, 51), (3, 48), (3, 45), (3, 42), (3, 39), (3, 36), (3, 33), (3, 30),
#     (3, 27), (3, 24), (3, 21), (6, 21), (9, 21), (9, 24), (15, 24), (18, 24),
#     (21, 24), (24, 24), (27, 24), (27, 21), (27, 18), (27, 15), (27, 12), (27, 9),
#     (27, 6), (24, 6), (21, 6), (21, 9), (21, 12), (18, 12), (15, 12), (15, 9),
#     (15, 6), (15, 3), (12, 3), (9, 3), (6, 3), (3, 3)
# ]
#     path = [ (9, 17), (9, 16), (9, 15), (9, 14), (8, 14), (8, 15), (7, 15), (7, 16), (7, 17), (6, 17),
#             (5, 17), (4, 17), (3, 17), (3, 18), (2, 18), (1, 18), (1, 17), (1, 16), (1, 15), (1, 14), (1, 13), (1, 12),
#             (1, 11), (1, 10), (1, 9), (1, 8), (1, 7), (2, 7), (3, 7), (3, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8),
#             (9, 7), (9, 6), (9, 5), (9, 4), (9, 3), (9, 2), (8, 2), (7, 2), (7, 3), (7, 4), (6, 4), (5, 4), (5, 3),
#             (5, 2), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1)]

    path = [  (3, 7), (3, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8),
            (9, 7), (9, 6), (9, 5), (9, 4), (9, 3), (9, 2), (8, 2), (7, 2), (7, 3), (7, 4), (6, 4), (5, 4), (5, 3),
            (5, 2), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1)]
    robot_path = [(9,18)]
    path_index = 0  # Start with the first position in the path

    move_delay = 30  # milliseconds
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
            # visibility_map.calculate_distance_field()  # Calculate the distance field
            # Draw the maze, visibility map, and boundary cells
            draw_maze()
            draw_visibility_map(visibility_map)
            # draw_boundary_cells(visibility_map.boundary_cells)
        # Draw the path and the target
        for i in range(path_index):
            if i + 1 < len(path):
                start_point = (path[i][1] * GRID_SIZE + GRID_SIZE // 2, path[i][0] * GRID_SIZE + GRID_SIZE // 2)
                end_point = (path[i + 1][1] * GRID_SIZE + GRID_SIZE // 2, path[i + 1][0] * GRID_SIZE + GRID_SIZE // 2)
                pygame.draw.line(screen, (128, 128, 128), start_point, end_point, 5)

        # Draw the target at the current path position
        if path_index < len(path):
            target_pos = path[path_index]
            target_screen_x = target_pos[1] * GRID_SIZE + GRID_SIZE // 2
            target_screen_y = target_pos[0] * GRID_SIZE + GRID_SIZE // 2
            pygame.draw.circle(screen, (0, 128, 255), (target_screen_x, target_screen_y), GRID_SIZE // 2)

        # # now draw the robot path
        # for i in range(len(robot_path)-1):
        #     if i + 1 < len(robot_path):
        #         # draw the robot
        #         pygame.draw.circle(screen, (0, 128, 0), (robot_path[i][1] * GRID_SIZE + GRID_SIZE // 2, robot_path[i][0] * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 2)
        #         # it must be update the robot postion based on the attraction field




        # all the above this which we have done is to draw the maze and the path and the target
        np_maze = parse_maze(maze)
        # now we are going to work on the robot_path based on attraction_field
        attraction_field = compute_attraction_field(np_maze, path[path_index])
        repulsion_field = visibility_map.generate_repulsion_field()


        a,b =1,1
        # now in the robot path we are going to append the robot position and intial position already appended
        robot_position = robot_path[-1]
        # we are going to find the next position of the robot based on the attraction field and the visibility map
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1),
                                  (1, 1)]:
            nx, ny = robot_position[0] + dx, robot_position[1] + dy
            if 0 <= nx < np_maze.shape[0] and 0 <= ny < np_maze.shape[1]  and visibility_map.visibility_map[nx, ny]:
                intial_field = attraction_field[robot_position[0],robot_position[1]]+repulsion_field[robot_position[0],robot_position[1]]
                total_field = a * attraction_field[nx, ny] + b * repulsion_field[nx, ny]
                if intial_field <total_field:
                    robot_position = (nx, ny)
                    break
        robot_path.append(robot_position)

        # print(robot_path)
        # print(total_field)
    #




    # now render a robot in the maze with circle which follows target path but 2 frames later then it moves along the path
        for i in range(len(robot_path)-1):
            if i + 1 < len(robot_path):
                # draw the robot
                pygame.draw.circle(screen, (0, 128, 0), (robot_path[-2][1] * GRID_SIZE + GRID_SIZE // 2, robot_path[-2][0] * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 2)
                # it must be update the robot postion based on the attraction field


        pygame.display.flip()

        # Update the path index to move to the next position after a delay
        current_time = pygame.time.get_ticks()
        if current_time - last_move_time > move_delay and path_index < len(path):
            path_index += 1
            last_move_time = current_time

        clock.tick(FPS)
    plt.show()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
