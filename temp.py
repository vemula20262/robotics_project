import numpy as np
import pygame
import sys
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 650, 350
GRID_SIZE = 30
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Maze definition
maze = [
    "#####################",
    "#     #             #",
    "# ### # ###   ####  #",
    "# #   #   #     #   #",
    "# ######  ###   # S #",
    "#      #    # E # ###",
    "##### ##  # ##### # #",
    "#     #   # #       #",
    "# #####   ###   #####",
    "#           #       #",
    "#####################"
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
    def convert_maze_to_vertices(self):
        vertices = []
        for row_index, row in enumerate(self.maze):
            for col_index, cell in enumerate(row):
                if cell == '#':  # Wall
                    # Add the wall's vertices (corners)
                    vertices.append((col_index, row_index))
        return vertices

    # def update_visibility(self, target_position):
    #     self.visibility_map.fill(False)
    #     self.visibility_map[target_position] = True
    #
    #     for row in range(self.grid_size[0]):
    #         for col in range(self.grid_size[1]):
    #             if self.maze[row][col] != '#':
    #                 self.visibility_map[row, col] = self.is_visible(target_position, (row, col))

    def update_visibility(self, target_position):
        self.visibility_map.fill(False)
        queue = deque([target_position])
        self.visibility_map[target_position] = True
        def chebyshev_distance(cell_a, cell_b):
            return max(abs(cell_a[0] - cell_b[0]), abs(cell_a[1] - cell_b[1]))

        while queue:
            print(queue)
            current = queue.popleft()
            current_distance_to_target = chebyshev_distance(current, target_position)

            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 directions: up, down, left, right
                next_cell = (current[0] + direction[0], current[1] + direction[1])

                if (0 <= next_cell[0] < self.grid_size[0] and
                        0 <= next_cell[1] < self.grid_size[1] and
                        not self.visibility_map[next_cell]):

                    # Determine Canalysis for the current cell
                    canalysis = [(next_cell[0] + dx, next_cell[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]

                                 if 0 <= next_cell[0] + dx < self.grid_size[0]
                                 and 0 <= next_cell[1] + dy < self.grid_size[1]
                                 and chebyshev_distance((next_cell[0] + dx, next_cell[1] + dy),
                                                        target_position) < current_distance_to_target]
                    print(canalysis, next_cell,maze[next_cell[0]][next_cell[1]], target_position)

                    # Check visibility based on Canalysis cells
                    if all(self.visibility_map[c[0], c[1]] for c in canalysis):
                        if maze[next_cell[0]][next_cell[1]] != '#':
                            self.visibility_map[next_cell] = True
                        queue.append(next_cell)
                    elif not any(self.visibility_map[c[0], c[1]] for c in canalysis) :
                        self.visibility_map[next_cell] = False
                    else:
                        print("nxt",next_cell)
                        return False
                    # elif any(self.visibility_map[c[0], c[1]] for c in canalysis):
                    #     # If any Canalysis cells are visible, perform ray casting
                    #     self.visibility_map[next_cell] = self.ray_casting(target_position, next_cell)
                    #     if self.visibility_map[next_cell]:
                    #         queue.append(next_cell)
    # def is_visible(self, target_position, cell):
    #     if cell == target_position:
    #         return True
    #
    #     # Calculate Chebyshev distance
    #     def chebyshev_distance(cell_a, cell_b):
    #         return max(abs(cell_a[0] - cell_b[0]), abs(cell_a[1] - cell_b[1]))
    #
    #     distance_to_target = chebyshev_distance(cell, target_position)
    #
    #     # Determine Canalysis cells
    #     canalysis = [(r, c) for r in range(max(0, cell[0] - 1), min(self.grid_size[0], cell[0] + 2))
    #                         for c in range(max(0, cell[1] - 1), min(self.grid_size[1], cell[1] + 2))
    #                         if chebyshev_distance((r, c), target_position) < distance_to_target]
    #     print(canalysis,cell,target_position)
    #     # Check visibility based on Canalysis cells
    #     if all(self.visibility_map[c[0], c[1]] for c in canalysis):
    #         return True
    #     elif not any(self.visibility_map[c[0], c[1]] for c in canalysis):
    #         return False
    #     else:
    #         # Perform ray casting from target to the cell
    #         return self.ray_casting(target_position, cell)

    def ray_casting(self, start, end):
        # Assume start is the point to test ('locationToTest') and the polygon is the maze walls
        x_test, y_test = start
        iCrossings = 0

        for iVertex in range(len(self.lVertices)):
            # Assume the vertices are defined with (x, y) tuples
            x_start, y_start = self.lVertices[iVertex]
            if iVertex < len(self.lVertices) - 1:
                x_end, y_end = self.lVertices[iVertex + 1]
            else:
                x_end, y_end = self.lVertices[0]

            # Check if point lies on a horizontal line segment
            if y_start == y_test == y_end and x_start <= x_test <= x_end:
                return True  # Point lies exactly on a horizontal line segment

            # Check if test point is to the right of both segment endpoints or to the left
            if y_start > y_test and y_end > y_test or y_start < y_test and y_end < y_test:
                continue  # Ray cannot intersect with this segment

            # Calculate intersection point of the ray with the edge
            x_intersect = ((y_test - y_end) * (x_start - x_end) / (y_start - y_end)) + x_end

            # Check if the intersection point is to the right of the test point
            if x_intersect > x_test:
                iCrossings += 1

        # Odd number of crossings indicates the point is inside the polygon
        return iCrossings % 2 != 0
def draw_visibility_map(visibility_map):
    for row in range(len(visibility_map.maze)):
        for col in range(len(visibility_map.maze[0])):
            if visibility_map.visibility_map[row, col]:
                pygame.draw.circle(screen, (0, 0, 255), (col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 4)

def main():
    start_position = find_start(maze)
    visibility_map = VisibilityMap(maze)
    visibility_map.update_visibility(start_position)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(WHITE)
        draw_maze()
        draw_visibility_map(visibility_map)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
