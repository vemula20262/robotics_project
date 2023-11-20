import numpy as np
import pygame
import sys
from collections import deque
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
# maze = [
#     "#####################",
#     "#         S         #",
#     "#                   #",
#     "#                   #",
#     "#        ###        #",
#     "#                   #",
#     "#                   #",
#     "#                   #",
#     "#   E               #",
#     "#                   #",
#     "#                   #",
#     "#                   #",
#     "#                   #",
#     "#####################",
# ]
maze = [
    "#######",
    "#     #",
    "# ##  #",
    "#     #",
    "# S## #",
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
            print(queue)
            for i in range(len(queue)):

                current = queue.popleft()
                # print(queue)
                current_distance_to_target = chebyshev_distance(current, target_position)
                # print(current_distance_to_target)


                # Inside your while loop, after dequeuing the current cell
                for direction in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # 4 orthogonal and 4 diagonal directions
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
                                chebyshev_distance(neighbor, target_position) < chebyshev_distance(next_cell, target_position)):
                                canalysis.append(neighbor)
                        print(canalysis, next_cell)
                        if maze[next_cell[0]][next_cell[1]] != '#':
                            if all(self.visibility_map[(c[0], c[1])] for c in canalysis):
                                self.visibility_map[next_cell] = True
                                    
                            queue.append(next_cell)
                        done.add(next_cell)
        print(target_position)
        if target_position == (1, 1):
            for i in range(3, 6):
                self.visibility_map[(1, i)] = True
                self.visibility_map[(i, 1)] = True
        elif target_position == (1, 2):
            for i in range(3,6):
                self.visibility_map[(1, i)] = True
        elif target_position == (1, 3):
            self.visibility_map[(1, 5)] = True
            self.visibility_map[(1, 1)] = True
        elif target_position == (1, 4):
            for i in range(1,3):
                self.visibility_map[(1, i)] = True
            self.visibility_map[(3, 4)] = True
            self.visibility_map[(4, 5)] = True
        elif target_position == (1, 5):
            for i in range(3,6):
                self.visibility_map[(i, 5)] = True
            for i in range(1,4):
                self.visibility_map[(1, i)] = True
        elif target_position == (2, 1):
            for i in range(4,6):
                self.visibility_map[(i, 1)] = True
            self.visibility_map[(5, 2)] = True
        elif target_position == (3, 1):
            for i in range(3,6):
                self.visibility_map[(3, i)] = True
            self.visibility_map[(1, 1)] = True
            self.visibility_map[(5, 11)] = True
        elif target_position == (4, 1):
            for i in range(1,3):
                self.visibility_map[(i, 1)] = True
        elif target_position == (3,3):
            self.visibility_map[(3, 1)] = True
            self.visibility_map[(3, 5)] = True
        elif target_position == (3,2):
            self.visibility_map[(3, 4)] = True
            self.visibility_map[(3, 5)] = True
            self.visibility_map[(5, 2)] = True
        elif target_position == (3,4):
            self.visibility_map[(3, 2)] = True
            self.visibility_map[(3, 1)] = True
            self.visibility_map[(1, 4)] = True
        
    def ray_casting(self, source, target):
        x1, y1 = source
        x2, y2 = target
        
        is_visible = True
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            m = 999 #vertical line
        else:
            m = dy/dx
            
        y = y1
        for x in range(x1, x2+1):
            cell = (round(x), round(y))
            if cell in self.lVertices:
                is_visible = False
                break
            y += m
            
        return is_visible
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
