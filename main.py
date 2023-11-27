import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 700, 500
GRID_SIZE = 30
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create a small maze with a starting point (S) and an ending point (E)
maze = [
    "#####################",
    "#     #   S         #",
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

def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Clear the screen
        screen.fill(WHITE)

        # Draw the maze
        draw_maze()

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(FPS)

if __name__ == "__main__":
    main()
