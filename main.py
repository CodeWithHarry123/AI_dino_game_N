import pygame
import os
import random
import neat
import sys

# --- Constants ---
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 600
FPS = 30
FONT_SIZE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Asset Loader ---
def load_image(name, fallback_color=(0, 0, 0), size=(50, 50)):
    """
    Loads an image from the assets directory. 
    If not found, returns a colored surface (fallback).
    """
    path = os.path.join(ASSETS_DIR, name)
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = pygame.image.load(path)
        # Scale if necessary (optional, but good for consistency) 
        # img = pygame.transform.scale(img, size) 
        return img
    except Exception as e:
        print(f"Warning: Could not load {name}. Using fallback shape. Error: {e}")
        surface = pygame.Surface(size)
        surface.fill(fallback_color)
        return surface

# Global Assets (Loaded in main to initialize pygame first)
DINO_IMG = None
CACTUS_IMG = None
BIRD_IMG = None
BG_IMG = None

# --- Classes ---
class Dino:
    X_POS = 80
    Y_POS = 310
    JUMP_VEL = 8.5

    def __init__(self, img=None):
        self.img = img if img else pygame.Surface((50, 50))
        if not img: self.img.fill((0, 255, 0)) # Green if no image
        
        self.dino_run = True
        self.dino_jump = False
        self.jump_vel = self.JUMP_VEL
        self.rect = self.img.get_rect()
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.step_index = 0

    def update(self, userInput):
        if self.dino_jump:
            self.jump()
        elif self.dino_run:
            self.run()

        if self.step_index >= 10:
            self.step_index = 0

        # AI Control (Jump)
        if userInput and not self.dino_jump:
            self.dino_jump = True
            self.dino_run = False

    def jump(self):
        self.rect.y -= self.jump_vel * 4
        self.jump_vel -= 0.8
        if self.jump_vel < -self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.dino_run = True
            self.rect.y = self.Y_POS

    def run(self):
        self.rect.y = self.Y_POS
        self.step_index += 1

    def draw(self, SCREEN):
        SCREEN.blit(self.img, (self.rect.x, self.rect.y))

class Obstacle:
    def __init__(self, img, x_pos, y_pos):
        self.img = img
        self.rect = self.img.get_rect()
        self.rect.x = x_pos
        self.rect.y = y_pos

    def update(self, speed):
        self.rect.x -= speed
        if self.rect.x < -self.rect.width:
            return True # Remove
        return False

    def draw(self, SCREEN):
        SCREEN.blit(self.img, (self.rect.x, self.rect.y))

class SmallCactus(Obstacle):
    def __init__(self, img):
        super().__init__(img, SCREEN_WIDTH + 400, 325) # Approx Y for small cactus

class LargeCactus(Obstacle):
    def __init__(self, img):
        super().__init__(img, SCREEN_WIDTH + 400, 300) # Approx Y for large cactus

class Game:
    def __init__(self):
        self.game_speed = 20
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.obstacles = []
        self.points = 0
    
    def update_background(self, SCREEN, bg_img):
        image_width = bg_img.get_width()
        SCREEN.blit(bg_img, (self.x_pos_bg, self.y_pos_bg))
        SCREEN.blit(bg_img, (image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -image_width:
            self.x_pos_bg = 0
        self.x_pos_bg -= self.game_speed

    def score(self):
        self.points += 1
        return self.points

# --- NEAT Integration ---
def eval_genomes(genomes, config):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    
    # Init Pygame inside evaluation loop if not already (or just once)
    # But usually eval_genomes runs the game loop
    
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    CLOCK = pygame.time.Clock()
    
    # Load assets here if not loaded (to handle re-runs or missing assets gracefully)
    global DINO_IMG, CACTUS_IMG, BG_IMG
    if DINO_IMG is None:
        DINO_IMG = load_image("dino.png", (0, 255, 0), (50, 60))
        CACTUS_IMG = load_image("cactus.png", (255, 0, 0), (50, 70))
        BG_IMG = load_image("track.png", (100, 100, 100), (SCREEN_WIDTH, 10))
        # Ensure BG is wide enough
        if BG_IMG.get_width() < SCREEN_WIDTH:
            BG_IMG = pygame.transform.scale(BG_IMG, (SCREEN_WIDTH, 10))

    nets = []
    dinos = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        dinos.append(Dino(DINO_IMG))
        ge.append(genome)

    obstacles = []
    game = Game()
    
    run = True
    while run and len(dinos) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()

        SCREEN.fill(WHITE)

        # Background
        game.update_background(SCREEN, BG_IMG)

        # Spawn Obstacles
        if len(obstacles) == 0:
            obstacles.append(SmallCactus(CACTUS_IMG) if random.randint(0, 1) == 0 else LargeCactus(CACTUS_IMG))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            if obstacle.update(game.game_speed):
                obstacles.remove(obstacle)
            
            for i, dino in enumerate(dinos):
                if dino.rect.colliderect(obstacle.rect):
                    ge[i].fitness -= 1
                    dinos.pop(i)
                    nets.pop(i)
                    ge.pop(i)

        # AI Logic
        for i, dino in enumerate(dinos):
            ge[i].fitness += 0.1
            dino.update(False) # Default run

            # Inputs to Neural Network
            # 1. Distance to next obstacle
            # 2. Height of next obstacle (optional, but good)
            # 3. Game speed (optional)
            
            output = nets[i].activate((dino.rect.y, abs(dino.rect.x - obstacles[0].rect.x)))
            
            if output[0] > 0.5:
                dino.update(True) # Jump
            
            dino.draw(SCREEN)

        # Score
        text = pygame.font.Font(None, FONT_SIZE).render(f'Points: {game.score()}', True, BLACK)
        SCREEN.blit(text, (1000, 50))
        
        # Statistics
        text_2 = pygame.font.Font(None, FONT_SIZE).render(f'Alive: {len(dinos)}', True, BLACK)
        SCREEN.blit(text_2, (1000, 80))

        pygame.display.update()
        CLOCK.tick(30)

# --- Main Execution ---
def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50) # Run for 50 generations
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    pygame.init()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
