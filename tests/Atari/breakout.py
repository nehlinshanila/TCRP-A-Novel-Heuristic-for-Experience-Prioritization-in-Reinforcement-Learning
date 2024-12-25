import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Breakout Game')
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Ball settings
ball_speed_x, ball_speed_y = 3, 3  # Adjusted speed for better control
ball = pygame.Rect(SCREEN_WIDTH // 2 - 15, SCREEN_HEIGHT // 2 - 15, 30, 30)

# Paddle settings
paddle = pygame.Rect(SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT - 30, 120, 10)
PADDLE_SPEED = 6

# Brick settings
brick_width = SCREEN_WIDTH // 10 - 5
brick_height = 20
bricks = [pygame.Rect(i * (brick_width + 5), 0, brick_width, brick_height) for i in range(10)]

def move_ball():
    global ball_speed_x, ball_speed_y
    
    ball.x += ball_speed_x
    ball.y += ball_speed_y
    
    # Ball collision with walls
    if ball.left <= 0 or ball.right >= SCREEN_WIDTH:
        ball_speed_x *= -1
    if ball.top <= 0:
        ball_speed_y *= -1

    # Ball collision with the paddle
    if ball.colliderect(paddle) and ball_speed_y > 0:
        ball_speed_y *= -1

    # Reset ball position if it goes below the screen
    if ball.bottom > SCREEN_HEIGHT:
        ball.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        ball_speed_x = 3
        ball_speed_y = -3

def move_paddle():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle.left > 0:
        paddle.x -= PADDLE_SPEED
    if keys[pygame.K_RIGHT] and paddle.right < SCREEN_WIDTH:
        paddle.x += PADDLE_SPEED

def check_collision():
    global ball_speed_y
    ball_collide = ball.collidelist(bricks)
    if ball_collide >= 0:
        ball_speed_y *= -1
        bricks.pop(ball_collide)

def game_loop():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        move_ball()
        move_paddle()
        check_collision()
        
        screen.fill(BLACK)
        pygame.draw.ellipse(screen, WHITE, ball)
        pygame.draw.rect(screen, WHITE, paddle)
        for brick in bricks:
            pygame.draw.rect(screen, RED, brick)
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    game_loop()
