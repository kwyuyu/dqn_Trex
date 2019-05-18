import os
import sys
import pygame
import random
from pygame import *
import numpy as np

# display option
DISPLAY = True
TXT_DISPLAY = False

# included element
CLOUD = False
SCOREBOARD = True
GROUND = False
PTERA = True
CACTUS = True
HIGHSCOREBOARD = False

# reward
# SUCCESS_REWARD = 5
# ALIVE_REWARD = 1
# JUMP_REWARD = -1
# DEAD_REWARD = -10

SUCCESS_JUMP = 10
SUCCESS_DUCKING = 10
ALIVE_REWARD = 1
JUMP_REWARD = 0
DUCKING_REWARD = 0
DEAD_REWARD = -10


# PYGAME
PTERA_HEIGHT = 62
SPEEDUP = False


if not DISPLAY:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()

scr_size = (width,height) = (600,150)
FPS = 99999999999999999999
gravity = 0.6

black = (0,0,0)
white = (255,255,255)
# background_col = (235,235,235)
background_col = (0,0,0)
high_score = 0

screen = pygame.display.set_mode(scr_size)
clock = pygame.time.Clock()
pygame.display.set_caption("T-Rex Rush")

def load_image(
    name,
    sizex=-1,
    sizey=-1,
    colorkey=None,
    ):

    fullname = os.path.join('sprites', name)
    image = pygame.image.load(fullname)
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))

    return (image, image.get_rect())

def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scalex = -1,
        scaley = -1,
        colorkey = None,
        ):
    fullname = os.path.join('sprites',sheetname)
    sheet = pygame.image.load(fullname)
    sheet = sheet.convert()

    sheet_rect = sheet.get_rect()

    sprites = []

    sizex = sheet_rect.width/nx
    sizey = sheet_rect.height/ny

    for i in range(0,ny):
        for j in range(0,nx):
            rect = pygame.Rect((j*sizex,i*sizey,sizex,sizey))
            image = pygame.Surface(rect.size)
            image = image.convert()
            image.blit(sheet,(0,0),rect)

            if colorkey is not None:
                if colorkey is -1:
                    colorkey = image.get_at((0,0))
                image.set_colorkey(colorkey,RLEACCEL)

            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image,(scalex,scaley))

            sprites.append(image)

    sprite_rect = sprites[0].get_rect()

    return sprites,sprite_rect

def disp_gameOver_msg(retbutton_image,gameover_image):
    retbutton_rect = retbutton_image.get_rect()
    retbutton_rect.centerx = width / 2
    retbutton_rect.top = height*0.52

    gameover_rect = gameover_image.get_rect()
    gameover_rect.centerx = width / 2
    gameover_rect.centery = height*0.35

    screen.blit(retbutton_image, retbutton_rect)
    screen.blit(gameover_image, gameover_rect)

def extractDigits(number):
    if number > -1:
        digits = []
        i = 0
        while(number/10 != 0):
            digits.append(number%10)
            number = int(number/10)

        digits.append(number%10)
        for i in range(len(digits),5):
            digits.append(0)
        digits.reverse()
        return digits

class Dino():
    def __init__(self,sizex=-1,sizey=-1):
        self.images,self.rect = load_sprite_sheet('dino.png',5,1,sizex,sizey,-1)
        self.images1,self.rect1 = load_sprite_sheet('dino_ducking.png',2,1,59,sizey,-1)
        self.rect.bottom = int(0.98*height)
        # self.rect.bottom = 107
        self.rect.left = width/15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0,0]
        self.jumpSpeed = 11.5

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        screen.blit(self.image,self.rect)

    def checkbounds(self):
        if self.rect.bottom > int(0.98*height):
            self.rect.bottom = int(0.98*height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity
            # self.movement[1] = 0

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1)%2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1)%2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2 + 2

        if self.isDead:
           self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index)%2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1
        self.counter = (self.counter + 1)

class Cactus(pygame.sprite.Sprite):
    def __init__(self,speed=5,sizex=-1,sizey=-1):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.images,self.rect = load_sprite_sheet('cacti-small.png',3,1,sizex,sizey,-1)
        self.rect.bottom = int(0.98*height)
        self.rect.left = width + self.rect.width
        self.image = self.images[random.randrange(0,3)]
        self.movement = [-1*speed,0]
        self.passed = False

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()

class Ptera(pygame.sprite.Sprite):
    def __init__(self,speed=5,sizex=-1,sizey=-1):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.images,self.rect = load_sprite_sheet('ptera.png',2,1,sizex,sizey,-1)
        self.ptera_height = [height*0.82,height*0.75,height*0.60]
        self.rect.centery = self.ptera_height[random.randrange(0,3)]
        self.rect.left = width + self.rect.width
        self.rect.top -= 13 # lower than 52
        self.image = self.images[0]
        self.movement = [-1*speed,0]
        self.index = 0
        self.counter = 0
        self.passed = False

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index+1)%2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()


class Ground():
    def __init__(self,speed=-5):
        self.image,self.rect = load_image('ground.png',-1,-1,-1)
        self.image1,self.rect1 = load_image('ground.png',-1,-1,-1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image,self.rect)
        screen.blit(self.image1,self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right

class Cloud(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.image,self.rect = load_image('cloud.png',int(90*30/42),30,-1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1*self.speed,0]

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()

class Scoreboard():
    def __init__(self,x=-1,y=-1):
        self.score = 0
        self.tempimages,self.temprect = load_sprite_sheet('numbers.png',12,1,11,int(11*6/5),-1)
        self.image = pygame.Surface((55,int(11*6/5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = width*0.89
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = height*0.1
        else:
            self.rect.top = y

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self,score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s],self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


class GameState:
    def __init__(self):
        global high_score
        self.gamespeed = 4
        self.playerDino = Dino(44, 47)
        if GROUND:
            self.new_ground = Ground(-1 * self.gamespeed)
        if SCOREBOARD:
            self.scb = Scoreboard()
        if HIGHSCOREBOARD:
            self.highsc = Scoreboard(width * 0.78)
        self.counter = 0

        self.last_obstacle = pygame.sprite.Group()
        if CACTUS:
            self.cacti = pygame.sprite.Group()
        if PTERA:
            self.pteras = pygame.sprite.Group()
        if CLOUD:
            self.clouds = pygame.sprite.Group()
        if CACTUS:
            Cactus.containers = self.cacti
        if PTERA:
            Ptera.containers = self.pteras
        if CLOUD:
            Cloud.containers = self.clouds

        temp_images, temp_rect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        HI_image = pygame.Surface((22, int(11 * 6 / 5)))
        HI_rect = HI_image.get_rect()
        HI_image.fill(background_col)
        HI_image.blit(temp_images[10], temp_rect)
        temp_rect.left += temp_rect.width
        HI_image.blit(temp_images[11], temp_rect)
        HI_rect.top = height * 0.1
        HI_rect.left = width * 0.73

    def frame_step(self, input_actions):
        global high_score
        pygame.event.pump()
        terminal = False
        # if do nothing reward is 1
        reward = ALIVE_REWARD

        # set ptera isDucking to False
        self.playerDino.isDucking = False

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: press jump
        # input_actions[2] == 1: ducking
        if input_actions[1] == 1:
            # if jump but not over obstacle
            reward = JUMP_REWARD
            if self.playerDino.rect.bottom == int(0.98 * height):
                self.playerDino.isJumping = True
                self.playerDino.movement[1] = -1 * self.playerDino.jumpSpeed

        if input_actions[2] == 1:
            reward = DUCKING_REWARD
            if not (self.playerDino.isJumping and self.playerDino.isDead):
                self.playerDino.isDucking = True


        if CACTUS:
            for c in self.cacti:
                c.movement[0] = -1*self.gamespeed
                if pygame.sprite.collide_mask(self.playerDino,c):
                    self.playerDino.isDead = True
                else:
                    def success_jump_c(dino, cacti):
                        left, top, w, h = dino.rect
                        cleft, ctop, cw, ch = cacti.rect
                        cright = cleft + cw
                        if left > cright and not c.passed:
                            c.passed = True
                            return True
                        return False
                    if success_jump_c(self.playerDino, c):
                        reward = SUCCESS_JUMP

        if PTERA:
            for p in self.pteras:
                p.movement[0] = -1*self.gamespeed
                if pygame.sprite.collide_mask(self.playerDino,p):
                    self.playerDino.isDead = True
                else:
                    def success_duck_p(dino, ptera):
                        left, top, w, h = dino.rect
                        pleft, ptop, pw, ph = ptera.rect
                        pright = pleft + pw
                        if left > pright and not p.passed:
                            p.passed = True
                            return True
                        return False
                    def success_jump_p(dino, ptera):
                        left, top, w, h = dino.rect
                        pleft, ptop, pw, ph = ptera.rect
                        pright = pleft + pw
                        if left > pright and not p.passed:
                            p.passed = True
                            return True
                        return False
                    if success_duck_p(self.playerDino, p):
                        reward = SUCCESS_DUCKING
                    if success_jump_p(self.playerDino, p):
                        reward = SUCCESS_JUMP

        # Generate cactus
        if CACTUS and PTERA:
            if len(self.cacti) < 2:
                if len(self.cacti) == 0:
                    if len(self.pteras) == 0:
                        self.cacti.empty()
                        self.cacti.add(Cactus(self.gamespeed, 40, 40))
                    else:
                        for p in self.pteras:
                            if p.rect.right < width*0.7:
                                self.cacti.empty()
                                self.cacti.add(Cactus(self.gamespeed,40,40))
                else:
                    for c in self.cacti:
                        if c.rect.right < width*0.7 and random.randrange(0,50) == 10:
                            if len(self.pteras) == 0:
                                # self.cacti.empty()
                                self.cacti.add(Cactus(self.gamespeed, 40, 40))
                            else:
                                for p in self.pteras:
                                    if p.rect.right < width*0.7:
                                        # self.cacti.empty()
                                        self.cacti.add(Cactus(self.gamespeed, 40, 40))
        elif CACTUS and not PTERA:
            if len(self.cacti) < 2:
                if len(self.cacti) == 0:
                    self.cacti.empty()
                    self.cacti.add(Cactus(self.gamespeed, 40, 40))
                else:
                    for p in self.cacti:
                        if p.rect.right < width * 0.7 and random.randrange(0, 50) == 10:
                            # self.cacti.empty()
                            self.cacti.add(Cactus(self.gamespeed, 40, 40))


        # Generate bird
        if PTERA and CACTUS:
            if len(self.pteras) == 0 and random.randrange(0,200) == 10 and self.counter > 500:
                last_cacti_right = 0
                for c in self.cacti:
                    last_cacti_right = max(last_cacti_right, c.rect.right)
                if last_cacti_right < width*0.7:
                    self.pteras.empty()
                    self.pteras.add(Ptera(self.gamespeed, 46, PTERA_HEIGHT))
        elif PTERA and not CACTUS:
            if len(self.pteras) < 2:
                if len(self.pteras) == 0:
                    self.pteras.empty()
                    self.pteras.add(Ptera(self.gamespeed,46,PTERA_HEIGHT))
                else:
                    for p in self.pteras:
                        if p.rect.right < width*0.7 and random.randrange(0,50) == 10:
                            # self.cacti.empty()
                            self.pteras.add(Ptera(self.gamespeed, 46, PTERA_HEIGHT))


        # For debug cacti
        if TXT_DISPLAY:
            print("Dino: ", self.playerDino.rect)
            if CACTUS:
                for c in self.cacti:
                    print("cacti: ", c.rect)
            if PTERA:
                for p in self.pteras:
                    print("ptera: ", p.rect)


        if CLOUD:
            if len(self.clouds) < 5 and random.randrange(0,300) == 10:
                Cloud(width,random.randrange(height/5,height/2))


        # List block will update every sprite
        self.playerDino.update()
        if CACTUS:
            self.cacti.update()
        if PTERA:
            self.pteras.update()
        if CLOUD:
            self.clouds.update()
        if GROUND:
            self.new_ground.update()
        if SCOREBOARD:
            self.scb.update(self.playerDino.score)
        if HIGHSCOREBOARD:
            self.highsc.update(high_score)

        if pygame.display.get_surface() != None:
            screen.fill(background_col)
            if GROUND:
                self.new_ground.draw()
            if CLOUD:
                self.clouds.draw(screen)
            if SCOREBOARD:
                self.scb.draw()
            if HIGHSCOREBOARD and high_score != 0:
                    self.highsc.draw()
                    screen.blit(HI_image,HI_rect)
            if CACTUS:
                self.cacti.draw(screen)
            if PTERA:
                self.pteras.draw(screen)
            self.playerDino.draw()

            pygame.display.update()
        clock.tick(FPS)

        cur_score = self.playerDino.score
        # check if Dino dead
        if self.playerDino.isDead:
            if self.playerDino.score > high_score:
                high_score = self.playerDino.score
            terminal = True
            self.__init__()
            reward = DEAD_REWARD

        if SPEEDUP:
            # This one will increase speed
            if self.counter%700 == 699:
                if GROUND:
                    self.new_ground.speed -= 1
                self.gamespeed += 1

        self.counter = (self.counter + 1)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        clock.tick(FPS)
        return image_data, reward, terminal, cur_score


def gameplay():
    game_state = GameState()
    action = [0,0,0]
    action[0] = 1
    game_state.frame_step(action)

    duck = False
    while True:
        jump = False
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    jump = True
                    action = [0, 1, 0]
                    game_state.frame_step(action)
                if event.key == pygame.K_DOWN:
                    duck = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    duck = False
        if not jump and duck:
            action = [0, 0, 1]
        else:
            action = [1, 0, 0]
        game_state.frame_step(action)


def main():
    gameplay()


if __name__ == "__main__":
    main()
