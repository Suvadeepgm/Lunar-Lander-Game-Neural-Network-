import pygame, time, ctypes, sys
from EventHandler import EventHandler
from Lander import Lander
from Controller import Controller
from Vector import Vector
from GameLogic import GameLogic
from Surface import Surface
from MainMenu import MainMenu
from ResultMenu import ResultMenu
from DataCollection import DataCollection
from NeuralNetHolder import NeuralNetHolder

class GameLoop:

    def __init__(self):
        self.controller = Controller()
        self.Handler = EventHandler(self.controller)
        self.object_list = []
        self.game_logic = GameLogic()
        self.fps_clock = pygame.time.Clock()
        self.fps = 60
        self.neuralnet = NeuralNetHolder()
        self.version = "v1.01"
        self.prediction_cycle = 0

    def init(self, config_data):
        # used to initialise the pygame library
        pygame.init()
        if config_data["FULLSCREEN"] == "TRUE":
            user32 = ctypes.windll.user32
            config_data['SCREEN_HEIGHT'] = int(user32.GetSystemMetrics(1))
            config_data['SCREEN_WIDTH'] = int(user32.GetSystemMetrics(0))
            self.screen = pygame.display.set_mode((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']),
                                                  pygame.FULLSCREEN)
        else:
            config_data['SCREEN_HEIGHT'] = int(config_data['SCREEN_HEIGHT'])
            config_data['SCREEN_WIDTH'] = int(config_data['SCREEN_WIDTH'])
            self.screen = pygame.display.set_mode((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        pygame.display.set_caption('CE889 Assignment Template')
        pygame.display.set_icon(pygame.image.load(config_data['LANDER_IMG_PATH']))

    def score_calculation(self):
        score = 1000.0 - (self.surface.centre_landing_pad[0] - self.lander.position.x)
        angle = self.lander.current_angle
        if(self.lander.current_angle == 0):
            angle = 1
        if(self.lander.current_angle > 180):
            angle = abs(self.lander.current_angle - 360)
        score = score / angle
        velocity = 500 - (self.lander.velocity.x + self.lander.velocity.y)
        score = score + velocity

        print("lander difference " + str(self.surface.centre_landing_pad[0] - self.lander.position.x))
        print("SCORE " + str(score))

        return score

    def main_loop(self, config_data):
        pygame.font.init()  # you have to call this at the start,
        # if you want to use this module you need to call pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        
        # create the group for visuals to be updated
        sprites = pygame.sprite.Group()

        # booleans for what the game state is
        on_menus = [True, False, False] # Main, Won, Lost
        game_start = False

        # Game modes: Play Game, Data Collection, Neural Net, Quit
        game_modes = [False, False, False, False]
        
        # The main loop of the window
        background_image = pygame.image.load(config_data['BACKGROUND_IMG_PATH']).convert_alpha()
        background_image = pygame.transform.scale(background_image, (config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))

        data_collector = DataCollection(config_data["ALL_DATA"])
        main_menu = MainMenu((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        result_menu = ResultMenu((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        score = 0
        # Initialize
        while True:
            # menus
            # check if Quit button was clicked
            if (game_modes[len(game_modes)-1]):
                pygame.quit()
                sys.exit()

            # if game is started, initialize all objects
            if game_start:
                self.controller = Controller()
                self.Handler = EventHandler(self.controller)
                sprites = pygame.sprite.Group()
                self.game_start(config_data, sprites)

            if on_menus[0] or on_menus[1] or on_menus[2]:
                if on_menus[1] or on_menus[2]:
                    result_menu.draw_result_objects(self.screen, on_menus[1], score)
                else:
                    main_menu.draw_buttons(self.screen)
                    # draw the version number
                    textsurface = myfont.render(self.version, False, (0, 0, 0))
                    self.screen.blit(textsurface, (0, 0))

                # main_menu.draw_buttons(self.screen)
                for event in pygame.event.get():
                    if on_menus[0]:
                        main_menu.check_hover(event)
                        button_clicked = main_menu.check_button_click(event)
                        main_menu.draw_buttons(self.screen)
                        if button_clicked > -1:
                            game_modes[button_clicked] = True
                            on_menus[0] = False
                            game_start = True
                    
                    elif on_menus[1] or on_menus[2]:
                        result_menu.check_hover(event)
                        on_menus[0] = result_menu.check_back_main_menu(event)
                        result_menu.draw_result_objects(self.screen, on_menus[1], score)
                        if on_menus[0]:
                            on_menus[1] = False
                            on_menus[2] = False
            else:
                # game
                self.Handler.handle(pygame.event.get())
                # check if data collection mode is activated
                if (game_modes[2]):
                    self.prediction_cycle += 1
                    self.prediction_cycle = self.prediction_cycle%2
                    #IF to add predictions every N frame, right now deactivated, every single frame we ask for prediction
                    if (self.prediction_cycle >= 0): 
                        input_row = data_collector.get_input_row(self.lander, self.surface, self.controller)
                        # nn_prediction = [x_vel, y_vel]
                        nn_prediction = self.neuralnet.predict(input_row)
                        # reset controls
                        self.controller.set_up(False)
                        self.controller.set_left(False)
                        self.controller.set_right(False)

                        # check Vel_Y prediction, if lower means lander should go up
                        if (self.lander.velocity.y > nn_prediction[1]):
                            self.controller.set_up(True)
                        
                        # check Vel_X prediction, if prediction is higher than current go right, if lower go left
                        if (self.lander.velocity.x < nn_prediction[0]):
                            self.controller.set_right(True)
                        elif(self.lander.velocity.x > nn_prediction[0]):
                            self.controller.set_left(True)
                        
                        # avoid infinite turning, limit max angle
                        print("current status controller: ", self.controller.up, " -- ", self.controller.left, " -- ", self.controller.right)
                        print("current status lander: ", self.lander.velocity.y, " -- ", self.lander.velocity.x, " -- ", self.lander.velocity.y > nn_prediction[1], " -- ", self.lander.velocity.x < nn_prediction[0], " -- ", self.lander.velocity.y - nn_prediction[1])
                        if (self.lander.current_angle > 30 and self.lander.current_angle < 330):
                            ang_val = (self.lander.current_angle - 30)/(330-30)
                            ang_val = round(ang_val)
                            if (ang_val == 0):
                                self.lander.current_angle = 30
                            else:
                                self.lander.current_angle = 330
                
                # if (game_modes[1]):
                #     data_collector.save_current_status(self.lander, self.surface, self.controller)
                self.screen.blit(background_image,(0,0))
                if(not self.Handler.first_key_press and game_start == True ):
                    self.update_objects()
                    game_start = False

                if (self.Handler.first_key_press):
                    data_input_row = data_collector.get_input_row(self.lander, self.surface, self.controller)
                    self.update_objects()
                    if (game_modes[1]):
                        data_collector.save_current_status(data_input_row, self.lander, self.surface, self.controller)
                    # then update the visuals on screen from the list
                sprites.draw(self.screen)
                # the win state and the score calculation
                if (self.lander.landing_pad_collision(self.surface)):
                    score = self.score_calculation()
                    on_menus[1] = True
                    if(game_modes[1]):
                        data_collector.write_to_file()
                        data_collector.reset()
                # check if lander collided with surface
                elif (self.lander.surface_collision(self.surface) or self.lander.window_collision((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))):
                    on_menus[2] = True
                    data_collector.reset()
                
                if (on_menus[1] or on_menus[2]):
                    game_start = False
                    for i in range(len(game_modes)):
                        game_modes[i] = False


            # surface_sprites.draw(self.screen)
            pygame.display.flip()
            self.fps_clock.tick(self.fps)

    def update_objects(self):
        # update the speeds and positions of the objects in game
        self.game_logic.update(0.2)

    def setup_lander(self, config_data):
        lander = Lander(config_data['LANDER_IMG_PATH'],
                        [config_data['SCREEN_WIDTH'] / 2, config_data['SCREEN_HEIGHT'] / 2], Vector(0, 0),
                        self.controller)
        self.game_logic.add_lander(lander)
        return lander

    def game_start(self, config_data, sprites):
        # Creates the lander object
        self.lander = self.setup_lander(config_data)
        self.surface = Surface((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        sprites.add(self.lander)
        sprites.add(self.surface)

