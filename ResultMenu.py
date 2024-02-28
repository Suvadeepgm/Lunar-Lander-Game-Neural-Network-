import pygame

class ResultMenu:

    def __init__(self, screen_dimension):
        self.colors=[(0,0,0), (128,128,128), (255,255,255), (255,0,0), (0,255,0)]
        self.top_middle=(screen_dimension[0]/2, screen_dimension[1]/2)
        self.screen_dimension = screen_dimension
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)

        FONT = pygame.font.SysFont("Times New Norman", 60)

        text_buttons = [FONT.render("You Won!", True, self.colors[2]),
                        FONT.render("Sorry, You Lost.", True, self.colors[2]),
                        FONT.render("Go Back to Main Menu", True, self.colors[2]),
                        ]
        rect_buttons = [pygame.Rect(self.top_middle[0]-150, self.top_middle[1], 300, 80),
                        pygame.Rect(self.top_middle[0]-150, self.top_middle[1], 300, 80),
                        pygame.Rect(self.top_middle[0]-250, self.top_middle[1]+100, 500, 80),
                        ]
        
        self.buttons = [
            [text_buttons[0], rect_buttons[0], self.colors[4]],
            [text_buttons[1], rect_buttons[1], self.colors[3]],
            [text_buttons[2], rect_buttons[2], self.colors[0]],
        ]

    def draw_result_objects(self, screen, result, score):
        screen.fill(self.colors[2])
        if (result):
            score = int(score * 100) / 100
            score_surface = self.myfont.render("Score: " + str(score), False, (0, 0, 0))
            pygame.draw.rect(screen, self.buttons[0][2], self.buttons[0][1])
            screen.blit(score_surface, ((self.screen_dimension[0] / 2) - 100, (self.screen_dimension[1] /2) - 50))
            screen.blit(self.buttons[0][0], self.buttons[0][1])
        else:
            pygame.draw.rect(screen, self.buttons[1][2], self.buttons[1][1])
            screen.blit(self.buttons[1][0], self.buttons[1][1])
        
        pygame.draw.rect(screen, self.buttons[2][2], self.buttons[2][1])
        screen.blit(self.buttons[2][0], self.buttons[2][1])

    def check_hover(self, event):
        if event.type == pygame.MOUSEMOTION:
            # only object 2 is a button, the first two are text
            if self.buttons[2][1].collidepoint(event.pos):
                self.buttons[2][2] = self.colors[1]
            else:
                self.buttons[2][2] = self.colors[0]

    def check_back_main_menu(self, event):
        # mouse button was clicked
        if event.type == pygame.MOUSEBUTTONDOWN:
            # 1 == left mouse button, 2 == middle button, 3 == right button
            if event.button == 1:
                return self.buttons[2][1].collidepoint(event.pos)