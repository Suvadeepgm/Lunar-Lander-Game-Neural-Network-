import pygame

class MainMenu:


    def __init__(self, screen_dimension):
        self.colors=[(0,0,0), (128,128,128), (255,255,255)]
        self.top_left=(screen_dimension[0]/2, screen_dimension[1]/2)
        
        FONT = pygame.font.SysFont("Times New Norman", 60)
        text_buttons = [FONT.render("Play Game", True, self.colors[2]), 
                        FONT.render("Data Collection", True, self.colors[2]), 
                        FONT.render("Neural Network", True, self.colors[2]), 
                        FONT.render("Quit", True, self.colors[2])]
        rect_buttons = [pygame.Rect(self.top_left[0]-200, self.top_left[1], 400, 80),
                        pygame.Rect(self.top_left[0]-200, self.top_left[1]+100, 400, 80), 
                        pygame.Rect(self.top_left[0]-200, self.top_left[1]+200, 400, 80), 
                        pygame.Rect(self.top_left[0]-200, self.top_left[1]+300, 400, 80)]

        self.buttons = [
            [text_buttons[0], rect_buttons[0], self.colors[0]],
            [text_buttons[1], rect_buttons[1], self.colors[0]],
            [text_buttons[2], rect_buttons[2], self.colors[0]],
            [text_buttons[3], rect_buttons[3], self.colors[0]],
        ]

    def draw_buttons(self, screen):
        screen.fill(self.colors[2])
        for text, rect, color in self.buttons:
            pygame.draw.rect(screen, color, rect)
            screen.blit(text, rect)
    
    def onHover(self, num_button):
        self.buttons[num_button][2] = self.colors[1]
    
    def offHover(self, num_button):
        self.buttons[num_button][2] = self.colors[0]

    def check_hover(self, event):
        if event.type == pygame.MOUSEMOTION:
            for button in self.buttons:
                if (button[1].collidepoint(event.pos)):
                    button[2] = self.colors[1]
                else:
                    button[2] = self.colors[0]

    def check_button_click(self, event):
        # mouse button was clicked
        if event.type == pygame.MOUSEBUTTONDOWN:
            # 1 == left mouse button, 2 == middle button, 3 == right button
            if event.button == 1:
                for i, button in enumerate(self.buttons):
                    if (button[1].collidepoint(event.pos)):
                        return i
        return -1