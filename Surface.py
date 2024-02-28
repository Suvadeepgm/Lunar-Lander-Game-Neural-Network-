import pygame, random, math


class Surface(pygame.sprite.Sprite):

    def __init__(self, screen_dimension):
        pygame.sprite.Sprite.__init__(self)
        # create the points for the polygon
        landing_pad_points = self.build_landing_pad(100, screen_dimension[1] * 0.1, screen_dimension[0],
                                                    screen_dimension[1])
        self.polygon_points = self.random_ground(screen_dimension[1], screen_dimension[0], 20, 50,
                                                    landing_pad_points)
        # create the canvas where the polygon will be painted, make it 
        self.image = pygame.Surface([screen_dimension[0], screen_dimension[1]])
        self.image.fill((255,255,255))
        self.image.set_colorkey((255, 255, 255))
        # create the polygon using the random points
        self.polygon_rect = pygame.draw.polygon(self.image, (192,192,192), self.polygon_points)        

        landing_pad_rect = self.get_landing_platform_rect(landing_pad_points, 20)
        self.landing_pad = pygame.draw.rect(self.image, (0, 255, 0), landing_pad_rect)
        self.centre_landing_pad = ((landing_pad_points[0][0]+landing_pad_points[1][0])/2,(landing_pad_points[0][1]+landing_pad_points[1][1])/2)
        self.rect = self.image.get_rect()

    def random_ground(self, screen_height, screen_width, spacing, variation, landing_pad):
        # set out the boundaries
        highest_point = screen_height - (screen_height / 8)
        lowest_point = screen_height + 10
        left_most_point = 0
        right_most_point = screen_width + 1
        ans = [(left_most_point, highest_point)]
        number_of_points = screen_width / spacing
        i = 0

        while i < number_of_points:
            rand = random.random()
            rand = rand * variation
            last_x_point = ans[i][0]
            if last_x_point > landing_pad[0][0] and last_x_point < landing_pad[1][0]:
                ans[-1] = ans[-2]
                ans.append(landing_pad[0])
                ans.append(landing_pad[1])
                i = i + 2
                continue
            next_y_point = highest_point - rand
            ans.append((last_x_point + spacing, + next_y_point))
            i = i + 1

        ans.append((right_most_point, highest_point))
        ans.append((right_most_point, lowest_point))
        ans.append((left_most_point, lowest_point))
        return ans

    def build_landing_pad(self, width, height, screen_width, screen_height):
        # width in pixels
        buffer = screen_width * 0.05
        max = screen_width - buffer - width
        min = buffer
        rand = random.random()
        starting_point = max - (rand * (max - min))
        return [(starting_point, screen_height - height), (starting_point + width, screen_height - height)]

    def get_landing_platform_rect(self, landing_pad_points, height):
        x = landing_pad_points[0][0]
        y = landing_pad_points[0][1] - height/2
        width = landing_pad_points[1][0]-landing_pad_points[0][0]

        return pygame.Rect(x,y,width,height)