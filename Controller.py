

class Controller:
    def __init__(self):
        self.up = False
        self.left = False
        self.right = False
        self.mouse = False
        self.mouse_pos = ()

    def set_up(self, action):
        self.up = action

    def is_up(self):
        return self.up

    def set_right(self, action):
        self.right = action

    def is_right(self):
        return self.right

    def set_left(self, action):
        self.left = action

    def is_left(self):
        return self.left

    def set_mouse_event(self, pos):
        self.mouse_pos = pos

    def set_mouse_pos(self, pos):
        self.mouse_pos = pos
        # print(pos)

    def get_mouse_pos(self):
        return self.mouse_pos

    def set_mouse(self, action):
        self.mouse = action
