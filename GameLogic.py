

class GameLogic:
    def __int__(self):
        self

    def add_lander(self, lander):
        self.lander = lander

    def update(self, delta_time):
        self.lander.update_lander(delta_time)