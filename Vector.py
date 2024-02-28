import math


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, vector):
        x = self.x + vector.x
        y = self.y + vector.y
        return Vector(x, y)

    def minus(self, vector):
        x = self.x - vector.x
        y = self.y - self.y
        return Vector(x, y)

    def scalar_multiply(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def rotate(self, theta):
        theta = math.radians(theta)
        x = (self.x * math.cos(theta)) - (self.y * math.sin(theta))
        y = (self.x * math.sin(theta)) + (self.y * math.cos(theta))
        return Vector(x, y)

    def normalise_vector(self):
        # returns a vector that only shows the direction of travel
        distance = self.length()
        x = self.x / distance
        y = self.y / distance
        return Vector(x, y)

    def length(self):
        x2 = self.x * self.x
        y2 = self.y * self.y
        return math.sqrt((x2 + y2))

    def print(self):
        print("X = " + str(self.x) + " Y = " + str(self.y))
