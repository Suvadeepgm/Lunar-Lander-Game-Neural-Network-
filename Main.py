# Main Runnable file for the CE889 Assignment
# Project built by Lewis Veryard and Hugo Leon-Garza
from GameLoop import GameLoop
from FinalNeural import LunarLanderFinalNew

def importConfigFile():
    keys = []
    values = []
    file = open("Files/Config.con", 'r')
    for line in file:
        line_split = line.split(',')
        for individual in line_split:
            individual = individual.replace(" ", "")
            individual = individual.replace("\n", "")
            content = individual.split('=')
            keys.append(content[0])
            values.append(content[1])
    return dict(zip(keys, values))


def start_game_window():
    game = GameLoop()
    game.init(config_data)
    game.main_loop(config_data)


config_data = importConfigFile()
start_game_window()
