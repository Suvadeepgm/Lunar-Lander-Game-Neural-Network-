
class DataCollection:

    def __init__(self, all_data):
        self.data_file = open("ce889_dataCollection.csv", "a")
        self.data_file.close()
        self.buffer = []
        self.all_data = (all_data == "TRUE")

    def get_input_row(self, lander, surface, controller):
        # inputs
        current_velocity = lander.velocity
        current_speed = current_velocity.length()
        current_angle = lander.current_angle
        x_target = surface.centre_landing_pad[0] - lander.position.x
        y_target = surface.centre_landing_pad[1] - lander.position.y
        dist_to_surface = surface.polygon_rect.topleft[1] - lander.position.y

        # create comma separated string row
        if self.all_data:
            input_row = str(current_speed)+"," + \
                        str(current_velocity.x) + "," + \
                        str(current_velocity.y) + "," + \
                        str(current_angle) + "," + \
                        str(x_target) + "," + \
                        str(y_target) + "," + \
                        str(dist_to_surface)
        else:
            #input_row = [x_target,y_target]
            input_row = str(x_target) + "," + \
                        str(y_target)

        return input_row

    def save_current_status(self, input_row, lander, surface, controller):
        # open file
        # input_row = self.get_input_row(lander, surface, controller)

        # outputs
        thrust = 0
        if (controller.is_up()):
            thrust = 1
        new_vel_y = lander.velocity.y
        new_vel_x = lander.velocity.x
        
        turning = [0, 0]
        if (controller.is_left()):
            turning = [1,0]
        elif (controller.is_right()):
            turning = [0,1]
        new_angle = lander.current_angle

        # add output values to the string input row
        if self.all_data:
            status_row = input_row + "," + \
                        str(thrust) + "," + \
                        str(new_vel_y) + "," + \
                        str(new_vel_x) + "," + \
                        str(new_angle) + "," + \
                        str(turning[0]) + "," + str(turning[1]) + "\n"
        else:
            status_row = input_row + "," + \
                        str(new_vel_y) + "," + \
                        str(new_vel_x) + "\n"

        # save comma separated row in the file
        self.buffer.append(status_row)


    def write_to_file(self):
        self.data_file = open("ce889_dataCollection.csv", "a")
        
        
        for row in self.buffer:
            self.data_file.write(row)
        self.data_file.close()

    def reset(self):
        self.buffer = []