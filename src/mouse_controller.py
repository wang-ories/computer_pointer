"""
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
"""
import pyautogui


class MouseController:
    def __init__(self, precision, speed):
        pyautogui.FAILSAFE = False
        precision_dict = {'high': 100, 'low': 1000, 'medium': 500}
        speed_dict = {'fast': 1, 'slow': 5, 'medium': 2}

        self.precision = precision_dict[precision]
        self.speed = speed_dict[speed]
        width, height = pyautogui.size()
        pyautogui.moveTo(width/2, height/2)

    def move(self, x, y):
        pyautogui.moveRel(x * self.precision, -1 * y * self.precision, duration=self.speed)
        width, height = pyautogui.size()
        x_position, y_position = pyautogui.position()
        if x_position >= width:
            pyautogui.moveTo(0, height)
        if y_position >= height:
            pyautogui.moveTo(width, 0)
