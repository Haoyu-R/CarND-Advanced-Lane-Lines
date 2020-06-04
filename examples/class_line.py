# Define a class to receive the characteristics of each line detection
import numpy as np


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # recent polynomial coefficients
        self.A_left = []
        self.B_left = []
        self.C_left = []

        self.A_right = []
        self.B_right = []
        self.C_right = []

        self.last_left = np.array([])
        self.last_right = np.array([])

    def insert(self, left, right):
        if len(self.A_left) > 4:
            self.A_left.pop(0)
            self.B_left.pop(0)
            self.C_left.pop(0)
            self.A_right.pop(0)
            self.B_right.pop(0)
            self.C_right.pop(0)

        self.A_left.append(left[0])
        self.B_left.append(left[1])
        self.C_left.append(left[2])
        self.A_right.append(right[0])
        self.B_right.append(right[1])
        self.C_right.append(right[2])

        self.last_left = (self.sumUp(self.A_left), self.sumUp(self.B_left), self.sumUp(self.C_left))
        self.last_right = (self.sumUp(self.A_right), self.sumUp(self.B_right), self.sumUp(self.C_right))

    def sumUp(self, coefficients):
        normalizer = 0
        start = 1
        coefficient = 0
        for i in range(len(coefficients)):
            normalizer += start
            coefficient += coefficients[len(coefficients) - i - 1]*start
            start *= 0.5
        return coefficient/normalizer

    def get_coefficient(self):
        return self.last_left, self.last_right

    def reset(self):
        self.detected = False
        # recent polynomial coefficients
        self.A_left = []
        self.B_left = []
        self.C_left = []

        self.A_right = []
        self.B_right = []
        self.C_right = []

        self.last_left = np.array([])
        self.last_right = np.array([])

    def pop_last(self):
        length = len(self.A_left)
        if length > 2:
            self.A_left.pop(length - 1)
            self.B_left.pop(length - 1)
            self.C_left.pop(length - 1)

            self.A_right.pop(length - 1)
            self.B_right.pop(length - 1)
            self.C_right.pop(length - 1)



    #average x values of the fitted line over the last n iterations
    # self.bestx = None
    #polynomial coefficients averaged over the last n iterations
    # self.best_fit = None
    #polynomial coefficients for the most recent fit
    # self.current_fit = [np.array([False])]
    #radius of curvature of the line in some units
    # self.radius_of_curvature = None
    #distance in meters of vehicle center from the line
    # self.line_base_pos = None
    #difference in fit coefficients between last and new fits
    # self.diffs = np.array([0,0,0], dtype='float')
    #x values for detected line pixels
    # self.allx = None
    #y values for detected line pixels
    # self.ally = None