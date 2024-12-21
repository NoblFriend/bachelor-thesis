class RunningAverage:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.current_average = None
        self.current_variance = None
        self.count = 0

    def update(self, value):
        self.count += 1
        if self.current_average is None:
            self.current_average = value
            self.current_variance = 0.0
        else:
            previous_average = self.current_average
            self.current_average = (self.alpha * value) + ((1 - self.alpha) * previous_average)
            self.current_variance = (self.alpha * (value - previous_average) ** 2) + ((1 - self.alpha) * self.current_variance)

    def average(self):
        return self.current_average

    def std_dev(self):
        if self.count < 2:
            return 0.0
        return self.current_variance ** 0.5



