

class Statistic:
    def __init__(self, domain, name):
        self.domain = domain
        self.name = name

    def __str__(self):
        return self.name

    def get_sensitivity(self):
        pass

    def get_stats_fn(self):
        pass

    def get_differentiable_stats_fn(self):
        pass
