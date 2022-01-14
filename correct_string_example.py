# flake8: noqa
from genetic_algorithm import GeneticAlgorithm


class CorrectStringExample(GeneticAlgorithm):
    def __init__(self, string=None, display_documentation=False):
        self.possible_str = list(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890., -;:_!#%&/()=?@${[]}'"
        )
        self.example_string = "Hustle always."
        if string == None:
            self.target_string = self.example_string
        self.target_string = string
        self.string_length = len(self.target_string)
        self.target_score = self.string_length

        super(CorrectStringExample, self).__init__(
            self.possible_str, self.string_length, self.target_score, display_documentation=display_documentation
        )

    def compute_fitness(self, string):
        fitness = 0
        for i in range(self.string_length):
            if string[i] == self.target_string[i]:
                fitness += 1
        return fitness

    def display(self, string=None):
        if string == None:
            string = self.target_string
        string = "".join(string)
        print(string, "\nFitness: {}".format(self.compute_fitness(string)))
