# flake8: noqa

import math

import numpy as np

from genetic_algorithm import GeneticAlgorithm


class NQueensExample(GeneticAlgorithm):
    def __init__(self, board_size, avg_accuracy=0.9, display_documentation=True):
        self.board_size = board_size
        self.target_fitness = int(math.factorial(self.board_size) / (2 * math.factorial(self.board_size - 2)))

        genepool = np.arange(1, self.board_size + 1)
        target_max_fitness = self.target_fitness
        target_avg_fitness = avg_accuracy * target_max_fitness
        super(NQueensExample, self).__init__(
            genepool, self.board_size, target_max_fitness, target_avg_fitness, display_documentation
        )

    def compute_fitness(self, board):
        board_size = len(board)
        attacking_pair = []

        # Check if a queens pair is not in the same row.
        for i in range(board_size):
            for j in range(board_size):
                if i != j and board[i] == board[j]:
                    attacking_pair.append(tuple(set([i, j])))

        # Check if a queens pair is not in the same diagonal.
        for i in range(board_size):
            for j in range(board_size):
                if i != j and int(board[i]) - int(board[j]) == abs(i - j):
                    attacking_pair.append(tuple(set([i, j])))

        return self.target_fitness - len(set(attacking_pair))

    def display(self, board):
        board_size = len(board)

        print("board : {}".format(board))
        for c in range(board_size):
            for r in range(board_size):
                if r == int(board[c]) - 1:
                    print("Q", end=" ")
                else:
                    print("*", end=" ")
            print()
        print("Fitness: {}".format(self.compute_fitness(board)))
