# Generic Genetic Algorithm

This repository contains the python code for abstract & general implementation of the Genetic Algorithm. It can be used to solve all kinds of relevant problems, such as n-queens, 8-puzzle, 15-puzzle, target string problem, knapsack problem, and all those problems which features deterministic state change (deterministic transition) and are quantifiable by a reward function.

## Overview

A Genetic Algorithm is a search heuristic that is inspired by Charles Darwin's theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation. It can be used to solve many problems; some of the examples are given below.

## Generic Implementation

A genetic algorithm primarily consists of three steps: Selection, Crossover and Mutation. I have programmed different methods for executing these steps, which can be set as a hyperparameter when instantiating the class.

The following Genetic Algorithm implementation provides multiple options for selection, crossover
and mutation steps; below is the definition for various arguments that you can use for these methods.
It may be a better practice to pass these arguments in the form of a dictionary, for e.g.:
self.run_simulation(\*\*kwargs). It is not compulsory to provide these arguments at all, as default values are
already provided for them.

### Selection

- selection_type = 'a' : Returns the most fittest and second most fittest genotype.

- selection_type = 'b' : Returns 2 genotypes based on the probabilities derived from fitness, where probability[i] = i/sum(fitness), where i is a single genotype.

### Crossover

- crossover_type_1 = 'a' : Single Point Crossover : A crossover point on the parent organism string
  is selected. All data beyond that point in the organism string is swapped between the two parent organisms.
  Strings are characterized by Positional Bias.

- crossover_type_1 = 'b' : Two Point Crossover : This is a specific case of a N-point Crossover
  technique. Two random points are chosen on the individual chromosomes (strings), and the genetic material
  is exchanged at these points.

- crossover_type_1 = 'c' : Uniform Crossover : Each gene is selected randomly from one of
  the corresponding genes of the parent chromosomes.

- crossover_type_2 = 'a': This produces two offsprings by alternating the selected crossover step
  for each one of them.

- crossover_type_2 = 'b': This produces two offsprings by alternating the selected crossover step
  with some probability for each one of them.

### Mutation

- mutation_type = 'a' : Random Resetting : In this, a random value from the set of
  permissible values is assigned to a randomly chosen gene.

- mutation_type = 'b': Swap Mutation: we select two positions on the chromosome at random
  and interchange the values. This is common in permutation-based encodings.

- mutation_type = 'c' : Scramble Mutation: from the entire chromosome, a subset of genes is
  chosen, and their values are scrambled or shuffled randomly.

- mutation_type = 'd': Inversion Mutation: We select a subset of genes like in scramble mutation,
  but instead of shuffling the subset, we merely invert the entire string in the subset.

- mutation_type = 'e' : Probability Random Resetting : In this, a random value from the set of
  permissible values are assigned to all the genes in a genotype with probability 1/(length of genotype) or
  some explicitly set probability.

- mutation_type_e_prob: This is only required if mutation_type = 'e' is selected and if you
  want to give explicit probability for this method. The probability values should be in between
  0 and 1.

## Examples

Starting with a random chromosome: a random sequence of genes from a given length gene pool, the algorithm applies evolutionary steps to reach the target chromosome described by the problem. Some of the examples in this repository:

- Target String: Given a target string, the algorithm starts with a random string and converts it into the target one through a series of Genetic Algorithm steps (Selection, Crossover and Mutation). Each character (out of all possible ones) is represented as a single gene. The fitness function evaluates the number of correct characters in the given string present at the correct index.

- N Queen: Given a nxn chessboard, the problem asks to arrange n queens on the board such that no queen is attacking or being attacked by any other one. The genetic algorithm solves this problem by assuming each queen's vertical position (column no. on the board) as a single gene. The index of that gene in the n-length chromosome represents the given queen's horizontal position (row no. on the board). The fitness function evaluates the correct position of each of the n-queens in the given chromosome.

- 8 Puzzle: Given a solvable random state of a 8 puzzle (3x3 board size), the algorithm attempts to finds the correct sequence of steps : which includes 'u' (white space up), 'd' (white space down), 'l' (white space left) & 'r' (white space right), that when executed in the given order, solves the puzzle. Here each action (u, d, l or r) is represented as a gene. The length of the chromosome or solution, which consists of a sequence of actions, can be set as a hyperparameter when instantiating the class. The chromosome corresponding to the target fitness represents the correct sequence of actions to solve the puzzle.

- 15 puzzle: The problem & solution concept is identical to the previous problem (8 Puzzle). The only difference is that the board is of 4x4 size, which implies that the puzzle becomes exponentially more complex, so is the solution.

Open **genetic_algorithm.ipynb** notebook to find these examples. Run the following commands to see them.

```
git clone https://github.com/1998apoorvmalik/generic-genetic-algorithm.git
cd generic-genetic-algorithm
jupyter notebook
```

Now open **genetic_algorithm.ipynb**

## Author

[Apoorv Malik](https://github.com/1998apoorvmalik)
