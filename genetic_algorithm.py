# flake8: noqa

import random
from abc import ABC, abstractmethod

import numpy as np


class GeneticAlgorithm(ABC):
    def __init__(
        self, gene_pool, genotype_length, target_max_fitness, target_avg_fitness=None, display_documentation=True
    ):
        self.gene_pool = gene_pool
        self.gene_pool_length = len(gene_pool)
        self.genotype_length = genotype_length
        self.target_max_fitness = target_max_fitness
        self.target_avg_fitness = target_avg_fitness
        self.selection_message_display = True
        self.crossover_message_display = True
        self.mutation_message_display = True

        if display_documentation:
            self.show_documentation()

    @abstractmethod
    def compute_fitness(self, genotype):
        pass

    # Override this method in your subclass.

    @abstractmethod
    def display(self, genotype):
        pass

    # Override this method in your subclass.

    def run_simulation(
        self,
        population_size=10,
        stats_frequency=100,
        selection_type="a",
        crossover_type_1="a",
        crossover_type_2="b",
        mutation_type="a",
        mutation_type_e_prob=0.02,
    ):
        self.selection_message_display = True
        self.crossover_message_display = True
        self.mutation_message_display = True

        print("Target Fitness = {}".format(self.target_max_fitness))
        print("Simulation started with following options:\n")

        population = self.generate_population(population_size)
        population_fitness = [self.compute_fitness(genotype) for genotype in population]
        max_population_fitness = max(population_fitness)
        avg_population_fitness = np.mean(population_fitness)
        generation = 0

        while True:
            # Selection
            parents = []
            for _ in range(int(population_size / 2)):
                parents.append(self.selection(population, type=selection_type))

            # Crossover
            children = []
            for parent in parents:
                offsprings = self.crossover(parent[0], parent[1], type1=crossover_type_1, type2=crossover_type_2)
                for offspring in offsprings:
                    children.append(offspring)

            # Mutation
            mutated_children = []
            for child in children:
                mutated_children.append(self.mutation(child, prob=mutation_type_e_prob, type=mutation_type))

            generation += 1
            population = mutated_children
            population_fitness = [self.compute_fitness(genotype) for genotype in population]
            max_population_fitness = max(population_fitness)
            avg_population_fitness = np.mean(population_fitness)
            max_fit_index = np.argmax(population_fitness)

            if generation == 1:
                print(
                    "Generation : {}, Population size = {}, Max population fitness = {}, Avg. population fitness = {}".format(
                        generation, population_size, max_population_fitness, avg_population_fitness
                    )
                )
            if generation % stats_frequency == 0:
                print(
                    "Generation : {}, Population size = {}, Max population fitness = {}, Avg. population fitness = {}".format(
                        generation, population_size, max_population_fitness, avg_population_fitness
                    )
                )

            if max_population_fitness >= self.target_max_fitness and (
                self.target_avg_fitness is None or avg_population_fitness >= self.target_avg_fitness
            ):
                break

        print(
            "\nSolution found after {} generations, Max population fitness = {}, Avg. population fitness = {}\n".format(
                generation, max_population_fitness, avg_population_fitness
            )
        )

        self.display(population[max_fit_index])

        return population, population[max_fit_index]

    def show_documentation(self):
        print("\t\t\t\t\tGENETIC ALGORITHM\n\n")

        print("You don't need to execute any method other than self.run_simulation(), (p.s: try it at your own risk).")
        print("The following Genetic Algorithm implementation provides multiple options for the selection, crossover")
        print("and mutation steps, below is the definition for various arguments that you can use for these methods.")
        print("It may be a better practice to pass these arguments in the form of a dictionary, for eg:")
        print(
            "self.run_simulation(**kwargs). It is not compulsory to provide these arguments at all, as default values are"
        )
        print("already provided for them.\n\n")

        print("Argument : population_size : Specifies population size for each generation.\n")
        print("Argument : stats_frequency : Specifies how frequently should the simulation statistics")
        print("be displayed.\n\n")

        print("Selection methods, and corresponding arguments ->\n")
        print("Argument : selection_type = 'a' : Returns the most fittest and second most fittest genotype.\n")
        print("Argument : selection_type = 'b' : Returns 2 genotypes based on the probabilities derived from fitness")
        print("where probability[i] = i/sum(fitness), where i is a single genotype.\n\n")

        print("Crossover methods, and corresponding arguments ->\n")

        print(
            "Argument : crossover_type_1 = 'a' : Single Point Crossover : A crossover point on the parent organism string"
        )
        print(
            "is selected. All data beyond that point in the organism string is swapped between the two parent organisms."
        )
        print("Strings are characterized by Positional Bias.\n")

        print(
            "Argument : crossover_type_1 = 'b' : Two Point Crossover : This is a specific case of a N-point Crossover"
        )
        print(
            "technique. Two random points are chosen on the individual chromosomes (strings) and the genetic material"
        )
        print("is exchanged at these points.\n")

        print("Argument : crossover_type_1 = 'c' : Uniform Crossover : Each gene is selected randomly from one of")
        print("the corresponding genes of the parent chromosomes.\n")

        print(
            "Argument : crossover_type_2 = 'a' : This produces two offsprings by alternating the selected crossover step"
        )
        print("for each one of them.\n")

        print(
            "Argument : crossover_type_2 = 'b' : This produces two offsprings by alternating the selected crossover step"
        )
        print("with some probability for each one of them.\n\n")

        print("Mutation methods, and corresponding arguments ->\n")

        print("Argument : mutation_type = 'a' : Random Resetting : In this, a random value from the set of")
        print("permissible values is assigned to a randomly chosen gene.\n")

        print("Argument : mutation_type = 'b' : Swap Mutation : we select two positions on the chromosome at random,")
        print("and interchange the values. This is common in permutation based encodings.\n")

        print("Argument : mutation_type = 'c' : Scramble Mutation: from the entire chromosome, a subset of genes is")
        print("chosen and their values are scrambled or shuffled randomly.\n")

        print(
            "Argument : mutation_type = 'd' : Inversion Mutation : We select a subset of genes like in scramble mutation,"
        )
        print("but instead of shuffling the subset, we merely invert the entire string in the subset.\n")

        print("Argument : mutation_type = 'e' : Probability Random Resetting : In this, a random value from the set of")
        print(
            "permissible values is assigned to all the genes in a genotype with probability 1/(length of genotype) or"
        )
        print("some explicitly set probability.\n")

        print("Argument : mutation_type_e_prob : This is only required if mutation_type = 'e' is selected, and if you")
        print("want to give some explicit probability for this method. The probability values should be in between")
        print("0 and 1.\n")

    def generate_population(self, size=2):
        population = []
        for _ in range(size):
            genotype = []

            for _ in range(self.genotype_length):
                genotype.append(np.random.choice(self.gene_pool, size=1, replace=False)[0])
            population.append(genotype)

        return population

    def selection(self, population, type="a"):
        assert type == "a" or type == "b", "Invalid 'type' argument in selection method."

        fitness = []
        for genotype in population:
            fitness.append(self.compute_fitness(genotype))

        """
        Type 'a' : Returns the most fittest and second most fittest genotype.
        """
        if type == "a":
            if self.selection_message_display:
                print("1)\tType 'a' Selection method")
                self.selection_message_display = False
            fittest_index = np.argmax(fitness)
            fitness.pop(fittest_index)

            return [population[fittest_index], population[np.argmax(fitness)]]

        """
        Type 'b' : Returns 2 genotypes based on the probabilities derived from fitness where probability[i] = i/sum(fitness),
        where i is a single genotype.
        """
        if type == "b":
            if self.selection_message_display:
                print("1)\tType 'b' Selection method")
                self.selection_message_display = False
            probs = list()
            temp_sum = sum(fitness)
            for i in fitness:
                probs.append(i / temp_sum)

            index = np.random.choice(len(population), size=2, replace=True, p=probs)
            return [population[index[0]], population[index[1]]]

    def single_point_crossover(self, first_genotype, second_genotype, crossover_index, type="a"):
        assert type == "a" or type == "b", "Invalid 'type' argument in single_point_crossover."
        offsprings = []

        if type == "a":
            offsprings.append(first_genotype[:crossover_index] + second_genotype[crossover_index:])
            offsprings.append(second_genotype[:crossover_index] + first_genotype[crossover_index:])

        if type == "b":
            for _ in range(2):
                if random.randint(1, 2) == 1:
                    offsprings.append(first_genotype[:crossover_index] + second_genotype[crossover_index:])
                else:
                    offsprings.append(second_genotype[:crossover_index] + first_genotype[crossover_index:])

        return offsprings

    def two_point_crossover(
        self, first_genotype, second_genotype, first_crossover_index, second_crossover_index, type="a"
    ):
        assert type == "a" or type == "b", "Invalid 'type' argument in two_point_crossover."
        offsprings = []

        if type == "a":
            offsprings.append(
                first_genotype[:first_crossover_index]
                + second_genotype[first_crossover_index:second_crossover_index]
                + first_genotype[second_crossover_index:]
            )
            offsprings.append(
                second_genotype[:first_crossover_index]
                + first_genotype[first_crossover_index:second_crossover_index]
                + second_genotype[second_crossover_index:]
            )

        if type == "b":
            for _ in range(2):
                if random.randint(1, 2) == 1:
                    offsprings.append(
                        first_genotype[:first_crossover_index]
                        + second_genotype[first_crossover_index:second_crossover_index]
                        + first_genotype[second_crossover_index:]
                    )
                else:
                    offsprings.append(
                        second_genotype[:first_crossover_index]
                        + first_genotype[first_crossover_index:second_crossover_index]
                        + second_genotype[second_crossover_index:]
                    )

        return offsprings

    def uniform_crossover(self, first_genotype, second_genotype):
        offsprings = []

        for _ in range(2):
            genotype = []
            for i in range(self.genotype_length):
                if random.randint(1, 2) == 1:
                    genotype.append(first_genotype[i])
                else:
                    genotype.append(second_genotype[i])
            offsprings.append(genotype)

        return offsprings

    def crossover(
        self,
        first_genotype,
        second_genotype,
        first_crossover_index=None,
        second_crossover_index=None,
        min_crossover_index=None,
        max_crossover_index=None,
        type1="a",
        type2="a",
    ):

        assert (type1 == "a" or type1 == "b" or type1 == "c") and (
            type2 == "a" or type2 == "b"
        ), "Invalid 'type' argument in crossover method."

        if min_crossover_index == None:
            min_crossover_index = 0
        if max_crossover_index == None:
            max_crossover_index = self.genotype_length - 1

        # Check if everything is valid.
        assert len(first_genotype) == len(
            second_genotype
        ), "The length of the two genotypes must be equal for the crossover to happen."
        assert (
            min_crossover_index >= 0 and max_crossover_index <= self.genotype_length - 1
        ), "The minmin_crossover_index >= 0 and max_crossover_index <= self.genotype_length"

        if first_crossover_index != None:
            assert (
                first_crossover_index >= min_crossover_index and first_crossover_index <= max_crossover_index
            ), "Invalid first crossover index."
        else:
            first_crossover_index = random.randint(min_crossover_index, max_crossover_index)

        if second_crossover_index != None:
            assert (
                second_crossover_index >= min_crossover_index
                and second_crossover_index <= max_crossover_index
                and second_crossover_index > first_crossover_index
            ), "Invalid second crossover index."
        else:
            second_crossover_index = random.randint(first_crossover_index, max_crossover_index)

        """
        Type 'a' => Single Point Crossover : A crossover point on the parent organism string is selected. 
        All data beyond that point in the organism string is swapped between the two parent organisms. 
        Strings are characterized by Positional Bias.
        """
        if type1 == "a":
            if self.crossover_message_display:
                print("2)\tSingle Point Crossover method")
                self.crossover_message_display = False

            return self.single_point_crossover(first_genotype, second_genotype, first_crossover_index, type2)

        """
        Type 'b' => Two Point Crossover : This is a specific case of a N-point Crossover technique. 
        Two random points are chosen on the individual chromosomes (strings) and the genetic material
        is exchanged at these points.
        """
        if type1 == "b":
            if self.crossover_message_display:
                print("2)\tTwo Point Crossover method")
                self.crossover_message_display = False

            return self.two_point_crossover(
                first_genotype, second_genotype, first_crossover_index, second_crossover_index, type2
            )

        """
        Type 'c' => Uniform Crossover : Each gene is selected randomly from one of the corresponding genes of the 
        parent chromosomes.
        """
        if type1 == "c":
            if self.crossover_message_display:
                print("2)\tUniform Crossover method")
                self.crossover_message_display = False

            return self.uniform_crossover(first_genotype, second_genotype)

    def mutation(self, genotype, prob=None, type="a"):
        assert (
            type == "a" or type == "b" or type == "c" or type == "d" or type == "e"
        ), "Invalid 'type' argument in selection."

        """
        Type 'a' => Random Resetting : In this, a random value from the set of permissible values is assigned 
        to a randomly chosen gene.
        """
        if type == "a":
            if self.mutation_message_display:
                print("3)\tRandom Resetting mutation method\n")
                self.mutation_message_display = False

            mutate_index = random.randrange(0, self.genotype_length)
            genotype[mutate_index] = self.gene_pool[random.randrange(0, self.gene_pool_length)]
            return genotype

        """
        Type 'b' => Swap Mutation : we select two positions on the chromosome at random, and interchange the values. 
        This is common in permutation based encodings.
        """
        if type == "b":
            if self.mutation_message_display:
                print("3)\tSwap Mutation method\n")
                self.mutation_message_display = False

            first_mutate_index = random.randrange(0, self.genotype_length - 1)
            second_mutate_index = random.randrange(first_mutate_index, self.genotype_length)

            genotype[first_mutate_index], genotype[second_mutate_index] = (
                genotype[second_mutate_index],
                genotype[first_mutate_index],
            )
            return genotype

        """
        Type 'c' => Scramble Mutation: from the entire chromosome, a subset of genes is chosen and 
        their values are scrambled or shuffled randomly.
        """
        if type == "c":
            if self.mutation_message_display:
                print("3)\tScramble Mutation method\n")
                self.mutation_message_display = False

            first_mutate_index = random.randrange(0, self.genotype_length - 1)
            second_mutate_index = random.randrange(first_mutate_index, self.genotype_length)

            for mutate_index in range(first_mutate_index, second_mutate_index + 1):
                genotype[mutate_index] = self.gene_pool[random.randrange(0, self.gene_pool_length)]
            return genotype

        """
        Type 'd' => Inversion Mutation : We select a subset of genes like in scramble mutation,
        but instead of shuffling the subset, we merely invert the entire string in the subset.
        """
        if type == "d":
            if self.mutation_message_display:
                print("3)\tInversion Mutation method\n")
                self.mutation_message_display = False

            first_mutate_index = random.randrange(0, self.genotype_length - 1)
            second_mutate_index = random.randrange(first_mutate_index, self.genotype_length)

            temp = genotype[first_mutate_index : second_mutate_index + 1]
            return genotype[0:first_mutate_index] + temp[::-1] + genotype[second_mutate_index + 1 :]

        """
        Type 'e' => Probability Random Resetting : In this, a random value from the set of permissible values is assigned 
        to all the genes in a genotype with probability 1/(length of genotype) or some explicitly set probability.
        """
        if type == "e":
            if prob == None:
                prob = 1 / self.genotype_length

            if self.mutation_message_display:
                print("3)\tProbability Random Resetting Mutation method, with mutation probability = {}\n".format(prob))
                self.mutation_message_display = False

            prob = prob * 100
            for mutate_index in range(self.genotype_length):
                if random.randint(1, 100) <= prob:
                    genotype[mutate_index] = self.gene_pool[random.randrange(0, self.gene_pool_length)]
            return genotype
