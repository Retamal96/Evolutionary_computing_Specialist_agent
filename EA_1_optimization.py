# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import random
from timeit import default_timer as timer


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'EA_1_enemy_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini="yes")



# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
ini = time.time()  # sets time marker


# genetic algorithm params
run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


dom_u = 1
dom_l = -1
npop = 100
gens = 20
mutation_probability = 0.1
last_best = 0


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))



# ----- SELECTION ALGORITHMS ------------
# --INPUTS--
# - population - dimensions: (size_population, nr_weights)
# - fitness_population - dimensions: (size_population, ) -- calculate fitness beforehand!
# - nr of parents - integer (EVEN!)

# --OUTPUT--
# parents - dimensions: (nr_parents, nr_weights)

# performs roulette wheel selection
# returns a list of parents

# roulette wheel selection
def roulette_wheel_selection(population, nrparents, fitness_population):
    parents = random.choices(population, weights=fitness_population, k=nrparents)
    return parents[0],parents[1]



# elitist selection - parameters can be adjusted
# returns a list of parents
def elitist_selection(population, nrparents, fitness_population):
    # rank them from lowest to highest fitness
    population_sorted = population[fitness_population.argsort()]

    # for elitism: let the top 10% mate always
    top_ten_percent = int(0.10 * len(population))

    # for the rest of the parents that need to be chosen, 75% will be "better solutions"
    number_of_better_solutions = int(0.75 * (nrparents - top_ten_percent))
    number_of_lesser_solutions = nrparents - number_of_better_solutions - top_ten_percent

    # if the top ten percent contains more instances than parents that are needed, only use the best ones
    if top_ten_percent > nrparents:
        top_ten_percent = nrparents
        number_of_better_solutions = 0
        number_of_lesser_solutions = 0

    # what we define as good solutions --> here it's the best 50%
    # the top ten can still occur here!
    better_solutions = 0.5

    # at which index the "better solutions" will start
    index_better_solutions = len(population) * better_solutions

    # create an array for the parents
    parents = np.zeros((0, n_vars))

    # put the top ten percent in the list of parents
    # (if the number of parents is too low --> only the fittest ones)
    for i in range(top_ten_percent):
        parents = np.vstack((parents, population_sorted[len(population) - 1 - i]))

    # uniformly choose the parents from the "better" section
    for i in range(number_of_better_solutions):
        index_parent = np.random.randint(index_better_solutions, len(population))
        parents = np.vstack((parents, population_sorted[index_parent]))

    # uniformly choose the parents from the "lesser" section
    for i in range(number_of_lesser_solutions):
        index_parent = np.random.randint(0, index_better_solutions)
        parents = np.vstack((parents, population_sorted[index_parent]))

    return parents


def limits(x):
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x

# ------ NATURAL SELECTION -------
# INPUTS
# old_generation - dimensions: (size_old_gen, nr_weights)
# old_gen_fitness - dimensions: (size_old_gen, )
# offspring - dimensions: (nr_offspring, nr_weights)
# offspring_fitness - dimensions: (nr_offspring, )
# new_pop_size - integer

# OUTPUT
# population_new -  dimensions: (nr_survivors, nr_weights)

# determines which candidate solutions survive
# this is a very basic one, using the mu + lambda selection (keeps the best individuals, age not taken into account)
def survival(old_generation, old_gen_fitness, offspring, offspring_fitness, new_pop_size):
    # append parents and offspring
    population = np.vstack((old_generation, offspring))

    # append fitness parents and fitness offspring
    fitness = np.append(old_gen_fitness, offspring_fitness)

    # rank them from lowest to highest fitness
    population_sorted = population[fitness.argsort()]

    # create a new population
    population_new = np.zeros((0, n_vars))

    # stack the top new_pop_size
    for i in range(new_pop_size):
        population_new = np.vstack((population_new, population_sorted[len(population)-1-i]))

    return population_new

# ----- CROSSOVER ALGORITHMS ------------
# --INPUTS--
# - population - dimensions: (size_population, nr_weights)
# - fitness_population - dimensions: (size_population, ) -- calculate fitness beforehand!
# - mutation type (1: unifornm mutation, 2: non-uniform mutation)
# - nr of parents - integer (EVEN!)
# -  alpha and beta -  parameters to determine the fitness importance in evolution 

# --OUTPUT--
# offsprings - dimensions: (npop, nr_weights)

# performs roulette wheel selection
# returns a numpy array with offsprings

#Blend crossover alpha

def blend_crossover_alpha(population,fitness_population, alfa=0.5, mutation_type=1, nrparents = 2,):
    total_offspring = np.zeros((0, n_vars))

    for p in range(0, population.shape[0], 2):
        # parent selection
        p_1,p_2 = roulette_wheel_selection(population,nrparents,fitness_population)
        
        # offsprign initialization
        n_offspring = 2
        # iterating over offspring to create them
        for offspring_individual in range(0, n_offspring):
            offspring = np.array([])
            for i in range(0, n_vars):
                p_max, p_min = max(p_1[i], p_2[i]), min(p_1[i], p_2[i])
                i = p_max - p_min
                lower_bound = p_min - (alfa * i)
                upper_bound = p_max + (alfa * i)

                offspring = np.append(offspring, np.random.uniform(lower_bound, upper_bound))

            # performs mutation on the newly created offspring (assuming offspring is one instance of the weights)
            if mutation_type == 1:
                uniform_mutation(offspring)
            else:
                non_uniform_mutation(offspring)

            offspring = np.array(list(map(lambda y: limits(y), offspring)))

            offspring = np.reshape(offspring, (-1, n_vars))
            total_offspring = np.vstack((total_offspring, offspring))

    return total_offspring


#Blend crossover alpha-beta\

def blend_crossover_ab(population, fitness_population, alpha=0.35, beta=0.65):
    # parent selection
    total_offspring = np.zeros((0,n_vars))
    parent_list = elitist_selection(population,50, fitness_population)
    i = 0
    while (i + 1) < len(parent_list):
        # choose parents
        fittest_parent = parent_list[i]
        not_fittest_parent = parent_list[i+1]

        if simulation(env, not_fittest_parent) > simulation(env,fittest_parent):
            fittest_parent, not_fittest_parent = not_fittest_parent, fittest_parent

        # create a child
        n_offsprings = 2
        #d = np.zeros((0,n_vars))

        offspring = np.zeros((n_offsprings, n_vars))
        for f in range(0, n_offsprings):
            for gen in range(0, n_vars):
                d = abs(not_fittest_parent[gen] - fittest_parent[gen])
                offspring[f] = np.random.uniform(not_fittest_parent[gen] - alpha * d,fittest_parent[gen] + d * beta)
            total_offspring = np.vstack((total_offspring, offspring[f]))
        i = i + 2

    return total_offspring



# ---- made two different types of mutation (can both be used!) ----

# uniform mutation
# changes the allele into a randomly chosen real value within the bounds
def uniform_mutation(offspring):
    np.random.uniform(dom_l, dom_u)

    # it's an ndarray - change independently!!
    for i in range(0, len(offspring)):
        if mutation_probability > random.random():
            offspring[i] = np.random.uniform(dom_l, dom_u)

    return


# non-uniform mutation (gaussian)
# adds a number drawn from a gaussian distribution
def non_uniform_mutation(offspring):
    # can add a number from a normal distribution to the existing weight
    for i in range(0, len(offspring)):
        if mutation_probability > random.random():
            offspring[i] = offspring[i] + np.random.normal()
    return



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    population = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fitness_population = evaluate(population)
    best = np.argmax(fitness_population)
    mean = np.mean(fitness_population)
    std = np.std(fitness_population)
    ini_g = 0
    solutions = [population, fitness_population]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    population = env.solutions[0]
    fitness_population = env.solutions[1]

    best = np.argmax(fitness_population)
    mean = np.mean(fitness_population)
    std = np.std(fitness_population)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fitness_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fitness_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()




for i in range(ini_g+1, gens):

    offspring_list = blend_crossover_alpha(population,fitness_population)  # crossover
    fitness_offspring_population = evaluate(offspring_list)   # evaluation

    population = survival(population, fitness_population, offspring_list,fitness_offspring_population,npop) #Survival Selection
    fitness_population = evaluate(population)

    best = np.argmax(fitness_population) #best solution in generation
    best_sol = fitness_population[best]

    mean = np.mean(fitness_population)
    std = np.std(fitness_population)

# saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fitness_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fitness_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',population[best])

    # saves simulation state
    solutions = [population, fitness_population]
    env.update_solutions(solutions)
    env.save_state()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state