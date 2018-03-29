#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

# CC214530 Dominique Kidd Applied AI Coursework Task 5
# Code herein largely taken from FaceRec.py on NOW and from 
# http://deap.readthedocs.io/en/master/examples/gp_symbreg.html

import operator
import math
import random
import numpy
from sklearn.datasets import fetch_lfw_people
import logging
import deap
from sklearn.cross_validation import train_test_split
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from time import time
from sklearn.decomposition import RandomizedPCA

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape
print ("number of samples")
print(n_samples)
  
  # for machine learning we use the 2 data directly (as relative pixel
  # positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]
  
  # the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
  
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
  
  
  ###############################################################################
  # Split into a training set and a test set using a stratified k fold
  
  # split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.25)
  
  
  ###############################################################################
  # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
  # dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150
  
print("Extracting the top %d eigenfaces from %d faces"
        % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
  
eigenfaces = pca.components_.reshape((n_components, h, w))
  
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))
print (pca.components_.shape)

# Define new functions
# remember - protecting from div by zero error / crash
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
pset = gp.PrimitiveSet("MAIN", 1)
# Dee the below is what we want but cannot run this as we are then passing
# the wrong number of Args all the way down the code which obviously causes error. 
#pset = gp.PrimitiveSet("MAIN", 150)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

#Dee the string name of the Ephem. Const has to be manually changed between runs
# else is causes an error. Must be a better way?
pset.addEphemeralConstant("rand0111", lambda: random.randint(-1,1))
#pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # for minimise
#creator.create("FitnessMin", base.Fitness, weights=(1.0,)) # for maximise
#creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    
    # Dee here we are working on just x but really we need to be working on
    # all 150 of our inputs? Would this be in the form of a loop? Or passing
    # them as an array - passing the pca.components? Not sure how to code this
    # maybe for item in pca do this? Or more accurately for ARGS in PrimitiveSet
    # do this? So we evaluate the fitness function of each input individually??
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),

#Dee here is where we need to pass in our 150 args, not just the 1 arg renamed as
# x in the exampleas in example
toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    
    
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof



if __name__ == "__main__":
    main()