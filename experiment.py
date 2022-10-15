# %%
import random
import operators
import sympy as sp
import numpy as np
import pandas as pd
import geppy as gep
from deap import tools, creator, base
from joblib import Parallel, delayed


# %%
adj_open = pd.read_parquet('data/raw_factor/adj_open.parquet')
adj_close = pd.read_parquet('data/raw_factor/adj_close.parquet')
adj_high = pd.read_parquet('data/raw_factor/adj_high.parquet')
adj_low = pd.read_parquet('data/raw_factor/adj_low.parquet')
volume = pd.read_parquet('data/raw_factor/volume.parquet')
label = pd.read_parquet('data/raw_factor/label.parquet')

# %%
pset = gep.PrimitiveSet("Main", input_names=['adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume'])
pset.add_function(operators.add, 2)
pset.add_function(operators.sub, 2)
pset.add_function(operators.mul, 2)
pset.add_function(operators.div, 2)
pset.add_function(operators.sqrt, 1)
pset.add_function(operators.ssqrt, 1)
pset.add_rnc_terminal()
# pset.add_ephemeral_terminal('const', lambda: random.uniform(-1, 1))

# %%
creator.create('FitnessMin', base.Fitness, weights=(1,))
creator.create('Individual', gep.Chromosome, fitness=creator.FitnessMin)

# %%
head_len = 10
n_genes = 1
rnc_len = 10

# %%
toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.uniform, a=-1, b=1)
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=head_len, rnc_gen=toolbox.rnc_gen, rnc_array_length=rnc_len)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gep.compile_, pset=pset)

# %%
def evaluate(individual):
    func = toolbox.compile(individual)
    pred = func(adj_open, adj_high, adj_low, adj_close, volume)
    if not isinstance(pred, pd.DataFrame):
        return np.nan,
    pred = pred.sub(pred.mean(axis=1), axis=0).div(pred.std(axis=1), axis=0)
    corr = pred.corrwith(label, axis=1, method='pearson')
    return np.abs(corr.mean()),

def async_map(func, jobs):
    return Parallel(n_jobs=-1, backend='threading')(delayed(func)(job) for job in jobs)
toolbox.register('map', async_map)

toolbox.register('evaluate', evaluate)

# %%
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1

# %%
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.nanmean)
stats.register("std", np.nanstd)
stats.register("min", np.nanmin)
stats.register("max", np.nanmax)
stats.register("count", lambda x: len(x) - np.isnan(x).sum())

# %%
n_pop = 200
n_gen = 50
champs = 10
pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(champs)

# %%
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1, stats=stats, hall_of_fame=hof, verbose=True)

for elite in hof:
    print(gep.simplify(elite, symbolic_function_map={
        "add": lambda x, y: sp.Symbol(f'{x} + {y}'),
        "sub": lambda x, y: sp.Symbol(f'{x} - {y}'),
        "mul": lambda x, y: sp.Symbol(f'{x} * {y}'),
        "div": lambda x, y: sp.Symbol(f'{x} / {y}'),
        "sqrt": lambda x: sp.Symbol(f'sqrt({x})'),
        "ssqrt": lambda x: sp.Symbol(f'ssqrt({x})'),
    }))
