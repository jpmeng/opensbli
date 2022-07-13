#!/usr/bin/env python
# Import all the functions from opensbli
from opensbli import *
import copy
from opensbli.utilities.helperfunctions import substitute_simulation_parameters
from opensbli.code_generation.opsc_compact import OPSCCompact
from opensbli.equation_types.opensbliequations import SimulationEquationsResidualCR,ConstituentRelationsGradient
from opensbli.utilities.helperfunctions import substitute_simulation_parameters
from opensbli.linear_solver.LinearSolver import TridiagonalSolver
from opensbli.schemes.spatial.compact import Compact
from opensbli.code_generation.algorithm import TraditionalAlgorithmRK
from opensbli.utilities.helperfunctions import Debug
import sys
# Problem dimension
ndim = 1
sc1 = "**{\'scheme\':\'Compact\'}"
# Define the wave equation in Einstein notation.
wave = "Eq(Der(phi,t), -c_j*Der(phi,x_j,scheme))"
wave = wave.replace("scheme",sc1)

equations = [wave]

# Substitutions
substitutions = []

# The wave speed
constants = ["c_j"]

# Coordinate direction symbol (x) this will be x_i, x_j, x_k
coordinate_symbol = "x"

simulation_eq = SimulationEquationsResidualCR()
eq = EinsteinEquation()
eqns = eq.expand(wave, ndim, coordinate_symbol, substitutions, constants)
simulation_eq.add_equations(eqns)

constituent = ConstituentRelationsGradient()

# Write the expanded equations to a Latex file with a given name and titile
latex = LatexWriter()
latex.open('equations.tex', "Einstein Expansion of the simulation equations")
latex.write_string("Simulation equations\n")
for index, eq in enumerate(flatten(simulation_eq.equations)):
    latex.write_expression(eq)

latex.write_string("Constituent relations\n")
for index, eq in enumerate(flatten(constituent.equations)):
    latex.write_expression(eq)

latex.close()

block = SimulationBlock(ndim, block_number=0)
trid = TridiagonalSolver(ndim, block.blockname, block.blocknumber)
block.sbli_rhs_discretisation = True

boundaries = []
# Create boundaries, one for each side per dimension
for direction in [0]:
    boundaries += [PeriodicBC(direction, 0)]
    boundaries += [PeriodicBC(direction, 1)]
local_dict = {"block": block, "GridVariable": GridVariable, "DataObject": DataObject}

block.set_block_boundaries(boundaries)

# Initial conditions
local_dict = {"block": block, "GridVariable": GridVariable, "DataObject": DataObject}
x0 = parse_expr("Eq(DataObject(x0), block.deltas[0]*block.grid_indexes[0])", local_dict=local_dict)
phi = parse_expr("Eq(DataObject(phi), sin(2.0*pi*DataObject(x0)))", local_dict=local_dict)
initial = GridBasedInitialisation()
initial.add_equations([x0, phi])

kwargs = {'iotype': "Write"}
output_arrays = simulation_eq.time_advance_arrays + [DataObject('x0')]
h5 = iohdf5(arrays=output_arrays, **kwargs)

simulation = copy.deepcopy(simulation_eq)
block.set_equations([initial, simulation])
block.setio([h5])

schemes = {}
if len(sys.argv)>1:
    compact = Compact(4, trid,ndim)
    schemes[compact.name] = compact
else:
    cent = Central(4)
    schemes[cent.name] = cent
rk = RungeKutta(3)
schemes[rk.name] = rk

block.set_discretisation_schemes(schemes)
# Discretise the equations on block
block.discretise()

# Algorithm for the block
alg = TraditionalAlgorithmRK(block)
SimulationDataType.set_datatype(Double)

if len(sys.argv)>1:
    OPSCCompact(alg, trid, compact)
else:
    OPSC(alg)

constants = ['c0', 'dt', 'niter', 'block0np0', 'Delta0block0']
values = ['0.5', '0.001', '1.0/0.001', '200', '1.0/block0np0']
substitute_simulation_parameters(constants, values)
