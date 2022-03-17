"""@brief Linear solver
   @author Jianping Meng
   @details This class will help users to setup the linear solvers, e.g., the options.
"""


class LinearSolver:
    def __init__(self, dim, block_name, block_number):
        self.name = "_at_"+block_name + str(block_number)
        self.block_name = block_name
        self.initialise_class = ""
        self.solver = ""
        self.space_dim = dim

    def __str__(self):
        return self.name

    def Initialise(self):
        return self.initialise_class + " *"+self.name + " = " + "new " + self.initialise_class+"("+self.block_name+");"

    def Finalise(self):
        return "delete " + self.name + ";"

    def Call(self, direction, result):
        return self.solver+"( "+str(self.space_dim)+","+str(direction)+", a, b, c, d, result, "+self.name+");"


class TridiagonalSolver(LinearSolver):
    def __init__(self, dim, block_name, block_number, options=()):
        super().__init__(dim, block_name, block_number)
        self.name = "Trid"+self.name
        self.initialise_class = "ops_tridsolver_params"
        self.solver = "ops_tridMultiDimBatch_Inc"
