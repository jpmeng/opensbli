"""@brief
   @authors Jianping Meng
   @contributors
   @details
"""

from sympy import IndexedBase, Symbol, Rational, solve, interpolating_poly, integrate, Abs, Float, flatten, S, postorder_traversal, Function, flatten, S, factor
from sympy.core import Add, Mul
from opensbli.code_generation.opsc import WriteString
from opensbli.core.opensblifunctions import Function, BasicDiscretisation, DerPrint,CompactDerivative
from opensbli.core.opensbliobjects import ConstantObject, DataSet, CoordinateObject
from opensbli.equation_types.opensbliequations import SimulationEquations, OpenSBLIEq, NonSimulationEquations
from opensbli.core.grid import GridVariable
from .scheme import Scheme
from sympy import horner, pprint
from opensbli.schemes.spatial.shock_capturing import ShockCapturing, LLFCharacteristic

from opensbli.core.kernel import Kernel, ImplicitKernel

from builtins import super

from opensbli.utilities.helperfunctions import Debug

## TODO we need to write the template code for the second-order derivative

## TODO for multi-block application we need to also modify block name


stencil_template="""
#ifdef OPS_3D
  int local[]{0, 0, 0};
  ops_stencil local_stencil{ops_decl_stencil(3, 1, local, "000")};

  int neighbor[]{0, 0, 0, -1, 0, 0, 1,  0, 0, 0, -1,
                0, 0, 1, 0,  0, 0, -1, 0, 0, 1};
  ops_stencil neighbor_stencil{ops_decl_stencil(3, 7, neighbor, "3d7Point")};
#endif
#ifdef OPS_2D
      int local[]{0, 0};
    ops_stencil local_stencil{ops_decl_stencil(2, 1, local, "00")};
    // declare stencils for the central differencing
    int neighbor[]{0, 0, 1, 0, -1, 0, 0, 1, 0, -1};
    ops_stencil neighbor_stencil{ops_decl_stencil(2, 5, neighbor, "2d5Point")};
#endif
#ifdef OPS_1D
      int local[]{0};
    ops_stencil local_stencil{ops_decl_stencil(1, 1, local, "0")};
    // declare stencils for the central differencing
    int neighbor[]{0, 1, -1};
    ops_stencil neighbor_stencil{ops_decl_stencil(2, 3, neighbor, "1d3Point")};
#endif
"""

varible_template="""ops_dat name_place;
{
#ifdef OPS_3D
    int halo_p[] = {halo_place, halo_place, halo_place};
    int halo_m[] = {-halo_place, -halo_place, -halo_place};
    int size[] = {block0np0, block0np1, block0np2};
    int base[] = {0, 0, 0};
#endif

#ifdef OPS_2D
    int halo_p[] = {halo_place, halo_place};
    int halo_m[] = {-halo_place, -halo_place};
    int size[] = {block0np0, block0np1};
    int base[] = {0, 0};
#endif

#ifdef OPS_1D
    int halo_p[] = {halo_place};
    int halo_m[] = {-halo_place};
    int size[] = {block0np0};
    int base[] = {0};
#endif
    double* value = NULL;
    name_place = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "name_place");
}
"""

kernel_derivative_x_template = """
void PreprocessX4thCompact1st(const ACC<double> &u, ACC<double> &a,
                              ACC<double> &b, ACC<double> &c, ACC<double> &d,
                              ACC<double> &ux, int *idx, int *nx, int *layer,
                              double *dx) {
  const int i{idx[0]};
#ifdef OPS_3D
  d(0, 0, 0) = u(1, 0, 0) - u(-1, 0, 0);
  ux(0, 0, 0) = 0;
  const int start{-(*layer)};
  const int end{(*nx) + (*layer) - 1};
  if (i == start) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * (*dx);
    c(0, 0, 0) = 0;

  } else if (i == end) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * (*dx);
    c(0, 0, 0) = 0;

  } else {
    a(0, 0, 0) = (*dx) / 3;
    b(0, 0, 0) = 4 * (*dx) / 3;
    c(0, 0, 0) = (*dx) / 3;
  }
#endif  // OPS_3D
#ifdef OPS_2D
  d(0, 0) = u(1, 0) - u(-1, 0);
  ux(0, 0) = 0;
  const int start{-(*layer)};
  const int end{(*nx) + (*layer) - 1};
  if (i == start) {
    a(0, 0) = 0;
    b(0, 0) = 2 * (*dx);
    c(0, 0) = 0;

  } else if (i == end) {
    a(0, 0) = 0;
    b(0, 0) = 2 * (*dx);
    c(0, 0) = 0;

  } else {
    a(0, 0) = (*dx) / 3;
    b(0, 0) = 4 * (*dx) / 3;
    c(0, 0) = (*dx) / 3;
  }
#endif  // OPS_2D

#ifdef OPS_1D
  d(0) = u(1) - u(-1);
  ux(0) = 0;
  const int start{-(*layer)};
  const int end{(*nx) + (*layer) - 1};
  if (i == start) {
    a(0) = 0;
    b(0) = 2 * (*dx);
    c(0) = 0;

  } else if (i == end) {
    a(0) = 0;
    b(0) = 2 * (*dx);
    c(0) = 0;

  } else {
    a(0) = (*dx) / 3;
    b(0) = 4 * (*dx) / 3;
    c(0) = (*dx) / 3;
  }
#endif  // OPS_1D
}
"""

kernel_derivative_y_template = """
void PreprocessY4thCompact1st(const ACC<double> &u, ACC<double> &a,
                              ACC<double> &b, ACC<double> &c, ACC<double> &d,
                              ACC<double> &uy, int *idx, int *ny, int *layer,
                              double *dy) {
 const int j{idx[1]};
#ifdef OPS_3D
  d(0, 0, 0) = u(0, 1, 0) - u(0, -1, 0);
  uy(0, 0, 0) = 0;
  const int start{-(*layer)};
  const int end{(*ny) + (*layer) - 1};
  if (j == start) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * (*dy);
    c(0, 0, 0) = 0;
  } else if (j == end) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * (*dy);
    c(0, 0, 0) = 0;
  } else {
    a(0, 0, 0) = (*dy) / 3;
    b(0, 0, 0) = 4 * (*dy) / 3;
    c(0, 0, 0) = (*dy) / 3;
  }
#endif  // OPS_3D

#ifdef OPS_2D
  const int j{idx[1]};
  d(0, 0) = u(0, 1) - u(0, -1);
  uy(0, 0) = 0;
  const int start{-(*layer)};
  const int end{(*ny) + (*layer) - 1};
  if (j == start) {
    a(0, 0) = 0;
    b(0, 0) = 2 * (*dy);
    c(0, 0) = 0;
  } else if (j == end) {
    a(0, 0) = 0;
    b(0, 0) = 2 * (*dy);
    c(0, 0) = 0;
  } else {
    a(0, 0) = (*dy) / 3;
    b(0, 0) = 4 * (*dy) / 3;
    c(0, 0) = (*dy) / 3;
  }
#endif  // OPS_2D
}
"""

kernel_derivative_z_template = """
#ifdef OPS_3D

void PreprocessZ4thCompact1st(const ACC<double> &u, ACC<double> &a,
                              ACC<double> &b, ACC<double> &c, ACC<double> &d,
                              ACC<double> &uz, int *idx, int *nz, int *layer,
                              double *dz) {
#ifdef OPS_3D
  const int k{idx[2]};
  d(0, 0, 0) = u(0, 0, 1) - u(0, 0, -1);
  uz(0, 0, 0) = 0;
  const int start{-(*layer)};
  const int end{(*nz) + (*layer) - 1};
  if (k == start) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * (*dz);
    c(0, 0, 0) = 0;

  } else if (k == end) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * (*dz);
    c(0, 0, 0) = 0;

  } else {
    a(0, 0, 0) = (*dz) / 3;
    b(0, 0, 0) = 4 * (*dz) / 3;
    c(0, 0, 0) = (*dz) / 3;
  }
#endif  // OPS_3D Z
}
#endif // OPS_3D Z

"""
kernel_templates = [kernel_derivative_x_template, kernel_derivative_y_template, kernel_derivative_z_template]

wrap_function_template_x_4th_1st = """
void CompactDifference4thX1st(ops_block& block, ops_dat& u, ops_dat& a, ops_dat& b,ops_dat& c, ops_dat& d, ops_dat& ux,ops_tridsolver_params* trid, double delta, int layer=0) {
  int* size{u->size};
  int* dm{u->d_m};
  int* dp{u->d_p};
  int spaceDim{block->dims};
  int* tridSize{new int[spaceDim]};
  for (int i = 0; i < spaceDim; i++) {
    tridSize[i] = size[i] - dp[i] + dm[i];
  };
  int* iterRange {new int[2*spaceDim]};
  for (int i = 0; i < spaceDim; i++) {
    iterRange[2*i] = -layer;
    iterRange[2*i+1] = size[0] + layer;
  };
  ops_par_loop(PreprocessX4thCompact1st, "preprocessX", block, SPACEDIM, iterRange,
               ops_arg_dat(u, 1, neighbor_stencil, "double", OPS_READ),
               ops_arg_dat(a, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(b, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(c, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(d, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(ux, 1, local_stencil, "double", OPS_WRITE), ops_arg_idx(),
               ops_arg_gbl(&size[0], 1, "int", OPS_READ),
               ops_arg_gbl(&layer, 1, "int", OPS_READ),
               ops_arg_gbl(&delta, 1, "double", OPS_READ));
  ops_tridMultiDimBatch_Inc(spaceDim, 0, tridSize, a, b, c, d, ux, trid);
  delete[] iterRange;
  delete[] tridSize;
}

"""

wrap_function_template_y_4th_1st = """
void CompactDifference4thY1st(ops_block& block, ops_dat& u, ops_dat& a, ops_dat& b,ops_dat& c, ops_dat& d, ops_dat& uy, ops_tridsolver_params* trid, double delta, int layer=0) {
  int* size{u->size};
  int* dm{u->d_m};
  int* dp{u->d_p};
  int spaceDim{block->dims};
  int* tridSize{new int[spaceDim]};
  for (int i = 0; i < spaceDim; i++) {
    tridSize[i] = size[i] - dp[i] + dm[i];
  };
  int* iterRange {new int[2*spaceDim]};
  for (int i = 0; i < spaceDim; i++) {
    iterRange[2*i] = -layer;
    iterRange[2*i+1] = size[0] + layer;
  };
  ops_par_loop(PreprocessY4thCompact1st, "preprocessY", block, SPACEDIM, iterRange,
               ops_arg_dat(u, 1, neighbor_stencil, "double", OPS_READ),
               ops_arg_dat(a, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(b, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(c, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(d, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(uy, 1, local_stencil, "double", OPS_WRITE), ops_arg_idx(),
               ops_arg_gbl(&size[1], 1, "int", OPS_READ),
               ops_arg_gbl(&layer, 1, "int", OPS_READ),
               ops_arg_gbl(&delta, 1, "double", OPS_READ)
               );
  ops_tridMultiDimBatch_Inc(spaceDim, 0, tridSize, a, b, c, d, uy, trid);
  delete[] iterRange;
  delete[] tridSize;
}
"""

wrap_function_template_z_4th_1st = """
void CompactDifference4thZ1st(ops_block& block, ops_dat& u, ops_dat& a, ops_dat& b,ops_dat& c, ops_dat& d, ops_dat& uz,ops_tridsolver_params* trid, double delta, int layer=0) {
  int* size{u->size};
  int* dm{u->d_m};
  int* dp{u->d_p};
  int spaceDim{block->dims};
  int* tridSize{new int[spaceDim]};
  for (int i = 0; i < spaceDim; i++) {
    tridSize[i] = size[i] - dp[i] + dm[i];
  };
int* iterRange {new int[2*spaceDim]};
  for (int i = 0; i < spaceDim; i++) {
    iterRange[2*i] = -layer;
    iterRange[2*i+1] = size[0] + layer;
  };
  ops_par_loop(PreprocessZ4thCompact1st, "preprocessZ", block, SPACEDIM, iterRange,
               ops_arg_dat(u, 1, neighbor_stencil, "double", OPS_READ),
               ops_arg_dat(a, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(b, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(c, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(d, 1, local_stencil, "double", OPS_WRITE),
               ops_arg_dat(uz, 1, local_stencil, "double", OPS_WRITE), ops_arg_idx(), ops_arg_gbl(&size[2], 1, "int", OPS_READ),
               ops_arg_gbl(&layer, 1, "int", OPS_READ),
               ops_arg_gbl(&delta, 1, "double", OPS_READ));
  ops_tridMultiDimBatch_Inc(spaceDim, 0, tridSize, a, b, c, d, uz, trid);
  delete[] iterRange;
  delete[] tridSize;
}

"""




class CompactHalos(object):
    """Number of halo points required for the compact scheme. Assumes they are the same in all dimensions.
  """

    def __init__(self, order=4):
        # we assume that compact schemes always requires one halo by default
        if order==4:
            self.halos = [-1, 1]
        if order==6:
            self.halos = [-2, 2]

    def get_halos(self, side):
        return self.halos[side]

    def __str__(self):
        return "CompactHalos"

class CustomHalos(object):
    """Number of halo points required for the compact scheme. Assumes they are the same in all dimensions.
  """

    def __init__(self, layer):        # we assume that compact schemes always requires one halo by default

        self.halos = [-layer, layer]

    def get_halos(self, side):
        return self.halos[side]

    def __str__(self):
        return "CustomHalos"




#TODO There are in fact a family of compact schemes such as we many need
# to overide the init method
class Compact(Scheme):

    """ Spatial discretisation scheme using compact differences.
    During the construction process, users can choose the proper linear solver by the linear_solver argument
    """

    def __init__(self, order, linear_solver,space_dim=3):
        """ Set up the scheme.

        :arg int order: The order of accuracy of the scheme.
        """
        super().__init__("CompactScheme", order)
        print("A compact scheme of order %d is being used." % order)
        self.haloNum = 1
        self.schemetype = "Spatial"
        # Points for the spatial scheme
        self.required_constituent_relations = {}
        self.halotype = CompactHalos(order)
        self.points = list(i for i in range(-1, 2))
        #self.preprocess_functions =[]
        self.kernel_file_name = "compactscheme_kernel.h"
        kernel=open(self.kernel_file_name, "w")
        kernel.write("#ifndef COMPACT_KERNEL_H\n")
        kernel.write("#define COMPACT_KERNEL_H\n")
        for kernelString in kernel_templates:
            kernel.write(kernelString)
            #self.preprocess_functions.append(kernelString)
        kernel.write("#endif\n")
        kernel.close()
        aString = varible_template.replace("name_place","a");
        aString = aString.replace("halo_place",str(5))
        bString = varible_template.replace("name_place","b");
        bString = bString.replace("halo_place",str(5))
        cString = varible_template.replace("name_place","c");
        cString = cString.replace("halo_place",str(5))
        dString = varible_template.replace("name_place","d");
        dString = dString.replace("halo_place",str(5))
        self.data_def = [aString,bString,cString,dString]
        self.wrap_templates_1st = [wrap_function_template_x_4th_1st.replace("SPACEDIM",str(space_dim)), wrap_function_template_y_4th_1st.replace("SPACEDIM",str(space_dim)), wrap_function_template_z_4th_1st.replace("SPACEDIM",str(space_dim))]
        self.wrap_function_1st = {0:"CompactDifference4thX1st(block_name, var,a,b,c, d, der,trid,delta,layer);",1:"CompactDifference4thY1st(block_name, var,a,b,c, d, der,trid,delta,layer);",2:"CompactDifference4thZ1st(block_name, var,a,b,c, d, der,trid,delta,layer);"}
        wrap_temp =  self.wrap_function_1st
        for key, fun_def in wrap_temp.items():
            fun_def = fun_def.replace("trid",linear_solver.name)
            self.wrap_function_1st[key] = fun_def
        self.stencils = stencil_template

        return

    def set_halos(self, block):
        """Sets the halos of the scheme to the block, Max of the halos of the block are used for setting the range of
        initialisation"""
        for direction in range(block.ndim):
            block.set_block_boundary_halos(direction, 0, self.halotype)
            block.set_block_boundary_halos(direction, 1, self.halotype)
        return

    def discretise(self, type_of_eq, block):
        """Discretisation application."""
        self.set_halos(block)
        if isinstance(type_of_eq, SimulationEquations):
            """ Simulation equations are always solved as sbli_rhs_discretisation as of now
            # TODO V2 change the name"""
            self.sbli_rhs_discretisation(type_of_eq, block)
            return self.required_constituent_relations
        else:
            block.store_work_index  # Store work
            local_kernels, discretised_eq = self.general_discretisation       (type_of_eq.equations, block, name=type_of_eq.__class__.__name__)
            block.reset_work_to_stored # Reset
            ## if there are gradients in the CR laws
            if discretised_eq:
                for ker in local_kernels:
                    eval_ker = local_kernels[ker]
                    type_of_eq.Kernels += [eval_ker]
                # Might be CR
                discretisation_kernel = Kernel(block, computation_name="%s evaluation work variable for gradients" % type_of_eq.__class__.__name__)
                discretisation_kernel.set_grid_range(block)
                for dir in range(0,block.ndim):
                    discretisation_kernel.set_halo_range(dir,0,self.halotype)
                    discretisation_kernel.set_halo_range(dir,1,self.halotype)
                for eq in discretised_eq:
                    discretisation_kernel.add_equation(eq)
                discretisation_kernel.update_block_datasets(block)
                type_of_eq.Kernels += [discretisation_kernel]
                return self.required_constituent_relations
            else:
                pass
            return self.required_constituent_relations


    def get_local_function(self, list_of_components):
        CompactDerivatives_in_class = []
        for c in list_of_components:
            CompactDerivatives_in_class += list(c.atoms(CompactDerivative))
        CompactDerivatives_in_class = list(set(CompactDerivatives_in_class))
        return CompactDerivatives_in_class

    def group_by_direction(self, eqs):
        all_compact_derivatives = []
        for eq in eqs:
            all_compact_derivatives += list(eq.atoms(CompactDerivative))
        all_compact_derivatives = list(set(all_compact_derivatives))
        grouped = {}
        for cd in all_compact_derivatives:
            direction = cd.get_direction[0]
            if direction in grouped.keys():
                grouped[direction] += [cd]
            else:
                grouped[direction] = [cd]
        return grouped

    ## TODO constituent_relations should not have derivative, using kernel
    def update_range_of_constituent_relations(self, compact_derivative, block):
        direction = compact_derivative.get_direction[0]
        if compact_derivative.required_datasets:
            for v in compact_derivative.required_datasets:
                if v in self.required_constituent_relations.keys():
                    self.required_constituent_relations[v].set_halo_range(direction, 0, self.halotype)
                    self.required_constituent_relations[v].set_halo_range(direction, 1, self.halotype)
                else:
                    self.required_constituent_relations[v] = Kernel(block, computation_name="CR%s" % v)
                    self.required_constituent_relations[v].set_grid_range(block)
                    self.required_constituent_relations[v].set_halo_range(direction, 0, self.halotype)
                    self.required_constituent_relations[v].set_halo_range(direction, 1, self.halotype)
        return

    def check_constituent_relations(self, block, list_of_eq):
        """ Checks all the datasets in equations provided are evaluated in constituent relations."""
        arrays = []
        for eq in flatten(list_of_eq):
            arrays += list(eq.atoms(DataSet))
        arrays = set(arrays)
        undefined = arrays.difference(self.required_constituent_relations.keys())

        for dset in undefined:
            self.required_constituent_relations[dset] = Kernel(block, computation_name="CR%s" % dset)
            self.required_constituent_relations[dset].set_grid_range(block)
        return

    def sbli_rhs_discretisation(self, type_of_eq, block):
        """
        This is the discretisation for the compressible Navier-Stokes equations by classifying them based on Reynolds number
        # TODO get the parameters dynamically from the problem script
        """
        equations = flatten(type_of_eq.equations)
        residual_arrays = [eq.residual for eq in equations]
        equations = [e._sanitise_equation for e in equations]
        classify_parameter = ConstantObject("Re")
        self.required_constituent_relations = {}
        viscous, convective = self.classify_equations_on_parameter(equations, classify_parameter)
        kernels = []
        convective_grouped = self.group_by_direction(convective)
        if convective_grouped:
            # Create equations for evaluation of derivatives
            for key, value in convective_grouped.items():
                for v in value:
                    v.update_work(block)
            local_evaluations_group = {}
            function_expressions_group = {}
            # Process the convective derivatives, this requires grouping of equations
            subs_conv = {}
            for key, value in convective_grouped.items():
                local_evaluations_group[key] = []
                ev_ker = Kernel(block)
                ev_ker.set_computation_name("Convective terms group %d" % key)
                ev_ker.set_grid_range(block)
                block.store_work_index
                local = []
                for v in value:
                    self.update_range_of_constituent_relations(v, block)
                    if len(v.required_datasets) > 1:
                        wk = block.work_array()
                        block.increase_work_index
                        expr = OpenSBLIEq(wk, v.args[0])
                        ev_ker.add_equation(expr)
                        ev_ker.set_halo_range(key, 0, self.halotype)
                        ev_ker.set_halo_range(key, 1, self.halotype)
                        v1 = v.subs(v.args[0], wk)
                    else:
                        v1 = v
                    #here we just need an equation of work and the terms to be discretised
                    expr = OpenSBLIEq(v.work, v1)
                    ker = ImplicitKernel(self,block)
                    ker.add_equation(expr)
                    ker.set_computation_name("Convective %s " % (v))
                    ker.set_grid_range(block)
                    local += [ker]
                    subs_conv[v] = v.work
                if ev_ker.equations:
                    local_evaluations_group[key] += [ev_ker]
                function_expressions_group[key] = local
                block.reset_work_to_stored
            # Convective evaluation
            for key, value in local_evaluations_group.items():
                kernels += value + function_expressions_group[key]
            # Create convective residual

            convective_discretized = convective[:]
            self.check_constituent_relations(block, convective_discretized)

            for no, c in enumerate(convective_discretized):
                convective_discretized[no] = convective_discretized[no].subs(subs_conv)
            residual_type = 'Convective'
            conv_residual_kernel = self.create_residual_kernel(residual_arrays, convective_discretized, block, residual_type)
            conv_residual_kernel.set_computation_name("Convective residual ")
            kernels += [conv_residual_kernel]
        # reset the work index of blocks
        block.reset_work_index
        # Discretise the viscous fluxes. This is straight forward as we need not modify any thing
        viscous_kernels, viscous_discretised = self.general_discretisation(viscous, block, name="Viscous")
        self.check_constituent_relations(block, viscous)
        if viscous_kernels:
            # Sort to have all first derivatives first, and then second derivatives after
            for ker in sorted(viscous_kernels, key=lambda x: len(x.atoms(CoordinateObject))):
                eval_ker = viscous_kernels[ker]
                kernels += [eval_ker]
        if viscous_discretised:
            residual_type = 'Viscous'
            visc_residual_kernel = self.create_residual_kernel(residual_arrays, viscous_discretised, block, residual_type)
            visc_residual_kernel.set_computation_name("Viscous residual")
            kernels += [visc_residual_kernel]
        # Add the kernels to the solutions
        type_of_eq.Kernels += kernels
        block.reset_work_index
        return

    def set_halo_range_kernel(self, kernel, direction, sides=None):
        """Sets the halo range for a kernel if the kernel needs to be evaluated into the halo points.
        For example, to evaluate ($\partial u/ \partial x0 \partial x1$) the inner derivative ($\partial u/ \partial x0$)
        should be evaluated into the halo points"""
        if not sides:
            kernel.set_halo_range(direction, 0, self.halotype)
            kernel.set_halo_range(direction, 1, self.halotype)
            return kernel
        else:
            raise NotImplementedError("")

    def create_residual_kernel(self, residual_arrays, discretised_eq, block, residual_type):
        """Creates a kernel for evaluating the residual of the discretised equations"""
        if len(residual_arrays) != len(discretised_eq):
            raise ValueError("")
        residue_kernel = Kernel(block)
        for no, array in enumerate(residual_arrays):
            # First time writing to the residual arrays
            if residual_type is 'Convective':
                expr = OpenSBLIEq(array, discretised_eq[no])
            else:
                expr = OpenSBLIEq(array, array+discretised_eq[no])
            residue_kernel.add_equation(expr)
        residue_kernel.set_grid_range(block)
        return residue_kernel

    def general_discretisation(self, equations, block, name=None):
        """
        This discretises the central derivatives, without a special treatment of grouping them
        """

        discretized_equations = flatten(equations)[:]
        cts = self.get_local_function(flatten(equations))
        extraLayer = False
        if name == 'ConstituentRelationsGradient':
            extraLayer = True
        if cts:
            local_kernels = {}
            if block.store_derivatives:
                for der in cts:
                    der.update_work(block)
                    layer = 0
                    if extraLayer:
                        ## hard code for compact scheme
                        layer = 2;
                    ker = ImplicitKernel(self,block,layer)
                    # for dir in arange(0,block.ndim):
                    #     ker.set_halo_range(dir,0,(2))
                    #     ker.set_halo_range(dir,1,(2))
                    if name:
                        ker.set_computation_name("%s %s " % (name, der))
                    local_kernels[der] = ker  # Reverted back
            # create a dictionary of works and kernels
            work_arry_subs = {}
            for der in cts:
                self.update_range_of_constituent_relations(der, block)
                expr, local_kernels = self.traverse(der, local_kernels, block)
                work_equation = OpenSBLIEq(der.work, expr)
                work_arry_subs[expr] = der.work
                local_kernels[der].add_equation(work_equation)
                local_kernels[der].set_grid_range(block)
            for no, c in enumerate(discretized_equations):
                discretized_equations[no] = discretized_equations[no].subs(work_arry_subs)
            return local_kernels, discretized_equations
        else:
            return None, None

    def classify_equations_on_parameter(self, equations, parameter):
        """Classifies the given equations into to different groups of equations.
        a. the terms of the equations that contains the parameter and
        b. the terms that donot contain the given parameter."""
        containing_terms = [S.Zero for eq in equations]
        other_terms = [S.Zero for eq in equations]
        for number, eq in enumerate(equations):
            if isinstance(eq.rhs, Add):
                for expr in eq.rhs.args:
                    if expr.has(parameter):
                        containing_terms[number] = Add(containing_terms[number], expr)
                    else:
                        other_terms[number] = Add(other_terms[number], expr)
            elif isinstance(eq.rhs, Mul):
                expr = eq.rhs
                if expr.has(parameter):
                    containing_terms[number] = Add(containing_terms[number], expr)
                else:
                    other_terms[number] = Add(other_terms[number], expr)
        # Zero out other derivatives in the containing terms and other terms
        for no, eq in enumerate(other_terms):
            fns = [fn for fn in eq.atoms(Function) if not isinstance(fn, CompactDerivative)]
            substitutions = dict(zip(fns, [0]*len(fns)))
            other_terms[no] = other_terms[no].subs(substitutions)

        for no, eq in enumerate(containing_terms):
            fns = [fn for fn in eq.atoms(Function) if not isinstance(fn, CompactDerivative)]
            substitutions = dict(zip(fns, [0]*len(fns)))
            containing_terms[no] = containing_terms[no].subs(substitutions)
        return containing_terms, other_terms

    def traverse(self, central_deriv, kernel_dictionary, block):
        expr = central_deriv.copy()
        inner_cds = []
        pot = postorder_traversal(central_deriv)
        inner_cds = []
        for p in pot:
            if isinstance(p, CompactDerivative):
                inner_cds += [p]
            else:
                continue
        # Check if derivatives exists with inner derivatives, i.e mixed higher order derivatives
        if len(inner_cds) > 1:
            for np, cd in enumerate(inner_cds[:-1]):
                if cd.is_store and cd.work:
                    cd.is_used(True)
                    expr = expr.subs(cd, cd.work)
                    # update the kernel ranges for inner central derivatives
                    dires = inner_cds[np+1].get_direction
                    for direction in dires:
                        kernel_dictionary[cd].set_halo_range(direction, 0, self.halotype)
                        kernel_dictionary[cd].set_halo_range(direction, 1, self.halotype)
                elif cd.is_store:
                    raise ValueError("NOT IMPLEMENTED THIS")
                elif not cd.is_store:
                    raise ValueError("This dependency should be validated for Carpenter BC")
                else:
                    raise ValueError("Could not classify this")
        return expr, kernel_dictionary

