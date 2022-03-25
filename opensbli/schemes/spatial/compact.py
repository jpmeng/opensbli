"""@brief
   @authors Jianping Meng
   @contributors
   @details
"""

from sympy import IndexedBase, Symbol, Rational, solve, interpolating_poly, integrate, Abs, Float, flatten, S
from opensbli.core.opensblifunctions import Function, BasicDiscretisation,DerPrint
from opensbli.core.opensbliobjects import ConstantObject
from opensbli.equation_types.opensbliequations import SimulationEquations, OpenSBLIEq, NonSimulationEquations
from opensbli.core.grid import GridVariable
from .scheme import Scheme
from sympy import horner, pprint
from opensbli.schemes.spatial.shock_capturing import ShockCapturing, LLFCharacteristic

from builtins import super

kernel_derivative_x_template="""
void PreprocessX4thCompact(const ACC<double> &u, ACC<double> &a, ACC<double> &b,
                 ACC<double> &c, ACC<double> &d, int *idx) {
  const int i{idx[0]};
  d(0, 0, 0) = u(1, 0, 0) - u(-1, 0, 0);
  if (i == 0) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * placeholderdx;
    c(0, 0, 0) = 0;

  } else if (i == (placeholdernumberofx - 1)) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * placeholderdx;
    c(0, 0, 0) = 0;

  } else {
    a(0, 0, 0) = left;
    b(0, 0, 0) = present;
    c(0, 0, 0) = right;
  }
}
"""

kernel_derivative_y_template="""void PreprocessY4thCompact(const ACC<double> &u, ACC<double> &a, ACC<double> &b,
                 ACC<double> &c, ACC<double> &d, int *idx) {
  const int j{idx[1]};
  d(0, 0, 0) = u(0, 1, 0) - u(0, -1, 0);
  if (j == 0) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * placeholderdy;
    c(0, 0, 0) = 0;
  } else if (j == (placeholdernumberofy - 1)) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * placeholderdy;
    c(0, 0, 0) = 0;
  } else {
    a(0, 0, 0) = left;
    b(0, 0, 0) = present;
    c(0, 0, 0) = right;
  }
}
"""

kernel_derivative_z_template="""void preprocessZ(const ACC<double> &u, ACC<double> &a, ACC<double> &b,
                 ACC<double> &c, ACC<double> &d, int *idx) {
  const int k{idx[2]};
  d(0, 0, 0) = u(0, 0, 1) - u(0, 0, -1);
  if (k == 0) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * h;
    c(0, 0, 0) = 0;

  } else if (k == (nz - 1)) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * h;
    c(0, 0, 0) = 0;

  } else {
    a(0, 0, 0) = left;
    b(0, 0, 0) = present;
    c(0, 0, 0) = right;
  }
}
"""

wrap_function_template_x="""
void CompactDifferenceX(ops_block& block, ops_dat& u, ops_dat& a, ops_dat& b,
                        ops_dat& c, ops_dat& d, ops_dat& ux,
                        ops_tridsolver_params* trid) {
  int* size{u->size};
  int* dm{u->d_m};
  int* dp{u->d_p};
  int spaceDim{block->dims};
  for (int i = 0; i < spaceDim; i++) {
    size[i] = size[i] - dp[i] + dm[i];
  };
  int iterRange[]{0, size[0], 0, size[1], 0, size[2]};
  ops_par_loop(preprocessX, "preprocessX", block, 3, iterRange,
               ops_arg_dat(u, 1, S3D_7PT, "double", OPS_READ),
               ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(d, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(ux, 1, S3D_000, "double", OPS_WRITE), ops_arg_idx());

  // Get the u_y, note that the solver will add result to ux so that uy must be
  // // zero before the call
  ops_tridMultiDimBatch_Inc(spaceDim, 0, size, a, b, c, d, ux, trid);
}
"""

wrap_function_template_y="""
void CompactDifferenceY(ops_block& block, ops_dat& u, ops_dat& a, ops_dat& b,
                        ops_dat& c, ops_dat& d, ops_dat& uy,
                        ops_tridsolver_params* trid) {
  int* size{u->size};
  int* dm{u->d_m};
  int* dp{u->d_p};
  int spaceDim{block->dims};
  for (int i = 0; i < spaceDim; i++) {
    size[i] = size[i] - dp[i] + dm[i];
  };
  int iterRange[]{0, size[0], 0, size[1], 0, size[2]};
  ops_par_loop(preprocessY, "preprocessY", block, 3, iterRange,
               ops_arg_dat(u, 1, S3D_7PT, "double", OPS_READ),
               ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(d, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(uy, 1, S3D_000, "double", OPS_WRITE), ops_arg_idx());

  // Get the u_y, note that the solver will add result to ux so that uy must be
  // // zero before the call
  ops_tridMultiDimBatch_Inc(spaceDim, 0, size, a, b, c, d, uy, trid);
}
"""

wrap_function_template_y="""
void CompactDifferenceY(ops_block& block, ops_dat& u, ops_dat& a, ops_dat& b,
                        ops_dat& c, ops_dat& d, ops_dat& uy,
                        ops_tridsolver_params* trid) {
  int* size{u->size};
  int* dm{u->d_m};
  int* dp{u->d_p};
  int spaceDim{block->dims};
  for (int i = 0; i < spaceDim; i++) {
    size[i] = size[i] - dp[i] + dm[i];
  };
  int iterRange[]{0, size[0], 0, size[1], 0, size[2]};
  ops_par_loop(preprocessY, "preprocessY", block, 3, iterRange,
               ops_arg_dat(u, 1, S3D_7PT, "double", OPS_READ),
               ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(d, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(uy, 1, S3D_000, "double", OPS_WRITE), ops_arg_idx());

  // Get the u_y, note that the solver will add result to ux so that uy must be
  // // zero before the call
  ops_tridMultiDimBatch_Inc(spaceDim, 0, size, a, b, c, d, uy, trid);
}
"""


class CompactDerivative(Function, BasicDiscretisation, DerPrint):
    """wrapper class to represent derivatives using the compact scheme
    Sympy already have a "Derivative" class, thus double D
    """
    def __new__(cls, expr, *args):
        args = tuple(flatten([expr] + list(args)))
        ret = super().__new__(cls, *args, evaluate=False)
        ret.store = True  # By default all the derivatives are stored
        ret.local_evaluation = True
        return ret

    def doit(cls):
        if aplaceholdernumberofy(arg == S.Zero for arg in cls.args):
            return S.Zero
        elif len(set(cls.args)) == 1:
            return S.One
        else:
            return cls

    def expand(self, **hints):
        from sympy.core.function import _coeff_isneg
        ders = self.args[1:]
        rets = 0
        arg = self.args[0]
        if arg.is_Add:
            aargs = list(arg.expand(deep=True).args)
            for i, ai in enumerate(aargs):
                if ai.is_Mul and _coeff_isneg(ai):
                    ai = ai*-S.One
                    rets -= self.func(ai, *ders)
                else:
                    rets += self.func(ai, *ders)
            return rets
        else:
            return self

    def _eval_expand_func(self, **hints):
        return self.expand()

    def _discretise_derivative(cls, scheme, block, boundary=True):
        """
        TODO V2 documentation
        This would return the discritized derivative of the
        local object depending on the order of accuracy specified
        Returns the formula for the derivative function, only first derivatives or homogeneous
        derivatives of higher order are supported. The mixed derivatives will be handled impl-
        citly while creating the kernels
        :arg derivative: the derivative on which discretisation should be performed
        :returns: the discritized derivative, in case of wall boundaries this is a Piecewise-
        function
        """
        order = cls.order
        form = 0
        # Put the coefficients of first and second derivatives in a dictionary and use them
        if cls.is_homogeneous:
            dire = cls.get_direction[0]
            weights = scheme._generate_weights(dire, order, block)
            for no, p in enumerate(scheme.points):
                expr = cls.args[0]
                for req in (cls.required_datasets):
                    loc = list(req.indices)
                    loc[dire] = loc[dire] + p
                    val = req.base[loc]
                    expr = expr.replace(req, val)
                form = form + weights[no]*expr
            if form == 0:
                raise ValueError("Central derivative formula is zero for %s" % cls)
        else:
            raise ValueError("The provided derivative is not homogeneous, %s" % cls)
        if boundary:
            form = cls.modify_boundary_formula(form, block)

        delta = S.One/block.deltas[dire]**order
        inv_delta = get_inverse_deltas(delta)
        form = form*(inv_delta)
        return form

    def modify_boundary_formula(cls, form, block):
        # Apply the boundary modifications
        modifications = block.check_modify_central()
        dire = cls.get_direction[0]
        if dire in modifications:
            boundary_mods = [k for k in modifications[dire] if k]
            expression_condition_pairs = []
            for b in boundary_mods:
                expression_condition_pairs += b.modification_scheme.expr_cond_pairs(cls.args[0], b.direction, b.side, cls.order, block)
            expression_condition_pairs += [ExprCondPair(form, True)]
            form = Piecewise(*expression_condition_pairs, **{'evaluate': False})
        return form

    @property
    def simple_name(cls):
        return "%s" % ("CD")

    def classical_strong_differentiabilty_transformation(cls, metric):
        direction = cls.get_direction
        if cls.order == 1:
            metric_der = metric.classical_strong_differentiabilty_transformation[direction[0]]
            transformed_der = metric_der.subs(metric.general_function, cls.args[0])
        elif cls.order == 2:
            metric.sd_used = True
            metric_der = metric.classical_strong_differentiabilty_transformation_sd[tuple(cls.get_direction)]
            transformed_der = metric_der.subs(metric.general_function, cls.args[0])
        return transformed_der


class Compact(Scheme):

    """ Spatial discretisation scheme using central differences. """

    def __init__(self, order):
        """ Set up the scheme.

        :arg int order: The order of accuracy of the scheme.
        """
        Scheme.__init__(self, "CentralDerivative", order)
        print("A Central scheme of order %d is being used." % order)
        self.schemetype = "Spatial"
        # Points for the spatial scheme
        self.points = list(i for i in range(int(-order/2), int(order/2+1)))
        self.required_constituent_relations = {}
        self.halotype = CentralHalos(order)
        return

    def _generate_weights(self, direction, order, block):
        """Finite difference weights for homogeneous derivatives of given order."""
        self.diffpoints = [i for i in self.points]
        weights = finite_diff_weights(order, self.diffpoints, 0)
        return weights[order][-1]

    # def add_required_database(self, dbases):
    #     # TODO V2: is it used??
    #     self.required_database += flatten(list(dbases))
    #     return

    # @property
    # def scheme_required_databases(self):
    #     # TODO V2: is it used??
    #     return set(self.required_database)

    # def update_works(self, to_descritse, block):
    #     # V2: Delete this?

    #     return

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
            local_kernels, discretised_eq = self.general_discretisation(type_of_eq.equations, block, name=type_of_eq.__class__.__name__)
            block.reset_work_to_stored  # Reset
            if discretised_eq:
                for ker in local_kernels:
                    eval_ker = local_kernels[ker]
                    type_of_eq.Kernels += [eval_ker]

                discretisation_kernel = Kernel(block, computation_name="%s evaluation" % type_of_eq.__class__.__name__)
                discretisation_kernel.set_grid_range(block)
                for eq in discretised_eq:
                    discretisation_kernel.add_equation(eq)
                discretisation_kernel.update_block_datasets(block)
                type_of_eq.Kernels += [discretisation_kernel]
                return self.required_constituent_relations
            else:
                pass
            return self.required_constituent_relations

    def get_local_function(self, list_of_components):
        centralderivatives_in_class = []
        for c in list_of_components:
            centralderivatives_in_class += list(c.atoms(CentralDerivative))
        centralderivatives_in_class = list(set(centralderivatives_in_class))
        return centralderivatives_in_class

    def group_by_direction(self, eqs):
        all_central_derivatives = []
        for eq in eqs:
            all_central_derivatives += list(eq.atoms(CentralDerivative))
        all_central_derivatives = list(set(all_central_derivatives))
        grouped = {}
        for cd in all_central_derivatives:
            direction = cd.get_direction[0]
            if direction in grouped.keys():
                grouped[direction] += [cd]
            else:
                grouped[direction] = [cd]
        return grouped

    def update_range_of_constituent_relations(self, central_derivative, block):
        direction = central_derivative.get_direction[0]

        if central_derivative.required_datasets:
            for v in central_derivative.required_datasets:
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
                    expr = OpenSBLIEq(v.work, v1._discretise_derivative(self, block))
                    ker = Kernel(block)
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
        # Discretise the viscous fluxes. This is straight forward as we need not modify aplaceholdernumberofything
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
        This discretises the central derivatives, without aplaceholdernumberofy special treatment of grouping them
        """
        discretized_equations = flatten(equations)[:]
        cds = self.get_local_function(flatten(equations))
        if cds:
            local_kernels = {}
            if block.store_derivatives:
                for der in cds:
                    der.update_work(block)
                    ker = Kernel(block)
                    if name:
                        ker.set_computation_name("%s %s " % (name, der))
                    local_kernels[der] = ker  # Reverted back
            # create a dictionary of works and kernels
            work_arry_subs = {}
            for der in cds:
                self.update_range_of_constituent_relations(der, block)
                expr, local_kernels = self.traverse(der, local_kernels, block)
                expr_discretised = OpenSBLIEq(der.work, factor(expr._discretise_derivative(self, block)))
                work_arry_subs[expr] = der.work
                local_kernels[der].add_equation(expr_discretised)
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
        # Zero out aplaceholdernumberofy other derivatives in the containing terms and other terms
        for no, eq in enumerate(other_terms):
            fns = [fn for fn in eq.atoms(Function) if not isinstance(fn, CentralDerivative)]
            substitutions = dict(zip(fns, [0]*len(fns)))
            other_terms[no] = other_terms[no].subs(substitutions)

        for no, eq in enumerate(containing_terms):
            fns = [fn for fn in eq.atoms(Function) if not isinstance(fn, CentralDerivative)]
            substitutions = dict(zip(fns, [0]*len(fns)))
            containing_terms[no] = containing_terms[no].subs(substitutions)
        return containing_terms, other_terms

    def traverse(self, central_deriv, kernel_dictionary, block):
        expr = central_deriv.copy()
        inner_cds = []
        pot = postorder_traversal(central_deriv)
        inner_cds = []
        for p in pot:
            if isinstance(p, CentralDerivative):
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
