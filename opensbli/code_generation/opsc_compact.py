"""@brief Algorithm generation for using the compact scheme
   @author Jianping Meng
"""


from sympy.core.compatibility import is_sequence
from sympy.printing.ccode import C99CodePrinter
from sympy.core.relational import Equality
from opensbli.core.opensbliobjects import ConstantObject, ConstantIndexed, Constant, DataSetBase, GroupedPiecewise
from sympy import Symbol, flatten
from opensbli.core.grid import GridVariable
from opensbli.core.datatypes import SimulationDataType
from sympy import Pow, Idx, pprint, count_ops
from opensbli.code_generation.opsc import OPSC, WriteString, ACCCodePrinter, OPSAccess, ccodeAcc, indent_code
from opensbli.linear_solver.LinearSolver import LinearSolver
from opensbli.utilities.helperfunctions import Debug
import os
import logging
LOG = logging.getLogger(__name__)
BUILD_DIR = os.getcwd()


class OPSCCompact(OPSC):
    """ Generating an OPSC code from the algorithm class.
        :arg object algorithm: An OpenSBLI algorithm class.
        :arg bool operation_count: If True, prints the number of arithmetic operations per kernel.
        :arg int OPS_diagnostics: OPS performance diagnostics. The default of 1 provides no kernel-based timing output
        A value of 5 gives a kernel breakdown of computational kernel and MPI exchange time."""

    # This revision is for the new OPS ACC template call
    ops_headers_acc = {'input': "const %s &%s", 'output': '%s &%s', 'inout': '%s &%s'}

    ops_headers_point = {'input': "const %s *%s", 'output': '%s *%s', 'inout': '%s *%s'}

    # Revised
    def __init__(self, algorithm, LinearSolver, ImplicitScheme,operation_count=False, OPS_diagnostics=1):
        # if not algorithm.MultiBlock:
        self.operation_count = operation_count
        self.OPS_diagnostics = OPS_diagnostics
        self.MultiBlock = False
        self.scheme = ImplicitScheme
        self.linear_solver = LinearSolver
        self.dtype = algorithm.dtype
        # Check if the simulation monitoring should be written to an output log file
        if algorithm.simulation_monitor:
            if algorithm.simulation_monitor.output_file:
                self.monitoring_output_file = True
        else:
            self.monitoring_output_file = False
        # First write the kernels, with this we will have the Rational constants to declare
        self.write_kernels(algorithm)
        def_decs = self.opsc_def_decs(algorithm)
        end = self.ops_exit()
        algorithm.prg.components = def_decs + algorithm.prg.components + end
        code = algorithm.prg.opsc_code
        code = self.before_main(algorithm) + code
        f = open('opensbli.cpp', 'w')
        f.write('\n'.join(code))
        f.close()
        print("Successfully generated the OPS C code.")
        return

    def kernel_header(self, tuple_list):
        """Function to generate a kernel function head compatible to the new OPS ACC template"""
        code = []
        dtype = SimulationDataType.opsc()
        for key, val in (tuple_list):
            Debug("Key=",key,"Type",type(key))
            # if any of the list has the datatype then use the data type
            if isinstance(key,DataSetBase):
                if hasattr(key, "datatype") and key.datatype:
                    code += [self.ops_headers_acc[val] % ("ACC<"+key.datatype.opsc()+">", key)]
                else:
                    code += [self.ops_headers_acc[val] % ("ACC<"+dtype+">", key)]
            else:
                if hasattr(key, "datatype") and key.datatype:
                    code += [self.ops_headers_point[val] % (key.datatype.opsc(), key)]
                else:
                    code += [self.ops_headers_point[val] % (dtype, key)]
        code = ', '.join(code)
        return code

    def kernel_computation_opsc(self, kernel):
        """ Function to write the out the contents of each computational kernel function."""
        ins = kernel.rhs_datasetbases
        outs = kernel.lhs_datasetbases
        inouts = ins.intersection(outs)
        ins = ins.difference(inouts)
        outs = outs.difference(inouts)
        # eqs = kernel.equations
        all_dataset_inps = list(ins) + list(outs) + list(inouts)
        all_dataset_types = ['input' for i in ins] + ['output' for o in outs] + ['inout' for io in inouts]
        # add the global variables to the inputs and outputs
        global_ins, global_outs = kernel.global_variables
        if global_ins.intersection(global_outs):
            raise NotImplementedError("Input output of global variables is not implemented")
        all_dataset_inps += list(global_ins) + list(global_outs)
        all_dataset_types += ['input' for i in global_ins] + ['output' for o in global_outs]
        # Use list of tuples as dictionary messes the order
        header_dictionary = list(zip(all_dataset_inps, all_dataset_types))
        if kernel.IndexedConstants:
            for i in kernel.IndexedConstants:
                header_dictionary += [tuple([(i.base), 'input'])]
        other_inputs = ""
        if kernel.grid_indices_used:
            # print kernel.grid_index_name
            other_inputs += ", const int *idx"  # WARNING hard coded here
        else:
            other_inputs = ''
        # print header_dictionary
        code = ["void %s(" % kernel.kernelname + self.kernel_header(header_dictionary) + other_inputs + ')' + '\n{']
        ops_accs = [OPSAccess(no) for no in range(len(all_dataset_inps))]
        ACCCodePrinter.dataset_accs_dictionary = dict(zip(all_dataset_inps, ops_accs))

        # Find all the grid variables and declare them at the top
        gridvariables = set()
        out = []
        for eq in kernel.equations:
            gridvariables = gridvariables.union(eq.atoms(GridVariable))
            if isinstance(eq, Equality):
                out += [ccodeAcc(eq, settings={'kernel': True}) + ';\n']
            elif isinstance(eq, GroupedPiecewise):
                for i, (expr, condition) in enumerate(eq.args):
                    if i == 0:
                        out += ['if (%s)' % ccodeAcc(condition, settings={'kernel': True}) + '{\n']
                        if is_sequence(expr):
                            for eqn in expr:
                                out += [ccodeAcc(eqn, settings={'kernel': True}) + ';\n']
                        else:
                            out += [ccodeAcc(expr, settings={'kernel': True}) + ';\n']
                        out += ['}\n']
                    elif condition != True:
                        out += ['else if (%s)' % ccodeAcc(condition, settings={'kernel': True}) + '{\n']
                        if is_sequence(expr):
                            for eqn in expr:
                                out += [ccodeAcc(eqn, settings={'kernel': True}) + ';\n']
                        else:
                            out += [ccodeAcc(expr, settings={'kernel': True}) + ';\n']
                        out += ['}\n']
                    else:
                        out += ['else{\n']
                        if is_sequence(expr):
                            for eqn in expr:
                                out += [ccodeAcc(eqn, settings={'kernel': True}) + ';\n']
                        else:
                            out += [ccodeAcc(expr, settings={'kernel': True}) + ';\n']
                        out += ['}\n']
            else:
                pprint(eq)
                raise TypeError("Unclassified type of equation.")
        for gv in gridvariables:
            code += ["%s %s = 0.0;" % (SimulationDataType.opsc(), str(gv))]
        code += out + ['}']  # close Kernel
        ACCCodePrinter.dataset_accs_dictionary = {}
        return code

    def write_kernels(self, algorithm):
        """ A function to write out the kernels header file definining all of the computations to be performed."""
        from opensbli.core.kernel import Kernel
        from opensbli.core.kernel import ImplicitKernel
        kernels = self.loop_alg(algorithm, Kernel)
        # Count the number of operations per kernel
        if self.operation_count:
            total = 0
            for k in kernels:
                n_operations = count_ops(k.equations)
                total += n_operations
                print([k.kernel_no, k.computation_name, 'operations: %d' % n_operations])
            print("Total operation count is: %d" % total)

        files = [open('%s_kernels.h' % b.block_name, 'w') for b in algorithm.block_descriptions]
        for i, f in enumerate(files):
            name = ('%s_kernel_H' % algorithm.block_descriptions[i].block_name).upper()
            f.write('#ifndef %s\n' % name)
            f.write('#define %s\n' % name)
        for k in kernels:
            if not isinstance(k,ImplicitKernel):
                out = self.kernel_computation_opsc(k) + ['\n']
                out = indent_code(out)
                out = self.wrap_long_lines(out)
                files[k.block_number].write('\n'.join(out))
        for f in files:
            f.write("#endif\n")
        files = [f.close() for f in files]
        return
    # revised
    def ops_exit(self):
        """ Exits the OPS program with optional kernel-based timing output."""
        output = []
        if self.OPS_diagnostics > 1:
            output += [WriteString("ops_timing_output(stdout);")]
        if self.monitoring_output_file:
            output += [WriteString("fclose(f);")]

        output += [WriteString(self.linear_solver.Finalise())]
        output += [WriteString("ops_exit();")]
        return output

    def before_main(self, algorithm):
        """ Adds the required preamble to the main opensbli.cpp file and declares the simulation constants."""
        out = ['#include <cstdio> \n#include <string> \n#include <cmath>']
        from opensbli.core.kernel import ConstantsToDeclare
        for d in ConstantsToDeclare.constants:
            if isinstance(d, ConstantObject):
                out += ["%s %s;" % (d.datatype.opsc(), d)]
            elif isinstance(d, ConstantIndexed):
                if not d.inline_array:
                    indices = ''
                    for s in d.shape:
                        indices = indices + '[%d]' % s
                    out += ["%s %s%s;" % (d.datatype.opsc(), d.base.label, indices)]
        for b in algorithm.block_descriptions:
            out += ['#define OPS_%dD' % b.ndim]

        out += ['#include \"ops_seq_v2.h\"']
        for b in algorithm.block_descriptions:
            out += ['#include \"%s_kernels.h\"' % b.block_name]

        out += ['#include \"'+self.scheme.kernel_file_name+'\"']
        out += [self.scheme.stencils]
        out += self.scheme.wrap_templates_1st
        # Include optional simulation monitoring reductions file
        if algorithm.simulation_monitor:
            out += ['#include \"%s\"' % algorithm.simulation_monitor.filename]
            if algorithm.simulation_monitor.output_file:
                out += ['FILE *f = fopen(\"%s\", \"w\");' % str(algorithm.simulation_monitor.output_file)]
        return out

    # revised
    def opsc_def_decs(self, algorithm):
        """ Declares the datasets and stencils required by the program."""
        from opensbli.core.kernel import StencilObject, ConstantsToDeclare
        from opensbli.core.boundary_conditions.exchange import Exchange

        defs = []
        decls = []
        # Add OPS_init to the declarations as it should be called before all ops
        decls += self.ops_init()
        # First process all the constants in the definitions
        for d in ConstantsToDeclare.constants:
            if isinstance(d, Constant):
                defs += self.define_constants(d)
                decls += self.declare_ops_constants(d)
        # Once the constants are done define and declare OPS dats
        output = defs + decls
        defs = []
        decls = []
        # Define and declare blocks
        for b in algorithm.block_descriptions:
            output += self.declare_block(b)
        # Define and declare datasets on each block
        f = open('defdec_data_set.h', 'w')
        datasets_dec = []
        output += [WriteString("#include \"defdec_data_set.h\"")]

        for opsdat in self.scheme.data_def:
            output +=  [WriteString(opsdat)]
        # Sort the declarations alphabetically before writing out
        store_stencils, store_dsets = [], []
        for d in algorithm.defnitionsdeclarations.components:
            if isinstance(d, DataSetBase):
                store_dsets.append(d)
            elif isinstance(d, StencilObject):
                store_stencils.append(d)
            else:
                print(d)
                print(type(d))
                raise TypeError("Not a stencil or dataset declaration.")
        dsets_to_declare = dict([(str(x), x) for x in store_dsets])

        for name in sorted(dsets_to_declare, key=str.lower):
            d = dsets_to_declare[name]
            datasets_dec += self.declare_dataset(d)
        f.write('\n'.join(flatten([dset.opsc_code for dset in datasets_dec])))
        f.close()
        # Declare stencils
        output += [WriteString("// Define and declare stencils")]
        for d in store_stencils:
            output += self.ops_stencils_declare(d)
        # Loop through algorithm components to include any halo exchanges
        exchange_list = self.loop_alg(algorithm, Exchange)
        if exchange_list:
            f = open('bc_exchanges.h', 'w')  # write BC_exchange code to a separate file
            exchange_code = []
            for e in exchange_list:
                call, code = self.bc_exchange_call_code(e)
                exchange_code += [code]
            f.write('\n'.join(flatten(exchange_code)))
            f.close()
            output += [WriteString("#include \"bc_exchanges.h\"")]  # Include statement in the code

        output += self.ops_partition()
        output += [WriteString("// initialize linear solver Library"), WriteString(self.linear_solver.Initialise())]
        return output

    def ops_stencils_declare(self, s):
        out = []
        dtype = s.dtype.opsc()
        name = s.name + 'temp'
        sorted_stencil = s.sort_stencil_indices()
        out = [self.declare_inline_array(dtype, name, [st for st in flatten(sorted_stencil) if not isinstance(st, Idx)])]
        out += [WriteString('ops_stencil %s = ops_decl_stencil(%d,%d,%s,\"%s\");' % (s.name, s.ndim, len(s.stencil), name, name))]
        return out

    def ops_partition(self):
        """ Initialise an OPS partition for the purpose of multi-block and/or MPI partitioning.

        :returns: The partitioning code in OPSC format. Each line is a separate list element.
        :rtype: list"""
        return [WriteString('// Init OPS partition'), WriteString('ops_partition(\"\");\n')]

    def linear_solver(self):
        return [WriteString('// Init OPS partition'), WriteString('ops_partition(\"\");\n')]

    def ops_init(self):
        """ The default diagnostics level is 1, which offers no diagnostic information and should be used for production runs.
        Refer to OPS user manual for more information.
        :returns: The call to ops_init.
        :rtype: list """
        out = [WriteString('// Initializing OPS ')]
        return out + [WriteString('ops_init(argc,argv,%d);' % (self.OPS_diagnostics))]

    def bc_exchange_call_code(self, instance):
        """ Generates the code for OPS exchanges. Used for example in the periodic boundary condition."""
        off = 0
        halo = 'halo'
        # instance.transfer_size = instance.transfer_from
        # Name of the halo exchange
        name = instance.name
        # self.halo_exchange_number = self.halo_exchange_number + 1
        code = ['// Boundary condition exchange code on %s direction %s %s' % (instance.block_name, instance.direction, instance.side)]
        code += ['ops_halo_group %s %s' % (name, ";")]
        code += ["{"]
        code += ['int halo_iter[] = {%s}%s' % (', '.join([str(s) for s in instance.transfer_size]), ";")]
        code += ['int from_base[] = {%s}%s' % (', '.join([str(s) for s in instance.transfer_from]), ";")]
        code += ['int to_base[] = {%s}%s' % (', '.join([str(s) for s in instance.transfer_to]), ";")]
        # dir in OPSC. WARNING: Not sure what it is, but 1 to ndim works.
        from_dir = [ind+1 for ind in range(len(instance.transfer_to))]
        to_dir = [ind+1 for ind in range(len(instance.transfer_to))]
        # MBCHANGE
        if instance.flip[-1]:
            to_dir[instance.flip[1]] = -to_dir[instance.flip[1]]
        # MBCHANGE
        code += ['int from_dir[] = {%s}%s' % (', '.join([str(ind) for ind in from_dir]), ";")]
        code += ['int to_dir[] = {%s}%s' % (', '.join([str(ind) for ind in to_dir]), ";")]
        # Process the arrays
        for no, arr in enumerate(instance.transfer_arrays):
            from_array = instance.from_arrays[no]
            to_array = instance.to_arrays[no]
            code += ['ops_halo %s%d = ops_decl_halo(%s, %s, halo_iter, from_base, to_base, from_dir, to_dir)%s'
                     % (halo, off, from_array.base, to_array.base, ";")]
            off = off+1
        code += ['ops_halo grp[] = {%s}%s' % (','.join([str('%s%s' % (halo, of)) for of in range(off)]), ";")]
        code += ['%s = ops_decl_halo_group(%d,grp)%s' % (name, off, ";")]
        code += ["}"]
        # Finished OPS halo exchange, now get the call
        instance.call_name = 'ops_halo_transfer(%s)%s' % (name, ";")
        call = ['// Boundary condition exchange calls', 'ops_halo_transfer(%s)%s' % (name, ";")]
        for no, c in enumerate(code):
            code[no] = WriteString(c).opsc_code
        return call, code

    def loop_alg(self, algorithm, type_of_component):
        type_list = []

        def _generate(components, type_list):
            for component1 in components:
                if hasattr(component1, 'components'):
                    _generate(component1.components, type_list)
                elif isinstance(component1, type_of_component):
                    if component1 in type_list:
                        pass
                    else:
                        type_list += [component1]
        for c in algorithm.prg.components:
            _generate([c], type_list)
        return type_list

    def define_block(self, b):
        if not self.MultiBlock:
            return [WriteString("ops_block %s;" % b.block_name)]
        else:
            raise NotImplementedError("")

    def declare_block(self, b):
        if not self.MultiBlock:
            out = [WriteString("// Define and Declare OPS Block")]
            out += [WriteString('ops_block %s = ops_decl_block(%d, \"%s\");' % (b.block_name, b.ndim, b.block_name))]
            return out
        else:
            raise NotImplementedError("")

    def define_constants(self, c):
        """ Declares all of the constants required by the simulation at the start of the program."""
        # Fix spacing on constant declarations %s=%s
        if isinstance(c, ConstantObject):
            if not isinstance(c.value, str):
                return [WriteString("%s = %s;" % (str(c), ccodeAcc(c.value, settings={'rational': True})))]
            else:
                return [WriteString("%s=%s;" % (str(c), c.value))]
        elif isinstance(c, ConstantIndexed):
            out = []
            if c.value:
                if len(c.shape) == 1:
                    if c.inline_array:
                        values = [ccodeAcc(c.value[i], settings={'rational': True}) for i in range(c.shape[0])]
                        return [WriteString("%s %s[] = {%s};" % (c.datatype.opsc(), c.base.label, ', '.join(values)))]
                    else:
                        indices = ''
                        for s in c.shape:
                            indices = indices + '[%d]' % s
                        out += [WriteString("%s %s%s;" % (c.base.label, indices))]
                        for i in range(c.shape[0]):
                            out += [WriteString("%s[%d] = %s;" % (str(c.base.label), i, ccodeAcc(c.value[i], settings={'rational': True})))]
                        return out
                else:
                    raise NotImplementedError("Indexed constant declaration is done for only one ")
            else:
                raise NotImplementedError("")
        else:
            print(c)
            raise ValueError("")

    def declare_ops_constants(self, c):
        """ Calls the OPS declare constant function for all of the defined constants."""
        if isinstance(c, ConstantObject):
            return [WriteString("ops_decl_const(\"%s\" , 1, \"%s\", &%s);" % (str(c), c.datatype.opsc(), str(c)))]
        elif isinstance(c, ConstantIndexed):
            return []
        return

    def declare_inline_array(self, dtype, name, values):
        return WriteString('%s %s[] = {%s};' % (dtype, name, ', '.join([str(s) for s in values])))

    def update_inline_array(self, name, values):
        out = []
        for no, v in enumerate(values):
            out += [WriteString("%s[%d] = %s;" % (name, no, v))]
        return out

    def define_dataset(self, dset):
        if not self.MultiBlock:
            return [WriteString("ops_dat %s;" % (dset))]

    def get_max_halos(self, halos):
        halo_m = []
        halo_p = []
        for direction in range(len(halos)):
            if halos[direction][0]:
                hal = [d.get_halos(0) for d in halos[direction][0]]
                halo_m += [int(min(hal))]
            else:
                halo_m += [0]
            if halos[direction][1]:
                hal = [d.get_halos(1) for d in halos[direction][1]]
                halo_p += [int(max(hal))]
            else:
                halo_p += [0]
        return halo_m, halo_p

    def declare_dataset(self, dset):
        declaration = WriteString("ops_dat %s;" % dset)
        out = [declaration, WriteString("{")]

        if dset.dtype:
            dtype = dset.dtype
        else:
            dtype = SimulationDataType.dtype()

        if dset.read_from_hdf5:
            temp = '%s = ops_decl_dat_hdf5(%s, 1, \"%s\", \"%s\", \"%s\");' % (dset,dset.block_name, dtype.opsc(), dset, dset.input_file_name)
            out += [WriteString(temp)]
        else:
            # Residual and time-advance arrays do not require halos
            if ('Residual' in str(dset) or 'tempRK' in str(dset) or 'RKold' in str(dset)):
                hm, hp = [0 for _ in range(len(dset.size))], [0 for _ in range(len(dset.size))]
            else:
                hm, hp = self.get_max_halos(dset.halo_ranges)
            halo_p = self.declare_inline_array("int", "halo_p", hp)
            halo_m = self.declare_inline_array("int", "halo_m", hm)
            sizes = self.declare_inline_array("int", "size", [str(s) for s in (dset.size)])
            base = self.declare_inline_array("int", "base", [0 for i in range(len(dset.size))])
            value = WriteString("%s* value = NULL;" % dtype.opsc())
            temp = '%s = ops_decl_dat(%s, 1, size, base, halo_m, halo_p, value, \"%s\", \"%s\");' % (dset,
                                                                                                     dset.block_name, dtype.opsc(), dset)
            out += [halo_p, halo_m, sizes, base, value, WriteString(temp)]

        out += [WriteString("}")]
        return out
