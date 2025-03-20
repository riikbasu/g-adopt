from gadopt import *
from gadopt.inverse import *
from checkpoint_schedules import SingleMemoryStorageSchedule
from pyadjoint.block import Block

if not annotate_tape():
    continue_annotation()

tape = get_working_tape()
tape.clear_tape()
tape.enable_checkpointing(SingleMemoryStorageSchedule())

mesh = UnitSquareMesh(10, 10)
Q = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)

# an initial field is the control
T_0 = Function(Q, name="control").assign(1.0)

control = Control(T_0)

# state variable will recorded as T
T = Function(Q, name="state").assign(T_0)

T.assign(T + 0.1)

weight = Function(R, name="weight").assign(1.0)


class VariableBlock(Block):
    """
    A block that computes the functional value of the difference between the
    """

    def __init__(self, function):
        """Initialises the Diagnostic block.
        """
        super().__init__()
        self.add_dependency(function)
        self.add_output(function.create_block_variable())
        self.my_idx = 0

    def recompute_component(self, inputs, block_variable, idx, prepared):
        is_on_disk = hasattr(block_variable.checkpoint, "restore")
        with stop_annotating():
            restored_function = block_variable.checkpoint.restore() if is_on_disk else block_variable.checkpoint
            restored_function.assign(restored_function + 1.0)
            self.my_idx += 1

        return restored_function

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return block_variable


diagnostic_block = VariableBlock(weight)
tape.add_block(diagnostic_block)
diagnostic_block.recompute()

J = assemble(weight * (T - 0.5)**2 * dx)
pause_annotation()
rf = ReducedFunctional(J, control)
