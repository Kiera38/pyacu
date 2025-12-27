from acu.refanal.flow_graph_ir import (
    BasicBlock,
    Branch,
    FuncIR,
    Goto,
    Register,
    Store,
    Unreachable,
)
from acu.refanal.ir_transform import IRTransform


def do_flag_elimination(fn: FuncIR) -> None:
    # Find registers that are used exactly once as source, and in a branch.
    counts: dict[Register, int] = {}
    branches: dict[Register, Branch] = {}
    labels: dict[Register, BasicBlock] = {}
    for block in fn.blocks:
        for i, op in enumerate(block.ops):
            for src in op.sources():
                if isinstance(src, Register):
                    counts[src] = counts.get(src, 0) + 1
            if i == 0 and isinstance(op, Branch) and isinstance(op.value, Register):
                branches[op.value] = op
                labels[op.value] = block

    # Based on these we can find the candidate registers.
    candidates: set[Register] = {
        r for r in branches if counts.get(r, 0) == 1 and r not in fn.args
    }

    # Remove candidates with invalid assignments.
    for block in fn.blocks:
        for i, op in enumerate(block.ops):
            if isinstance(op, Store) and op.dest in candidates:
                next_op = block.ops[i + 1]
                if isinstance(op.dest, Register) and not (
                    isinstance(next_op, Goto) and next_op.label is labels[op.dest]
                ):
                    # Not right
                    candidates.remove(op.dest)

    transform = FlagEliminationTransform(
        {x: y for x, y in branches.items() if x in candidates}
    )
    fn.blocks = transform.transform_blocks(fn.blocks)


class FlagEliminationTransform(IRTransform):
    def __init__(self, branch_map: dict[Register, Branch]) -> None:
        super().__init__()
        self.branch_map = branch_map
        self.branches = set(branch_map.values())

    def store(self, op: Store) -> None:
        if isinstance(op.dest, Register) and (
            old_branch := self.branch_map.get(op.dest)
        ):
            # Replace assignment with a copy of the old branch, which is in a
            # separate basic block. The old branch will be deleted in visit_branch.
            new_branch = Branch(
                op.location,
                op.src,
                old_branch.true_label,
                old_branch.false_label,
            )
            self.add(new_branch)
        else:
            self.add(op)

    def goto(self, op: Goto) -> None:
        # This is a no-op if basic block already terminated
        if not self.blocks[-1].terminated:
            self.add(Goto(op.location, op.label))

    def branch(self, op: Branch) -> None:
        if op in self.branches:
            # This branch is optimized away
            self.add(Unreachable(op.location))
        else:
            self.add(op)
