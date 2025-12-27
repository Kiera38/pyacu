from acu.refanal.flow_graph_ir import FuncIR, GetAddress, Register, Store, Value
from acu.refanal.ir_transform import IRTransform


def do_copy_propagation(fn: FuncIR) -> None:
    """Perform copy propagation optimization for fn."""

    counts: dict[Value, int] = {}
    replacements: dict[Value, Value] = {}
    for arg in fn.args:
        counts[arg] = 1

    for block in fn.blocks:
        for op in block.ops:
            if isinstance(op, Store):
                c = counts.get(op.dest, 0)
                counts[op.dest] = c + 1
                if c == 1:
                    replacements.pop(op.dest, 0)
            elif isinstance(op, GetAddress):
                if isinstance(op.src, Register):
                    counts[op.src] = 2
                    replacements.pop(op.src, 0)

    for src, dst in list(replacements.items()):
        if counts.get(dst, 0) > 1:
            del replacements[src]
        else:
            while dst in replacements:
                dst = replacements[dst]
                if counts.get(dst, 0) > 1:
                    del replacements[src]
        if src in replacements:
            replacements[src] = dst

    transform = CopyPropagationTransform(replacements)
    transform.transform_blocks(fn.blocks)
    fn.blocks = transform.blocks


class CopyPropagationTransform(IRTransform):
    def __init__(self, map: dict[Value, Value]) -> None:
        super().__init__()
        self.op_map.update(map)
        self.removed = set(map)

    def store(self, op: Store) -> Value | None:
        if op.dest in self.removed:
            return None
        return self.add(op)