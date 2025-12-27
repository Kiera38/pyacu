from collections.abc import Iterable, Iterator
from acu.refanal.flow_graph_ir import (
    BasicBlock,
    Branch,
    ControlOp,
    Float,
    Goto,
    Integer,
    Op,
    RegisterOp,
    Return,
    Undef,
    Unreachable,
    Value,
    Store,
    OpVisitor,
)


class CFG:
    """Control-flow graph.

    Node 0 is always assumed to be the entry point. There must be a
    non-empty set of exits.
    """

    def __init__(
        self,
        succ: dict[BasicBlock, list[BasicBlock]],
        pred: dict[BasicBlock, list[BasicBlock]],
        exits: set[BasicBlock],
    ) -> None:
        assert exits
        self.succ = succ
        self.pred = pred
        self.exits = exits

    def __str__(self) -> str:
        exits = sorted(self.exits, key=lambda e: int(e.label))
        return f"exits: {exits}\nsucc: {self.succ}\npred: {self.pred}"


def get_cfg(blocks: list[BasicBlock], *, use_yields: bool = False) -> CFG:
    """Calculate basic block control-flow graph.

    If use_yields is set, then we treat returns inserted by yields as gotos
    instead of exits.
    """
    succ_map = {}
    pred_map: dict[BasicBlock, list[BasicBlock]] = {}
    exits = set()
    for block in blocks:
        assert not any(
            isinstance(op, ControlOp) for op in block.ops[:-1]
        ), "Control-flow ops must be at the end of blocks"

        succ = list(block.terminator.targets())
        if not succ:
            exits.add(block)

        succ_map[block] = succ
        pred_map[block] = []
    for prev, nxt in succ_map.items():
        for label in nxt:
            pred_map[label].append(prev)
    return CFG(succ_map, pred_map, exits)


def get_real_target(label: BasicBlock) -> BasicBlock:
    if len(label.ops) == 1 and isinstance(label.ops[-1], Goto):
        label = label.ops[-1].label
    return label


def cleanup_cfg(blocks: list[BasicBlock]) -> None:
    """Cleanup the control flow graph.

    This eliminates obviously dead basic blocks and eliminates blocks that contain
    nothing but a single jump.

    There is a lot more that could be done.
    """
    changed = True
    while changed:
        # First collapse any jumps to basic block that only contain a goto
        for block in blocks:
            for i, tgt in enumerate(block.terminator.targets()):
                block.terminator.set_target(i, get_real_target(tgt))

        # Then delete any blocks that have no predecessors
        changed = False
        cfg = get_cfg(blocks)
        orig_blocks = blocks.copy()
        blocks.clear()
        for i, block in enumerate(orig_blocks):
            if i == 0 or cfg.pred[block]:
                blocks.append(block)
            else:
                changed = True


type AnalysisDict[T] = dict[tuple[BasicBlock, int], set[T]]


class AnalysisResult[T]:
    def __init__(self, before: AnalysisDict[T], after: AnalysisDict[T]) -> None:
        self.before = before
        self.after = after

    def __str__(self) -> str:
        return f"before: {self.before}\nafter: {self.after}\n"


type GenAndKill[T] = tuple[set[T], set[T]]


class BaseAnalysisVisitor[T](OpVisitor[GenAndKill[T]]):
    def goto(self, op: Goto) -> GenAndKill[T]:
        return set(), set()


class DefinedVisitor(BaseAnalysisVisitor[Value]):
    def branch(self, op: Branch) -> GenAndKill[Value]:
        return set(), set()

    def visit_return(self, op: Return) -> GenAndKill[Value]:
        return set(), set()

    def unreachable(self, op: Unreachable) -> GenAndKill[Value]:
        return set(), set()

    def register_op(self, op: RegisterOp) -> GenAndKill[Value]:
        return set(), set()

    def store(self, op: Store) -> GenAndKill[Value]:
        return {op.dest}, set()


def analyze_maybe_defined_regs(
    blocks: list[BasicBlock], cfg: CFG, initial_defined: set[Value]
) -> AnalysisResult[Value]:
    """Calculate potentially defined registers at each CFG location.

    A register is defined if it has a value along some path from the initial location.
    """
    return run_analysis(
        blocks=blocks,
        cfg=cfg,
        gen_and_kill=DefinedVisitor(),
        initial=initial_defined,
        backward=False,
        kind=MAYBE_ANALYSIS,
    )


def analyze_must_defined_regs(
    blocks: list[BasicBlock],
    cfg: CFG,
    initial_defined: set[Value],
    regs: Iterable[Value],
) -> AnalysisResult[Value]:
    """Calculate always defined registers at each CFG location.

    This analysis can work before exception insertion, since it is a
    sound assumption that registers defined in a block might not be
    initialized in its error handler.

    A register is defined if it has a value along all paths from the
    initial location.
    """
    return run_analysis(
        blocks=blocks,
        cfg=cfg,
        gen_and_kill=DefinedVisitor(),
        initial=initial_defined,
        backward=False,
        kind=MUST_ANALYSIS,
        universe=set(regs),
    )


def non_trivial_sources(op: Op) -> set[Value]:
    result = set()
    for source in op.sources():
        if not isinstance(source, (Integer, Float, Undef)):
            result.add(source)
    return result


class LivenessVisitor(BaseAnalysisVisitor[Value]):
    def branch(self, op: Branch) -> GenAndKill[Value]:
        return non_trivial_sources(op), set()

    def visit_return(self, op: Return) -> GenAndKill[Value]:
        if not isinstance(op.value, (Integer, Float)):
            return {op.value}, set()
        else:
            return set(), set()

    def unreachable(self, op: Unreachable) -> GenAndKill[Value]:
        return set(), set()

    def register_op(self, op: RegisterOp) -> GenAndKill[Value]:
        gen = non_trivial_sources(op)
        if not op.is_void:
            return gen, {op}
        else:
            return gen, set()

    def store(self, op: Store) -> GenAndKill[Value]:
        return non_trivial_sources(op), {op.dest}
    

def analyze_live_regs(blocks: list[BasicBlock], cfg: CFG) -> AnalysisResult[Value]:
    """Calculate live registers at each CFG location.

    A register is live at a location if it can be read along some CFG path starting
    from the location.
    """
    return run_analysis(
        blocks=blocks,
        cfg=cfg,
        gen_and_kill=LivenessVisitor(),
        initial=set(),
        backward=True,
        kind=MAYBE_ANALYSIS,
    )


# Analysis kinds
MUST_ANALYSIS = 0
MAYBE_ANALYSIS = 1


def run_analysis[T](
    blocks: list[BasicBlock],
    cfg: CFG,
    gen_and_kill: OpVisitor[GenAndKill[T]],
    initial: set[T],
    kind: int,
    backward: bool,
    universe: set[T] | None = None,
) -> AnalysisResult[T]:
    block_gen = {}
    block_kill = {}

    # Calculate kill and gen sets for entire basic blocks.
    for block in blocks:
        gen: set[T] = set()
        kill: set[T] = set()
        ops = block.ops
        if backward:
            ops = list(reversed(ops))
        for op in ops:
            opgen, opkill = op.accept(gen_and_kill)
            gen = (gen - opkill) | opgen
            kill = (kill - opgen) | opkill
        block_gen[block] = gen
        block_kill[block] = kill

    # Set up initial state for worklist algorithm.
    worklist = list(blocks)
    if not backward:
        worklist.reverse()  # Reverse for a small performance improvement
    workset = set(worklist)
    before: dict[BasicBlock, set[T]] = {}
    after: dict[BasicBlock, set[T]] = {}
    for block in blocks:
        if kind == MAYBE_ANALYSIS:
            before[block] = set()
            after[block] = set()
        else:
            assert universe is not None, "Universe must be defined for a must analysis"
            before[block] = set(universe)
            after[block] = set(universe)

    if backward:
        pred_map = cfg.succ
        succ_map = cfg.pred
    else:
        pred_map = cfg.pred
        succ_map = cfg.succ

    # Run work list algorithm to generate in and out sets for each basic block.
    while worklist:
        label = worklist.pop()
        workset.remove(label)
        if pred_map[label]:
            new_before: set[T] | None = None
            for pred in pred_map[label]:
                if new_before is None:
                    new_before = set(after[pred])
                elif kind == MAYBE_ANALYSIS:
                    new_before |= after[pred]
                else:
                    new_before &= after[pred]
            assert new_before is not None
        else:
            new_before = set(initial)
        before[label] = new_before
        new_after = (new_before - block_kill[label]) | block_gen[label]
        if new_after != after[label]:
            for succ in succ_map[label]:
                if succ not in workset:
                    worklist.append(succ)
                    workset.add(succ)
        after[label] = new_after

    # Run algorithm for each basic block to generate opcode-level sets.
    op_before: dict[tuple[BasicBlock, int], set[T]] = {}
    op_after: dict[tuple[BasicBlock, int], set[T]] = {}
    for block in blocks:
        label = block
        cur = before[label]
        ops_enum: Iterator[tuple[int, Op]] = enumerate(block.ops)
        if backward:
            ops_enum = reversed(list(ops_enum))
        for idx, op in ops_enum:
            op_before[label, idx] = cur
            opgen, opkill = op.accept(gen_and_kill)
            cur = (cur - opkill) | opgen
            op_after[label, idx] = cur
    if backward:
        op_after, op_before = op_before, op_after

    return AnalysisResult(op_before, op_after)
