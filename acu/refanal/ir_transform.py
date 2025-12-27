from typing import Final

from acu.refanal.flow_graph_ir import (
    Array,
    BasicBlock,
    Binary,
    Branch,
    Call,
    Cast,
    Comparison,
    ControlOp,
    CreateStruct,
    Deref,
    GetAddress,
    GetField,
    GetFieldPtr,
    GetItem,
    GetItemPtr,
    GetPtr,
    Goto,
    LoadPtr,
    Op,
    OpVisitor,
    Return,
    SetField,
    SetItem,
    Store,
    StorePtr,
    Unary,
    Unreachable,
    Value,
)


class IRTransform(OpVisitor[Value | None]):
    def __init__(self) -> None:
        self.op_map: dict[Value, Value | None] = {}
        self.current: BasicBlock | None = None
        self.blocks = []

    def transform_blocks(self, blocks: list[BasicBlock]) -> list[BasicBlock]:
        block_map: dict[BasicBlock, BasicBlock] = {}
        op_map = self.op_map
        empties = set()
        for block in blocks:
            new_block = BasicBlock()
            block_map[block] = new_block
            self.blocks.append(new_block)
            self.current = new_block
            for op in block.ops:
                new_op = op.accept(self)
                if new_op is not op:
                    op_map[op] = new_op
            # A transform can produce empty blocks which can be removed.
            if is_empty_block(new_block) and not is_empty_block(block):
                empties.add(new_block)
        self.blocks = [block for block in self.blocks if block not in empties]
        # Update all op/block references to point to the transformed ones.
        patcher = PatchVisitor(op_map, block_map)
        for block in self.blocks:
            for op in block.ops:
                op.accept(patcher)
        return self.blocks

    def add(self, op: Op) -> Value:
        if self.current is None:
            raise RuntimeError("No current block to add op to.")
        self.current.ops.append(op)
        return op

    def op(self, op: Op) -> Value | None:
        return self.add(op)

    def control_op(self, op: ControlOp) -> Value | None:
        self.add(op)

    def store(self, op: Store) -> Value | None:
        if op.src in self.op_map and self.op_map[op.src] is None:
            # Eliminate stores of eliminated values.
            return None
        return self.add(op)


class PatchVisitor(OpVisitor[None]):
    def __init__(
        self, op_map: dict[Value, Value | None], block_map: dict[BasicBlock, BasicBlock]
    ) -> None:
        self.op_map: Final = op_map
        self.block_map: Final = block_map

    def fix_op(self, op: Value) -> Value:
        new = self.op_map.get(op, op)
        assert new is not None, "use of removed op"
        return new

    def fix_block(self, block: BasicBlock) -> BasicBlock:
        return self.block_map.get(block, block)

    def goto(self, op: Goto) -> None:
        op.label = self.fix_block(op.label)

    def branch(self, op: Branch) -> None:
        op.value = self.fix_op(op.value)
        op.true_label = self.fix_block(op.true_label)
        op.false_label = self.fix_block(op.false_label)

    def visit_return(self, op: Return) -> None:
        op.value = self.fix_op(op.value)

    def visit_unreachable(self, op: Unreachable) -> None:
        pass

    def store(self, op: Store) -> None:
        op.src = self.fix_op(op.src)

    def get_field(self, op: GetField) -> None:
        op.obj = self.fix_op(op.obj)

    def set_field(self, op: SetField) -> None:
        op.obj = self.fix_op(op.obj)
        op.value = self.fix_op(op.value)

    def call(self, op: Call) -> None:
        op.args = [self.fix_op(arg) for arg in op.args]

    def cast(self, op: Cast) -> None:
        op.obj = self.fix_op(op.obj)

    def binary(self, op: Binary) -> None:
        op.left = self.fix_op(op.left)
        op.right = self.fix_op(op.right)

    def comparison_op(self, op: Comparison) -> None:
        op.left = self.fix_op(op.left)
        op.right = self.fix_op(op.right)

    def get_item(self, op: GetItem) -> None:
        op.obj = self.fix_op(op.obj)
        op.index = self.fix_op(op.index)

    def set_item(self, op: SetItem) -> None:
        op.obj = self.fix_op(op.obj)
        op.index = self.fix_op(op.index)
        op.value = self.fix_op(op.value)

    def get_address(self, op: GetAddress) -> None:
        op.src = self.fix_op(op.src)

    def unary(self, op: Unary) -> None:
        op.value = self.fix_op(op.value)

    def array(self, op: Array) -> None:
        op.items = [self.fix_op(elem) for elem in op.items]

    def create_struct(self, op: CreateStruct) -> None:
        op.fields = [self.fix_op(field) for field in op.fields]

    def comparison(self, op: Comparison) -> None:
        op.left = self.fix_op(op.left)
        op.right = self.fix_op(op.right)

    def load_ptr(self, op: LoadPtr) -> None:
        op.ptr = self.fix_op(op.ptr)

    def get_ptr(self, op: GetPtr) -> None:
        op.src = self.fix_op(op.src)

    def get_field_ptr(self, op: GetFieldPtr) -> None:
        op.ptr = self.fix_op(op.ptr)

    def get_item_ptr(self, op: GetItemPtr) -> None:
        op.ptr = self.fix_op(op.ptr)
        op.index = self.fix_op(op.index)

    def store_ptr(self, op: StorePtr) -> None:
        op.ptr = self.fix_op(op.ptr)
        op.value = self.fix_op(op.value)

    def deref(self, op: Deref) -> None:
        op.ptr = self.fix_op(op.ptr)

    def unreachable(self, op: Unreachable) -> None:
        pass

    def op(self, op: Op) -> None:
        raise NotImplementedError


def is_empty_block(block: BasicBlock) -> bool:
    return len(block.ops) == 1 and isinstance(block.ops[0], Unreachable)
