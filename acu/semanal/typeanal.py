from __future__ import annotations

from collections.abc import Callable, Iterable

from acu.semanal.ir import (
    AddressOf,
    Arg,
    Array,
    Binary,
    BinaryOp,
    Block,
    Break,
    Call,
    Comparison,
    Continue,
    Deref,
    Func,
    GetAttr,
    GetItem,
    If,
    Inst,
    InstVisitor,
    Literal,
    Load,
    Logical,
    Loop,
    Return,
    SetAttr,
    SetItem,
    Store,
    Unary,
    UnaryOp,
    VarDecl,
)
from acu.semanal.types import (
    ArrayType,
    Builtin,
    BuiltinType,
    FuncType,
    PointerType,
    Struct,
    StructType,
    Type,
)
from acu.source import Location


class TypeVar:
    def __init__(self, analyzer: TypeAnalyzer, inst: Inst) -> None:
        self.analyzer = analyzer
        self.inst = inst
        self.type = None
        self.locked = False
        self.location = None

    def add_type(self, type: Type, location: Location):
        if self.locked:
            if type != self.type:
                assert self.type is not None
                if not type.can_convert(self.type):
                    raise Exception("cannot convert type")
        else:
            if self.type is not None:
                unified = self.analyzer.unify_types((self.type, type))
                if unified is None:
                    raise Exception("connot unify types")
            else:
                unified = type
                self.location = location
            self.type = unified
        return self.type

    def lock(self, type: Type, location: Location):
        assert not self.locked
        if self.type is not None and not self.type.can_convert(type):
            raise Exception("no conversion type")
        self.type = type
        self.locked = True
        if not self.location:
            self.location = location

    def union(self, other: TypeVar, location: Location):
        if other.type is not None:
            self.add_type(other.type, location)

    @property
    def defined(self):
        return self.type is not None

    def get(self):
        return (self.type,) if self.type is not None else ()

    def get_one(self):
        if self.type is None:
            raise Exception("undecided type")
        return self.type

    def __len__(self):
        return 1 if self.type is not None else 0


class TypeVarMap(dict[Inst, TypeVar]):
    def __init__(self, analyzer: TypeAnalyzer):
        super().__init__()
        self.analyzer = analyzer

    def __getitem__(self, inst: Inst) -> TypeVar:
        if inst not in self:
            self[inst] = TypeVar(self.analyzer, inst)
        return super().__getitem__(inst)

    def __setitem__(self, inst: Inst, value: TypeVar) -> None:
        if inst in self:
            raise KeyError("Cannot redefine typevar %s" % inst)
        else:
            super().__setitem__(inst, value)


class TypeAnalyzer(InstVisitor[None]):
    def __init__(self, func: Func) -> None:
        super().__init__()
        self.func = func
        self.typevars = TypeVarMap(self)
        self.refine_map: dict[Inst, Callable[[Type], None]] = {}
        self.first = True

    def lock_type(self, inst: Inst, type: Type, location: Location):
        self.typevars[inst].lock(type, location)

    def copy_type(self, src_inst: Inst, dst_inst: Inst, location: Location):
        self.typevars[dst_inst].union(self.typevars[src_inst], location)

    def unify_types(self, types: Iterable[Type]):
        for first in types:
            if all(second.can_convert(first) for second in types):
                return first
        return None

    def add_type(self, inst: Inst, type: Type, location: Location):
        tv = self.typevars[inst]
        old_type = tv.type
        unified = tv.add_type(type, location)
        if old_type != unified:
            assert unified is not None
            self.propagate_refined_type(inst, unified)

    def propagate_refined_type(self, inst: Inst, type: Type):
        refine = self.refine_map.get(inst)
        if refine is not None:
            refine(type)

    def propagate(self):
        old_state = self.get_state()
        self.propagate_block(self.func.code)
        new_state = self.get_state()
        self.first = False
        return old_state == new_state

    def propagate_block(self, block: Block):
        for inst in block.code:
            inst.accept(self)

    def unify(self) -> dict[Inst, Type]:
        types = {}
        for inst, tp in self.typevars.items():
            if not tp.defined:
                raise Exception("unknown type")
            types[inst] = tp.get_one()
        return types

    def get_state(self):
        return [tv.type for tv in self.typevars.values()]

    def literal(self, inst: Literal):
        if not self.first:
            return
        match inst.value:
            case int():
                self.lock_type(inst, BuiltinType(Builtin.INT), inst.location)
            case float():
                self.lock_type(inst, BuiltinType(Builtin.FLOAT), inst.location)
            case Func():
                self.lock_type(inst, inst.value.get_type(), inst.location)
            case Struct():
                self.lock_type(inst, StructType(inst.value), inst.location)
            case _:
                raise Exception("unsupported literal")

    def load(self, inst: Load) -> None:
        self.copy_type(inst.var, inst, inst.location)

    def store(self, inst: Store) -> None:
        self.copy_type(inst.value, inst.var, inst.location)

        def refine(type: Type):
            self.add_type(inst.value, type, inst.location)

        self.refine_map[inst.var] = refine

    def arg(self, inst: Arg) -> None:
        if not self.first:
            return
        assert inst.type is not None
        self.lock_type(inst, inst.type, inst.location)

    def binary(self, inst: Binary) -> None:
        left = self.typevars[inst.left]
        right = self.typevars[inst.right]
        if not left.defined or not right.defined:
            return
        if type := self.unify_types((left.get_one(), right.get_one())):
            if inst.op in (
                BinaryOp.ADD,
                BinaryOp.SUB,
                BinaryOp.MUL,
                BinaryOp.DIV,
                BinaryOp.MOD,
            ):
                if not isinstance(type, BuiltinType) or type.type not in (
                    Builtin.INT,
                    Builtin.FLOAT,
                ):
                    raise Exception("operation is not supported")
            else:
                if not isinstance(type, BuiltinType) or type.type != Builtin.INT:
                    raise Exception("operation is not supported")
            self.add_type(inst, type, inst.location)
        else:
            raise Exception("cannot unify types")

    def unary(self, inst: Unary) -> None:
        value = self.typevars[inst.value]
        if not value.defined:
            return
        type = value.get_one()
        if inst.op == UnaryOp.NOT:
            self.add_type(inst, BuiltinType(Builtin.BOOL), inst.location)
            if not type.can_convert(BuiltinType(Builtin.BOOL)):
                raise Exception("cannot convert type")
        elif inst.op == UnaryOp.BIT_NOT:
            self.add_type(inst, BuiltinType(Builtin.INT), inst.location)
            if type != BuiltinType(Builtin.INT):
                raise Exception("unsupported operation")
        else:
            if not isinstance(type, BuiltinType) or type.type not in (
                Builtin.INT,
                Builtin.FLOAT,
            ):
                raise Exception("unsupported operation")
            self.add_type(inst, type, inst.location)

    def get_block_type_var(self, block: Block) -> TypeVar:
        self.propagate_block(block)
        return self.typevars[block.code[-1]]

    def logical(self, inst: Logical) -> None:
        self.add_type(inst, BuiltinType(Builtin.BOOL), inst.location)
        left = self.typevars[inst.left]
        if not left.defined:
            return
        if not left.get_one().can_convert(BuiltinType(Builtin.BOOL)):
            raise Exception("cannot convert type")
        right = self.get_block_type_var(inst.right)
        if not right.defined:
            return
        if not right.get_one().can_convert(BuiltinType(Builtin.BOOL)):
            raise Exception("cannot convert type")

    def comparison(self, inst: Comparison) -> None:
        self.add_type(inst, BuiltinType(Builtin.BOOL), inst.location)
        left = self.typevars[inst.left]
        if not left.defined:
            return
        left_type = left.get_one()
        for comparator in inst.comparators:
            right = self.get_block_type_var(comparator.value)
            if not right.defined:
                return
            right_type = right.get_one()
            if not self.unify_types((left_type, right_type)):
                raise Exception("cannot unify types")
            left_type = right_type

    def call(self, inst: Call) -> None:
        value = self.typevars[inst.value]
        if not value.defined:
            return
        value_type = value.get_one()
        if isinstance(value_type, FuncType):
            for arg, type in zip(inst.args, value_type.args):
                arg_tv = self.typevars[arg]
                if not arg_tv.defined:
                    return
                arg_type = arg_tv.get_one()
                if not arg_type.can_convert(type):
                    raise Exception(f"cannot convert type {type} to {arg_type}")
            self.add_type(inst, value_type.return_type, inst.location)
        elif isinstance(value_type, StructType):
            for arg, field in zip(inst.args, value_type.struct.fields.values()):
                arg_tv = self.typevars[arg]
                if not arg_tv.defined:
                    return
                arg_type = arg_tv.get_one()
                if not arg_type.can_convert(field.type):
                    raise Exception("cannot convert type")
            self.add_type(inst, value_type.struct, inst.location)
        else:
            raise Exception("type not callable")

    def loop(self, inst: Loop) -> None:
        self.propagate_block(inst.block)
        self.add_type(inst, BuiltinType(Builtin.NOTHING), inst.location)

    def if_inst(self, inst: If) -> None:
        cond_tv = self.typevars[inst.value]
        if not cond_tv.defined:
            return
        if not cond_tv.get_one().can_convert(BuiltinType(Builtin.BOOL)):
            raise Exception("cannot convert type")
        self.propagate_block(inst.then_block)
        self.propagate_block(inst.else_block)
        self.add_type(inst, BuiltinType(Builtin.NOTHING), inst.location)

    def return_inst(self, inst: Return) -> None:
        if inst.value is None:
            if self.func.return_type != BuiltinType(Builtin.NOTHING):
                raise Exception("need return value")
        else:
            value_tv = self.typevars[inst.value]
            if not value_tv.defined:
                return
            if not value_tv.get_one().can_convert(self.func.return_type):
                raise Exception("cannot convert type")
        self.add_type(inst, BuiltinType(Builtin.NOTHING), inst.location)

    def break_inst(self, inst: Break) -> None:
        self.add_type(inst, BuiltinType(Builtin.NOTHING), inst.location)

    def continue_inst(self, inst: Continue) -> None:
        self.add_type(inst, BuiltinType(Builtin.NOTHING), inst.location)

    def address_of(self, inst: AddressOf) -> None:
        tv = self.typevars[inst.value]
        if not tv.defined:
            return
        type = tv.get_one()
        if isinstance(type, StructType):
            raise Exception("unsupported operation")
        if isinstance(type, FuncType):
            raise Exception("func pointer type unsupported")
        self.add_type(inst, PointerType(type), inst.location)

    def get_item(self, inst: GetItem) -> None:
        value_tv = self.typevars[inst.value]
        index_tv = self.typevars[inst.index]
        if not value_tv.defined or not index_tv.defined:
            return
        value_type = value_tv.get_one()
        index_type = index_tv.get_one()
        if not isinstance(value_type, (PointerType, ArrayType)):
            raise Exception("unsupported get item")
        if not index_type.can_convert(BuiltinType(Builtin.INT)):
            raise Exception("index type is not converted to int")
        self.add_type(inst, value_type.type, inst.location)

    def var(self, inst: VarDecl) -> None:
        # Lock declared variable types on the first pass
        if not self.first:
            return
        if hasattr(inst, "type") and inst.type is not None:
            self.lock_type(inst, inst.type, inst.location)

    def set_item(self, inst: SetItem) -> None:
        var_tv = self.typevars[inst.var]
        index_tv = self.typevars[inst.index]
        value_tv = self.typevars[inst.value]
        if not var_tv.defined or not index_tv.defined or not value_tv.defined:
            return
        var_type = var_tv.get_one()
        index_type = index_tv.get_one()
        value_type = value_tv.get_one()
        if not isinstance(var_type, (PointerType, ArrayType)):
            raise Exception("unsupported set item")
        if not index_type.can_convert(BuiltinType(Builtin.INT)):
            raise Exception("index type is not converted to int")
        if not value_type.can_convert(var_type.type):
            raise Exception("cannot convert type")
        self.add_type(inst, BuiltinType(Builtin.NOTHING), inst.location)

    def set_attr(self, inst: SetAttr) -> None:
        var_tv = self.typevars[inst.var]
        value_tv = self.typevars[inst.value]
        if not var_tv.defined or not value_tv.defined:
            return
        t = var_tv.get_one()
        if not isinstance(t, Struct):
            raise Exception("set attr supported only for structs")
        if field := t.fields.get(inst.name):
            if not value_tv.get_one().can_convert(field.type):
                raise Exception("cannot convert type")
            # inst.field = field
            self.add_type(inst, BuiltinType(Builtin.NOTHING), inst.location)
        else:
            raise Exception("field not found")

    def deref(self, inst: Deref) -> None:
        value_tv = self.typevars[inst.value]
        if not value_tv.defined:
            return
        t = value_tv.get_one()
        if not isinstance(t, PointerType):
            raise Exception("deref supported only for pointers")
        self.add_type(inst, t.type, inst.location)

    def array(self, inst: Array) -> None:
        # Infer array element type from items
        if not inst.items:
            raise Exception("empty array literal")
        types = []
        for item in inst.items:
            tv = self.typevars[item]
            if not tv.defined:
                return
            types.append(tv.get_one())
        unified = self.unify_types(types)
        if unified is None:
            raise Exception("cannot unify array element types")
        self.add_type(inst, ArrayType(unified, len(inst.items)), inst.location)

    def get_attr(self, inst: GetAttr) -> None:
        value_tv = self.typevars[inst.value]
        if not value_tv.defined:
            return
        type = value_tv.get_one()
        if not isinstance(type, Struct):
            raise Exception("get attr supported only from structs")
        if field := type.fields.get(inst.name):
            self.add_type(inst, field.type, inst.location)
        else:
            raise Exception("field not found")
