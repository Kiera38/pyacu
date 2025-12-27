import llvmlite.ir as llir

from acu.refanal.flow_graph_ir import (
    Array,
    BasicBlock,
    Binary,
    Branch,
    Call,
    Cast,
    Comparison,
    CreateStruct,
    Float,
    FuncIR,
    GetFieldPtr,
    GetItemPtr,
    GetPtr,
    Goto,
    Integer,
    LoadPtr,
    Op,
    OpVisitor,
    Register,
    Return,
    StorePtr,
    Unary,
    Unreachable,
    Value,
)
from acu.semanal.ir import BinaryOp, ComparisonOp, UnaryOp
from acu.semanal.types import (
    ArrayType,
    Builtin,
    BuiltinType,
    PointerType,
    Struct,
    Type,
)


class LLVMGenerator(OpVisitor[llir.Value]):
    def __init__(self, module: llir.Module) -> None:
        self.block_map: dict[BasicBlock, llir.Block] = {}
        self.value_map: dict[Value, llir.Value] = {}
        self.module = module
        self.builder = llir.IRBuilder()
        self.funcs: dict[FuncIR, llir.Function] = {}
        self.const_int_type = llir.IntType(32)

    def int_const(self, value: int) -> llir.Value:
        return self.const_int_type(value)

    def type(self, type: Type) -> llir.Type:
        if isinstance(type, BuiltinType):
            match type.type:
                case Builtin.BOOL:
                    return llir.IntType(1)
                case Builtin.FLOAT:
                    return llir.DoubleType()
                case Builtin.INT:
                    return llir.IntType(64)
                case Builtin.NOTHING:
                    return llir.VoidType()
        elif isinstance(type, PointerType):
            return llir.PointerType()
        elif isinstance(type, ArrayType):
            return llir.ArrayType(self.type(type.type), type.size)
        elif isinstance(type, Struct):
            return llir.LiteralStructType(
                [self.type(field[1].type) for field in type.field_list]
            )
        else:
            raise Exception("unknown type")

    def create(self, func: FuncIR):
        self.funcs[func] = llir.Function(
            self.module,
            llir.FunctionType(
                self.type(func.return_type), [self.type(arg.type) for arg in func.args]
            ),
            func.name,
        )

    def generate(self, func: FuncIR):
        self.block_map.clear()
        self.value_map.clear()
        ll_func = self.funcs[func]
        for block in func.blocks:
            self.block_map[block] = ll_func.append_basic_block()

        self.builder.position_at_end(ll_func.entry_basic_block)
        for arg, ll_arg in zip(func.args, ll_func.args):
            ptr = self.builder.alloca(ll_arg.type)
            self.builder.store(ll_arg, ptr)
            self.value_map[arg] = ptr

        for block, ll_block in self.block_map.items():
            self.builder.position_at_end(ll_block)
            for op in block.ops:
                value = op.accept(self)
                self.value_map[op] = value

    def block(self, block: BasicBlock) -> llir.Block:
        return self.block_map[block]

    def value(self, value: Value) -> llir.Value:
        if ll_value := self.value_map.get(value):
            if isinstance(value, Register):
                return self.builder.load(ll_value)
            return ll_value
        if isinstance(value, (Integer, Float)):
            ll_value = self.type(value.type)(value.value)
        elif isinstance(value, Register):
            ll_value = self.builder.alloca(self.type(value.type))
        else:
            raise Exception("unknown value")
        self.value_map[value] = ll_value
        if isinstance(value, Register):
            return self.builder.load(ll_value)
        return ll_value

    def get_ptr(self, op: GetPtr) -> llir.Value:
        ll_value = self.value_map.get(op.src)
        if ll_value is None:
            ll_value = self.builder.alloca(self.type(op.src.type))
            self.value_map[op.src] = ll_value
        return ll_value

    def load_ptr(self, op: LoadPtr) -> llir.Value:
        return self.builder.load(self.value(op.ptr), typ=self.type(op.type))

    def store_ptr(self, op: StorePtr) -> llir.Value:
        return self.builder.store(self.value(op.value), self.value(op.ptr))

    def get_field_ptr(self, op: GetFieldPtr) -> llir.Value:
        assert isinstance(op.ptr.type, PointerType)
        return self.builder.gep(
            self.value(op.ptr),
            [self.int_const(0), self.int_const(op.field)],
            source_etype=self.type(op.ptr.type.type),
        )

    def get_item_ptr(self, op: GetItemPtr) -> llir.Value:
        assert isinstance(op.ptr.type, PointerType)
        return self.builder.gep(
            self.value(op.ptr),
            [self.int_const(0), self.value(op.index)],
            source_etype=self.type(op.ptr.type.type),
        )

    def goto(self, op: Goto) -> llir.Value:
        return self.builder.branch(self.block(op.label))

    def branch(self, op: Branch) -> llir.Value:
        return self.builder.cbranch(
            self.value(op.value), self.block(op.true_label), self.block(op.false_label)
        )

    def visit_return(self, op: Return) -> llir.Value:
        if op.value.is_void:
            return self.builder.ret_void()
        return self.builder.ret(self.value(op.value))

    def unreachable(self, op: Unreachable) -> llir.Value:
        return self.builder.unreachable()

    def func(self, func: FuncIR) -> llir.Function:
        return self.funcs[func]

    def call(self, op: Call) -> llir.Value:
        return self.builder.call(self.func(op.fn), [self.value(arg) for arg in op.args])

    def cast(self, op: Cast) -> llir.Value:
        assert isinstance(op.type, BuiltinType)
        assert isinstance(op.obj.type, BuiltinType)
        match op.type.type:
            case Builtin.BOOL:
                match op.obj.type:
                    case Builtin.INT:
                        return self.builder.icmp_signed(
                            "!=", self.value(op.obj), self.int_const(0)
                        )
                    case Builtin.FLOAT:
                        return self.builder.fcmp_unordered(
                            "!=", self.value(op.obj), llir.DoubleType()(0)
                        )
                    case _:
                        raise Exception("unknown convertion type")
            case Builtin.FLOAT:
                match op.obj.type:
                    case Builtin.INT | Builtin.FLOAT:
                        return self.builder.sitofp(
                            self.value(op.obj), self.type(op.type)
                        )  # type: ignore
                    case _:
                        raise Exception("unknown conversion type")
            case Builtin.INT:
                match op.obj.type:
                    case Builtin.BOOL:
                        return self.builder.sext(self.value(op.obj), self.type(op.type))  # type: ignore
                    case Builtin.FLOAT:
                        return self.builder.fptosi(
                            self.value(op.obj), self.type(op.type)
                        )  # type: ignore
                    case _:
                        raise Exception("unknown conversion type")
            case _:
                raise Exception("Unknown conversion type")

    def binary(self, op: Binary) -> llir.Value:
        assert isinstance(op.left.type, BuiltinType)
        assert isinstance(op.right.type, BuiltinType)
        assert isinstance(op.type, BuiltinType)
        match op.type.type:
            case Builtin.INT:
                match op.op:
                    case BinaryOp.ADD:
                        return self.builder.add(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.SUB:
                        return self.builder.sub(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.MUL:
                        return self.builder.mul(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.DIV:
                        return self.builder.sdiv(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.MOD:
                        return self.builder.srem(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.LSHIFT:
                        return self.builder.shl(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.RSHIFT:
                        return self.builder.ashr(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.BIT_AND:
                        return self.builder.and_(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.BIT_OR:
                        return self.builder.or_(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.BIT_XOR:
                        return self.builder.xor(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
            case Builtin.FLOAT:
                match op.op:
                    case BinaryOp.ADD:
                        return self.builder.fadd(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.SUB:
                        return self.builder.fsub(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.MUL:
                        return self.builder.fmul(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.DIV:
                        return self.builder.fdiv(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
                    case BinaryOp.MOD:
                        return self.builder.frem(
                            self.value(op.left), self.value(op.right)
                        )  # type: ignore
        raise Exception("unknown binary op")

    def unary(self, op: Unary) -> llir.Value:
        assert isinstance(op.value.type, BuiltinType)
        assert isinstance(op.type, BuiltinType)
        match op.type.type:
            case Builtin.INT:
                match op.op:
                    case UnaryOp.NEG:
                        return self.builder.neg(self.value(op.value))  # type: ignore
                    case UnaryOp.BIT_NOT:
                        return self.builder.not_(self.value(op.value))  # type: ignore
            case Builtin.BOOL:
                match op.op:
                    case UnaryOp.NOT:
                        return self.builder.icmp_signed(
                            "==", self.value(op.value), self.type(op.type)(0)
                        )
            case Builtin.FLOAT:
                match op.op:
                    case UnaryOp.NEG:
                        return self.builder.fneg(self.value(op.value))  # type: ignore
        raise Exception("unknonw unary op")

    def comparison(self, op: Comparison) -> llir.Value:
        assert isinstance(op.left.type, BuiltinType)
        assert isinstance(op.right.type, BuiltinType)
        assert op.left.type == op.right.type
        cmpop = {
            ComparisonOp.EQUAL: "==",
            ComparisonOp.NOT_EQUAL: "!=",
            ComparisonOp.GREATER: ">",
            ComparisonOp.GREATER_EQUAL: ">=",
            ComparisonOp.LESS: "<",
            ComparisonOp.LESS_EQUAL: "<=",
        }[op.op]

        match op.left.type.type:
            case Builtin.BOOL | Builtin.INT:
                return self.builder.icmp_signed(
                    cmpop, self.value(op.left), self.value(op.right)
                )
            case Builtin.FLOAT:
                return self.builder.fcmp_unordered(
                    cmpop, self.value(op.left), self.value(op.right)
                )
        raise Exception("unknown comaprison op")

    def array(self, op: Array) -> llir.Value:
        ll_value = self.type(op.type)(llir.Undefined)
        for i, item in enumerate(op.items):
            ll_value = self.builder.insert_value(ll_value, self.value(item), i)
        return ll_value

    def create_struct(self, op: CreateStruct) -> llir.Value:
        ll_value = self.type(op.type)(llir.Undefined)
        for i, field in enumerate(op.fields):
            ll_value = self.builder.insert_value(ll_value, self.value(field), i)
        return ll_value

    def op(self, op: Op) -> llir.Value:
        raise Exception("unknown op")


def generate_llvm_ir(funcs: list[FuncIR]) -> llir.Module:
    module = llir.Module()
    generator = LLVMGenerator(module)
    for func in funcs:
        generator.create(func)
    for func in funcs:
        generator.generate(func)
    return module
