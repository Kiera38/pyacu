"""
Test to verify that all reference-to-pointer conversion rules are properly implemented.

The rules are:
- All LET/VAR specifiers become VAL
- All types for references become pointers to original types
- All Store operations with LoadPtr destination become StorePtr
- All Store operations with Register destination have source converted to pointer
- All SetItem/SetField operations become GetItemPtr/GetFieldPtr + StorePtr
- All GetItem/GetField operations become GetItemPtr/GetFieldPtr (result is pointer)
- Convert values to pointer by unwrapping LoadPtr
- Convert pointers to values by wrapping in LoadPtr
"""

from acu.refanal.flow_graph_ir import (
    BasicBlock,
    Builtin,
    BuiltinType,
    Deref,
    FuncIR,
    GetField,
    GetFieldPtr,
    GetItem,
    GetItemPtr,
    GetPtr,
    Integer,
    Op,
    Register,
    SetField,
    SetItem,
    Specifier,
    Store,
    StorePtr,
)
from acu.refanal.lower_refs import lower_refs
from acu.semanal.types import ArrayType, PointerType, Struct, StructField
from acu.source import Location


def make_simple_func(args: list[Register], ops: list[Op]):
    return FuncIR("test", args, [BasicBlock(ops=ops)])


def test_specifiers_become_val():
    """Test Rule 1: LET/VAR specifiers become VAL"""
    loc = Location(0, 0, 0, 0)

    # Create registers with different specifiers
    var_reg = Register(name="var_reg", location=loc, type=BuiltinType(Builtin.INT))
    var_reg.specifier = Specifier.VAR
    let_reg = Register(name="let_reg", location=loc, type=BuiltinType(Builtin.INT))
    let_reg.specifier = Specifier.LET
    val_reg = Register(name="val_reg", location=loc, type=BuiltinType(Builtin.INT))
    val_reg.specifier = Specifier.VAL

    ptr_var_reg = Register(
        name="ptr_var_reg", location=loc, type=PointerType(BuiltinType(Builtin.INT))
    )
    ptr_var_reg.specifier = Specifier.VAR
    ptr_val_reg = Register(
        name="ptr_val_reg", location=loc, type=PointerType(BuiltinType(Builtin.INT))
    )
    ptr_val_reg.specifier = Specifier.VAL

    func = make_simple_func([var_reg, let_reg, val_reg, ptr_var_reg, ptr_val_reg], [])

    lower_refs(func)
    var_reg, let_reg, val_reg, ptr_var_reg, ptr_val_reg = func.args

    assert var_reg.specifier == Specifier.VAL, (
        f"VAR should become VAL, got {var_reg.specifier}"
    )
    assert let_reg.specifier == Specifier.VAL, (
        f"LET should become VAL, got {let_reg.specifier}"
    )
    assert val_reg.specifier == Specifier.VAL, (
        f"VAL should remain VAL, got {val_reg.specifier}"
    )
    assert ptr_val_reg.specifier == Specifier.VAL, (
        f"Ptr VAL should remain VAL, got {val_reg.specifier}"
    )
    assert ptr_var_reg.specifier == Specifier.VAL, (
        f"Ptr VAR should remain VAL, got {val_reg.specifier}"
    )
    assert var_reg.type == PointerType(BuiltinType(Builtin.INT))
    assert let_reg.type == PointerType(BuiltinType(Builtin.INT))
    assert val_reg.type == BuiltinType(Builtin.INT)
    assert ptr_var_reg.type == PointerType(PointerType(BuiltinType(Builtin.INT)))
    assert ptr_val_reg.type == PointerType(BuiltinType(Builtin.INT))


def test_store_with_loadptr():
    """Test Rule 2: Store with LoadPtr destination becomes StorePtr"""
    loc = Location(0, 0, 0, 0)

    # Create registers for the test
    obj_reg = Register(
        name="obj", location=loc, type=PointerType(BuiltinType(Builtin.INT))
    )
    obj_reg.specifier = Specifier.VAL
    int_val = Integer(value=42, location=loc)
    int_val.specifier = Specifier.VAL

    # deref operation
    load_ptr = Deref(ptr=obj_reg, location=loc)
    load_ptr.specifier = Specifier.VAL

    # Store operation with deref destination
    store_op = Store(dest=load_ptr, src=int_val, location=loc)
    store_op.specifier = Specifier.VAL

    func = make_simple_func([obj_reg], [load_ptr, store_op])

    lower_refs(func)

    block = func.blocks[0]

    assert len(block.ops) == 2, f"Expected 2 op in block, got {len(block.ops)}"
    assert isinstance(block.ops[0], GetPtr)
    assert isinstance(block.ops[1], StorePtr), (
        f"Expected StorePtr, got {type(block.ops[1])}"
    )
    assert block.ops[0].specifier == Specifier.VAL, "GetPtr should have VAL specifier"
    assert block.ops[1].ptr == block.ops[0]


def test_getfield():
    """Test Rule 3: GetField becomes GetFieldPtr"""
    loc = Location(0, 0, 0, 0)

    obj_reg = Register(
        name="obj3",
        location=loc,
        type=Struct(
            "test",
            {"field": StructField(BuiltinType(Builtin.INT), 0, loc)},
            loc,
        ),
    )
    obj_reg.specifier = Specifier.VAL

    get_field_op = GetField(obj=obj_reg, field=0, location=loc)
    get_field_op.specifier = Specifier.VAR

    func = make_simple_func([obj_reg], [get_field_op])

    lower_refs(func)

    block = func.blocks[0]
    obj_reg = func.args[0]

    assert isinstance(block.ops[0], GetPtr)
    assert block.ops[0].src == obj_reg
    assert isinstance(block.ops[1], GetFieldPtr), (
        f"Expected GetFieldPtr, got {type(block.ops[1])}"
    )
    assert block.ops[1].specifier == Specifier.VAL, (
        "GetFieldPtr result should be VAL (pointer)"
    )
    assert block.ops[1].type == PointerType(BuiltinType(Builtin.INT)), (
        "GetFieldPtr type должен быть Ptr[Int]"
    )
    assert block.ops[1].ptr == block.ops[0]


def test_getitem():
    """Test Rule 4: GetItem becomes GetItemPtr"""
    loc = Location(0, 0, 0, 0)

    obj_reg = Register(
        name="obj4", location=loc, type=ArrayType(BuiltinType(Builtin.INT), 1)
    )
    obj_reg.specifier = Specifier.VAL
    index_reg = Register(name="index", location=loc, type=BuiltinType(Builtin.INT))
    index_reg.specifier = Specifier.VAL

    get_item_op = GetItem(obj=obj_reg, index=index_reg, location=loc)
    get_item_op.specifier = Specifier.VAR

    func = make_simple_func([obj_reg, index_reg], [get_item_op])

    lower_refs(func)

    block = func.blocks[0]
    obj_reg = func.args[0]

    assert isinstance(block.ops[0], GetPtr)
    assert block.ops[0].src == obj_reg
    assert isinstance(block.ops[1], GetItemPtr), (
        f"Expected GetItemPtr, got {type(block.ops[1])}"
    )
    assert block.ops[1].specifier == Specifier.VAL, (
        "GetItemPtr result should be VAL (pointer)"
    )
    assert block.ops[1].type == PointerType(BuiltinType(Builtin.INT)), (
        "тип GetItemPtr должен быть Ptr[Int]"
    )
    assert block.ops[1].ptr == block.ops[0]


def test_setfield():
    """Test Rule 5: SetField becomes StorePtr with GetFieldPtr"""
    loc = Location(0, 0, 0, 0)

    obj_reg = Register(
        name="obj5",
        location=loc,
        type=Struct(
            "test",
            {"field": StructField(BuiltinType(Builtin.INT), 0, loc)},
            loc,
        ),
    )
    obj_reg.specifier = Specifier.VAL
    value_reg = Register(name="value", location=loc, type=BuiltinType(Builtin.INT))
    value_reg.specifier = Specifier.VAL

    set_field_op = SetField(
        obj=obj_reg,
        field=0,
        value=value_reg,
        location=loc,
    )
    set_field_op.specifier = Specifier.VAL

    func = make_simple_func([obj_reg, value_reg], [set_field_op])

    lower_refs(func)

    block = func.blocks[0]
    obj_reg, value_reg = func.args
    assert isinstance(block.ops[0], GetPtr)
    assert block.ops[0].src == obj_reg
    assert isinstance(block.ops[1], GetFieldPtr)
    result_op = block.ops[2]

    assert isinstance(result_op, StorePtr), f"Expected StorePtr, got {type(result_op)}"
    assert result_op.ptr == block.ops[1]


def test_setitem():
    """Test Rule 6: SetItem becomes StorePtr with GetItemPtr"""
    loc = Location(0, 0, 0, 0)

    obj_reg = Register(
        name="obj6", location=loc, type=ArrayType(BuiltinType(Builtin.INT), 1)
    )
    obj_reg.specifier = Specifier.VAL
    index_reg6 = Register(name="index6", location=loc, type=BuiltinType(Builtin.INT))
    index_reg6.specifier = Specifier.VAL
    value_reg6 = Register(name="value6", location=loc, type=BuiltinType(Builtin.INT))
    value_reg6.specifier = Specifier.VAL

    set_item_op = SetItem(
        obj=obj_reg,
        index=index_reg6,
        value=value_reg6,
        location=loc,
    )
    set_item_op.specifier = Specifier.VAL

    func = make_simple_func([obj_reg, index_reg6, value_reg6], [set_item_op])

    lower_refs(func)
    block = func.blocks[0]
    obj_reg, index_reg6, value_reg6 = func.args
    assert isinstance(block.ops[0], GetPtr)
    assert block.ops[0].src == obj_reg
    assert isinstance(block.ops[1], GetItemPtr)
    result_op = block.ops[2]

    assert isinstance(result_op, StorePtr), f"Expected StorePtr, got {type(result_op)}"
    assert result_op.ptr == block.ops[1]


def test_getfield_ref():
    """Test Rule 3: GetField becomes GetFieldPtr"""
    loc = Location(0, 0, 0, 0)
    type = Struct(
        "test",
        {"field": StructField(BuiltinType(Builtin.INT), 0, loc)},
        loc,
    )
    obj_reg = Register(name="obj3", location=loc, type=type)
    obj_reg.specifier = Specifier.VAR

    get_field_op = GetField(obj=obj_reg, field=0, location=loc)
    get_field_op.specifier = Specifier.VAR

    func = make_simple_func([obj_reg], [get_field_op])

    lower_refs(func)

    block = func.blocks[0]
    obj_reg = func.args[0]

    assert isinstance(block.ops[0], GetFieldPtr)
    assert block.ops[0].specifier == Specifier.VAL, (
        "GetFieldPtr result should be VAL (pointer)"
    )
    assert block.ops[0].type == PointerType(BuiltinType(Builtin.INT)), (
        "GetFieldPtr type должен быть Ptr[Int]"
    )
    assert block.ops[0].ptr == obj_reg

    assert obj_reg.type == PointerType(type)
    assert obj_reg.specifier == Specifier.VAL


def test_getitem_ref():
    """Test Rule 4: GetItem becomes GetItemPtr"""
    loc = Location(0, 0, 0, 0)
    type = ArrayType(BuiltinType(Builtin.INT), 1)
    obj_reg = Register(name="obj4", location=loc, type=type)
    obj_reg.specifier = Specifier.VAR
    index_reg = Register(name="index", location=loc, type=BuiltinType(Builtin.INT))
    index_reg.specifier = Specifier.VAL

    get_item_op = GetItem(obj=obj_reg, index=index_reg, location=loc)
    get_item_op.specifier = Specifier.VAR

    func = make_simple_func([obj_reg, index_reg], [get_item_op])

    lower_refs(func)

    block = func.blocks[0]
    obj_reg, index_reg = func.args

    assert isinstance(block.ops[0], GetItemPtr)
    assert block.ops[0].specifier == Specifier.VAL, (
        "GetItemPtr result should be VAL (pointer)"
    )
    assert block.ops[0].type == PointerType(BuiltinType(Builtin.INT)), (
        "тип GetItemPtr должен быть Ptr[Int]"
    )
    assert block.ops[0].ptr == obj_reg
    assert obj_reg.type == PointerType(type)
    assert obj_reg.specifier == Specifier.VAL
