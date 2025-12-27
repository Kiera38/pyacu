from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

from acu.source import Location


@dataclass
class Type:
    def can_convert(self, to: Type) -> bool:
        return self == to


class Builtin(Enum):
    NOTHING = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()


@dataclass
class BuiltinType(Type):
    type: Builtin

    def can_convert(self, to: Type) -> bool:
        if super().can_convert(to):
            return True
        if self.type == Builtin.NOTHING:
            return True
        if not isinstance(to, BuiltinType):
            return False
        if self.type == to.type:
            return True
        if to.type == Builtin.BOOL:
            return True
        if self.type == Builtin.INT:
            if to.type == Builtin.FLOAT:
                return True
        return False


@dataclass
class FuncType(Type):
    args: list[Type]
    return_type: Type


@dataclass
class ArrayType(Type):
    type: Type
    size: int


@dataclass
class PointerType(Type):
    type: Type


@dataclass
class StructField:
    type: Type
    index: int
    location: Location


@dataclass
class Struct(Type):
    name: str
    fields: dict[str, StructField]
    location: Location

    @cached_property
    def field_list(self):
        return sorted(self.fields.items(), key=lambda f: f[1].index)


@dataclass
class StructType(Type):
    struct: Struct
