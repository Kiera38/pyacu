from acu.refanal.flow_graph_ir import (
    Array,
    Binary,
    Branch,
    Call,
    Cast,
    Comparison,
    CreateStruct,
    Deref,
    FuncIR,
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
    Register,
    Return,
    SetField,
    SetItem,
    Specifier,
    Store,
    StorePtr,
    Unary,
    Unreachable,
    Value,
)
from acu.semanal.types import PointerType, Struct
from acu.source import Location


class RefToLoweringPass(OpVisitor[Value | None]):
    def __init__(self):
        self.replacements: dict[Value, Value] = {}
        self.current_ops: list[Op] = []
        self.temp_counter = 0

    def run(self, func: FuncIR) -> None:
        self.replacements.clear()
        self.temp_counter = 0

        # 1. Преобразуем аргументы
        # Все аргументы VAR/LET превращаем в указатели
        new_args = []
        for arg in func.args:
            arg = self.register(arg)
            new_args.append(arg)
        func.args = new_args

        # 2. Проход по блокам
        for block in func.blocks:
            self.current_ops = []
            for op in block.ops:
                # Результат визитора — это новое значение (регистр), которое заменяет старое op
                replacement = op.accept(self)
                if replacement is not None and op is not replacement:
                    self.replacements[op] = replacement
            block.ops = self.current_ops

    # --- Helpers: Ядро логики ---

    def get_mapped(self, val: Value) -> Value:
        return self.replacements.get(val, val)

    def register(self, reg: Register) -> Register:
        new_reg = self.get_mapped(reg)
        if new_reg is not reg:
            assert isinstance(new_reg, Register)
            return new_reg
        if reg.specifier in (Specifier.VAR, Specifier.LET):
            new_type = PointerType(reg.type)
            # Новый регистр всегда VAL, но имеет тип указателя
            new_reg = Register(
                reg.location, new_type, Specifier.VAL, name=reg.name, is_arg=reg.is_arg
            )
        else:
            new_reg = reg
        self.replacements[reg] = new_reg
        return new_reg

    def ensure_val(self, val: Value, location: Location) -> Value:
        """
        Гарантирует получение значения (r-value).
        Если исходный val был VAR/LET, значит сейчас это указатель -> делаем LoadPtr.
        """
        if isinstance(val, Register):
            mapped = self.register(val)
        else:
            mapped = self.get_mapped(val)

        # Если исходный операнд был ссылкой (VAR/LET), то mapped — это адрес.
        # Нам нужно значение, лежащее по этому адресу.
        if val.specifier in (Specifier.VAR, Specifier.LET):
            load = LoadPtr(location, mapped)
            self.current_ops.append(load)
            return load

        # Если это был VAL, то mapped — это уже значение.
        return mapped

    def ensure_ptr(self, val: Value, location: Location) -> Value:
        """
        Гарантирует получение адреса (l-value).
        Если исходный val был VAL (значение), создаем временный слот и сохраняем туда -> возвращаем адрес.
        """
        if isinstance(val, Register):
            mapped = self.register(val)
        else:
            mapped = self.get_mapped(val)

        # Если исходный операнд был VAL, то у нас нет адреса, есть только значение.
        # Чтобы получить адрес, нужно "материализовать" значение в памяти (spill).
        if val.specifier == Specifier.VAL:
            # Создаем временное значение (alloca)
            self.temp_counter += 1
            temp_reg = Register(
                location, val.type, Specifier.VAL, name=f".spill_{self.temp_counter}"
            )
            ptr = GetPtr(location, temp_reg)
            self.current_ops.append(ptr)
            # Сохраняем значение по адресу
            store = StorePtr(location, ptr, mapped)
            self.current_ops.append(store)
            return ptr

        # Если это был VAR/LET, то mapped — это уже адрес.
        return mapped

    # --- Visitor Implementation ---

    # 1. Store
    def store(self, op: Store) -> None:
        # Store(dest, src) -> *dest = src
        # dest должен быть адресом, src должен быть значением
        if isinstance(op.dest, Register):
            assert op.dest.specifier in (Specifier.VAR, Specifier.LET)
            new_reg = self.register(op.dest)
            dest = GetPtr(op.location, new_reg)
            self.current_ops.append(dest)
            src = self.ensure_ptr(op.src, op.location)
        else:
            dest = self.ensure_ptr(op.dest, op.location)
            src = self.ensure_val(op.src, op.location)
        self.current_ops.append(StorePtr(op.location, dest, src))
        return None

    # 2. GetField -> GetFieldPtr
    def get_field(self, op: GetField) -> Value:
        # GetField возвращал VAR (ссылку на поле).
        # Теперь мы возвращаем VAL указатель (адрес поля).
        # Для этого нам нужен адрес самой структуры.
        obj_ptr = self.ensure_ptr(op.obj, op.location)

        new_op = GetFieldPtr(op.location, obj_ptr, op.field)
        self.current_ops.append(new_op)
        return new_op

    # 3. SetField
    def set_field(self, op: SetField) -> None:
        # obj.field = value
        # 1. Получаем адрес объекта
        obj_ptr = self.ensure_ptr(op.obj, op.location)
        # 2. Вычисляем адрес поля
        field_ptr = GetFieldPtr(op.location, obj_ptr, op.field)
        self.current_ops.append(field_ptr)
        # 3. Получаем значение для записи
        val = self.ensure_val(op.value, op.location)
        # 4. Записываем
        self.current_ops.append(StorePtr(op.location, field_ptr, val))
        return None

    # 4. GetItem -> GetItemPtr
    def get_item(self, op: GetItem) -> Value:
        # Аналогично GetField, нужен адрес массива/указателя
        base_ptr = self.ensure_ptr(op.obj, op.location)
        index_val = self.ensure_val(op.index, op.location)

        new_op = GetItemPtr(op.location, base_ptr, index_val)
        self.current_ops.append(new_op)
        return new_op

    # 5. SetItem
    def set_item(self, op: SetItem) -> None:
        base_ptr = self.ensure_ptr(op.obj, op.location)
        index_val = self.ensure_val(op.index, op.location)
        val = self.ensure_val(op.value, op.location)

        item_ptr = GetItemPtr(op.location, base_ptr, index_val)
        self.current_ops.append(item_ptr)

        self.current_ops.append(StorePtr(op.location, item_ptr, val))
        return None

    # 6. GetAddress
    def get_address(self, op: GetAddress) -> Value:
        # GetAddress(x) возвращает указатель на x.
        # В новом IR нам просто нужен адрес x.
        # Функция ensure_ptr делает именно это:
        #   - если x был VAR, возвращает его (который уже указатель).
        #   - если x был VAL (например, literal 5), создает слот, сохраняет 5 и возвращает адрес слота.
        return self.ensure_ptr(op.src, op.location)

    # 7. Deref
    def deref(self, op: Deref) -> Value:
        # Deref(ptr) превращает VAL T* -> VAR T.
        # В новом IR VAR T представляется как VAL T*.
        # То есть Deref ничего не делает с данными, просто пробрасывает указатель дальше.
        # Единственное требование: операнд должен быть значением (самим указателем).
        return self.ensure_val(op.ptr, op.location)

    # 8. Операции над значениями (Binary, Unary, Cast, Comparison)
    def binary(self, op: Binary) -> Value:
        left = self.ensure_val(op.left, op.location)
        right = self.ensure_val(op.right, op.location)
        new_op = Binary(op.location, left, right, op.op)
        self.current_ops.append(new_op)
        return new_op

    def unary(self, op: Unary) -> Value:
        val = self.ensure_val(op.value, op.location)
        new_op = Unary(op.location, val, op.op)
        self.current_ops.append(new_op)
        return new_op

    def comparison(self, op: Comparison) -> Value:
        left = self.ensure_val(op.left, op.location)
        right = self.ensure_val(op.right, op.location)
        new_op = Comparison(op.location, left, right, op.op)
        self.current_ops.append(new_op)
        return new_op

    def cast(self, op: Cast) -> Value:
        val = self.ensure_val(op.obj, op.location)
        new_op = Cast(op.location, op.type, val)
        self.current_ops.append(new_op)
        return new_op

    def array(self, op: Array) -> Value:
        new_op = Array(
            op.location, [self.ensure_val(item, op.location) for item in op.items]
        )
        self.current_ops.append(new_op)
        return new_op

    def create_struct(self, op: CreateStruct) -> Value:
        assert isinstance(op.type, Struct)
        new_op = CreateStruct(
            op.location,
            [self.ensure_val(field, op.location) for field in op.fields],
            op.type,
        )
        self.current_ops.append(new_op)
        return new_op

    # 9. Call
    def call(self, op: Call) -> Value:
        new_args = []
        # Важно: здесь мы не можем слепо верить specifier аргумента,
        # нам нужно знать сигнатуру вызываемой функции, чтобы понять,
        # ожидает ли она значение или указатель (бывшую ссылку).
        # Но, предполагая, что func.args уже обновлены (стали Ptr),
        # мы можем просто смотреть на тип ожидаемого аргумента?

        # Однако, самый надежный способ, совместимый с вашей задачей:
        # Если аргумент в вызываемой функции объявлен как VAR (теперь Ptr), мы передаем Ptr.
        # Если как VAL, мы передаем Val.

        for arg in op.args:
            # сейчас для упрощения все параметры функции let
            new_args.append(self.ensure_ptr(arg, op.location))

        new_op = Call(op.location, op.fn, new_args)
        self.current_ops.append(new_op)
        return new_op

    # 10. Control Ops
    def branch(self, op: Branch) -> None:
        cond = self.ensure_val(op.value, op.location)
        self.current_ops.append(
            Branch(op.location, cond, op.true_label, op.false_label)
        )
        return None

    def visit_return(self, op: Return) -> None:
        val = self.ensure_val(op.value, op.location)
        self.current_ops.append(Return(op.location, val))
        return None

    def goto(self, op: Goto) -> None:
        self.current_ops.append(op)
        return None

    def unreachable(self, op: Unreachable) -> None:
        self.current_ops.append(op)
        return None

    def op(self, op: Op) -> Value:
        # Fallback
        self.current_ops.append(op)
        return op


def lower_refs(func: FuncIR):
    ref_to_ptr = RefToLoweringPass()
    ref_to_ptr.run(func)
