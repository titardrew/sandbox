import gdb

int_ptr_t = gdb.lookup_type("int32_t").pointer()

class v4i_sse_printer:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        int_ptr = self.val.address.reinterpret_cast(int_ptr_t)
        return f"[{int_ptr[0]}, {int_ptr[1]}, {int_ptr[2]}, {int_ptr[3]}]"

def my_pp_func(val):
    if str(val.type) == '__m128i': return v4i_sse_printer(val)

gdb.pretty_printers.append(my_pp_func)
