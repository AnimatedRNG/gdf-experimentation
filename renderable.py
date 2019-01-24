#!/usr/bin/env python3

import torch

OPERATIONS = [
    'add',
    'sub',
    'mul',
    'div',
    "pow",
    "min",
    "max",
    "dot",
    "abs"
]


def num_or_none(s):
    try:
        if isinstance(s, (list, tuple)):
            return s
        else:
            return float(s)
    except ValueError:
        return None


def float_str(v):
    if isinstance(v, (list, tuple)):
        return ", ".join([float_str(a) for a in v])
    else:
        return "{}".format(float(v))


def extract_type(length):
    if length == 1:
        return "float"
    elif length <= 4:
        return "vec{}".format(length)


class Function():

    device = torch.cuda.current_device()
    #device = torch.device('cpu')

    def __init__(self, op, *arguments):
        if op in OPERATIONS:
            self.op = op
            self.arguments = list(arguments)

            if self.op in ('add', 'sub', 'mul', 'div', 'pow', 'min', 'max', 'abs'):
                self.ret = max(arg.ret for arg in self.arguments)
            elif self.op in ('dot'):
                self.ret = 1
        else:
            n = num_or_none(op)
            if n is not None:
                self.op = 'const'
                if isinstance(op, (tuple, list)):
                    self.ret = len(op)
                else:
                    self.ret = 1
            else:
                self.op = 'var'
                self.ret = arguments[0]
            self.arguments = list(arguments)
            self.arguments.insert(0, op)
        self.shader = self.generate_shader()
        self.model = self.generate_model()

    def get_free_variables(self):
        if self.op == "var":
            return [self]
        elif self.op != "const":
            vs = []
            for arg in self.arguments:
                seen = set()
                for arg in arg.get_free_variables():
                    if arg.arguments[0] not in seen:
                        vs.append(arg)
                        seen.add(arg.arguments[0])
            return vs
        else:
            return []

    def generate_shader(self):
        args = ["{} {}".format(extract_type(
            var.ret), var.arguments[0]) for var in self.get_free_variables()]
        return "{return_type} sdf({arguments}) {{\n return {body}; \n }}"\
            .format(return_type=extract_type(self.ret),
                    arguments=", ".join(args),
                    body=self.function_body())

    def function_body(self):
        if self.op == 'add':
            return "({} + {})".format(self.arguments[0].function_body(), self.arguments[1].function_body())
        elif self.op == 'sub':
            return "({} - {})".format(self.arguments[0].function_body(), self.arguments[1].function_body())
        elif self.op == 'mul':
            return "({} * {})".format(self.arguments[0].function_body(), self.arguments[1].function_body())
        elif self.op == 'div':
            return "({} / {})".format(self.arguments[0].function_body(), self.arguments[1].function_body())
        elif self.op == 'pow':
            return "pow({}, {})".format(self.arguments[0].function_body(), self.arguments[1].function_body())
        elif self.op == 'min':
            return "min({}, {})".format(self.arguments[0].function_body(), self.arguments[1].function_body())
        elif self.op == 'max':
            return "max({}, {})".format(self.arguments[0].function_body(), self.arguments[1].function_body())
        elif self.op == 'dot':
            return "dot({}, {})".format(self.arguments[0].function_body(), self.arguments[1].function_body())
        elif self.op == 'abs':
            return "abs({})".format(self.arguments[0].function_body())
        elif self.op == 'const':
            if isinstance(self.arguments[0], (list, tuple)):
                argc = len(self.arguments[0])
                assert(argc <= 4)
                if argc == 1:
                    return "{}".format(float_str(self.arguments[0][0]))
                else:
                    return "{}({})".format(extract_type(argc),
                                           float_str(tuple(self.arguments[0])))
            else:
                return "{}".format(float_str(self.arguments[0]))
        elif self.op == 'var':
            return self.arguments[0]

    def generate_model(self, params={}):
        if self.op == 'add':
            return self.arguments[0].generate_model(params) + self.arguments[1].generate_model(params)
        elif self.op == 'sub':
            return self.arguments[0].generate_model(params) - self.arguments[1].generate_model(params)
        elif self.op == 'mul':
            return self.arguments[0].generate_model(params) * self.arguments[1].generate_model(params)
        elif self.op == 'div':
            return self.arguments[0].generate_model(params) / self.arguments[1].generate_model(params)
        elif self.op == 'pow':
            # print("pow of {}, {}".format(self.arguments[0].generate_model(
            #    params), self.arguments[1].generate_model(params)))
            return torch.pow(self.arguments[0].generate_model(params), self.arguments[1].generate_model(params))
        elif self.op == 'min':
            return torch.min(self.arguments[0].generate_model(params), self.arguments[1].generate_model(params))
        elif self.op == 'max':
            return torch.max(self.arguments[0].generate_model(params), self.arguments[1].generate_model(params))
        # matrix multiplication cases
        elif self.op == 'dot':
            m0 = self.arguments[0].generate_model(params)
            m1 = self.arguments[1].generate_model(params)
            if len(m0.shape) == 2 and len(m1.shape) == 1:
                b = m0.shape[0]
                return torch.bmm(m0.expand(b, -1, 1), m1.view(b, 1, -1))
            elif len(m0.shape) == 1 and len(m1.shape) == 2:
                b = m1.shape[0]
                return torch.bmm(m0.expand(b, 1, -1), m1.view(b, -1, 1))
            elif len(m0.shape) == 1 and len(m1.shape) == 1:
                return torch.dot(m0, m1)
            else:
                assert(False)
        elif self.op == 'abs':
            return torch.abs(self.arguments[0].generate_model(params))
        elif self.op == 'const':
            try:
                return self.model
            except AttributeError:
                if isinstance(self.arguments[0], (list, tuple)):
                    return torch.tensor(
                        self.arguments[0], requires_grad='requires_grad' in self.arguments, dtype=torch.float32,
                        device=Function.device)
                else:
                    return torch.tensor([self.arguments[0]], requires_grad='requires_grad' in self.arguments, dtype=torch.float32,
                                        device=Function.device)
        elif self.op == 'var':
            if self.arguments[0] in params:
                return params[self.arguments[0]]
            else:
                return torch.tensor([0.0 for _ in range(self.ret)],
                                    device=Function.device)

    def update(self):
        if self.op == 'const':
            if isinstance(self.arguments[0], (list, tuple)):
                self.arguments[0] = self.model.clone(
                ).detach().cpu().numpy().tolist()
            else:
                self.arguments[0] = self.model.clone().detach().cpu().item()
        elif self.op != 'var':
            [arg.update() for arg in self.arguments]

    def __add__(self, other_func):
        return Function('add', self, other_func)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Function('add', Function(other), self)
        else:
            return Function('add', other, self)

    def __sub__(self, other_func):
        return Function('sub', self, other_func)

    def __mul__(self, other_func):
        return Function('mul', self, other_func)

    def __truediv__(self, other_func):
        return Function('div', self, other_func)

    def pow(self, other_func):
        return Function('pow', self, other_func)

    def min(self, other_func):
        return Function('min', self, other_func)

    def max(self, other_func):
        return Function('max', self, other_func)

    def dot(self, other_func):
        return Function('dot', self, other_func)

    def abs(self):
        return Function('abs', self)


def main():
    a = Function([3.0, 5.0])
    b = Function([4.0, 6.0])
    print((a + b).generate_shader())
    print((a + b).generate_model())
    c = a + b
    d = Function([5.0, 7.0])
    print(c.pow(d).generate_shader())

    b.model = torch.tensor([0.0, 0.0])
    b.update()
    print(c.generate_model())
    print(c.generate_shader())

    print(Function("a", 3).generate_shader())

    from model import gdf
    g = gdf([Function((0.577, 0.577, 0.577)),
             Function((0.577, 0.577, 0.577))],
            Function(9.0, 'requires_grad'))
    at_pos = g(Function((0.0, 0.0, 0.0)))
    print(at_pos.generate_shader())


if __name__ == '__main__':
    main()
