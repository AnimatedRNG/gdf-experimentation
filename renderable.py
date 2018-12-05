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
    "dot"
]


def float_str(v):
    if isinstance(v, (list, tuple)):
        return ", ".join([float_str(a) for a in v])
    else:
        return "{:.8f}".format(v)


class Function():

    def __init__(self, op, *arguments):
        if op in OPERATIONS:
            self.op = op
            self.arguments = list(arguments)
        else:
            self.op = 'const'
            self.arguments = list(arguments)
            self.arguments.insert(0, op)
        self.shader = self.generate_shader()
        self.model = self.generate_model()

    def generate_shader(self):
        if self.op == 'add':
            return "({} + {})".format(self.arguments[0].generate_shader(), self.arguments[1].generate_shader())
        elif self.op == 'sub':
            return "({} - {})".format(self.arguments[0].generate_shader(), self.arguments[1].generate_shader())
        elif self.op == 'mul':
            return "({} * {})".format(self.arguments[0].generate_shader(), self.arguments[1].generate_shader())
        elif self.op == 'div':
            return "({} / {})".format(self.arguments[0].generate_shader(), self.arguments[1].generate_shader())
        elif self.op == 'pow':
            return "pow({}, {})".format(self.arguments[0].generate_shader(), self.arguments[1].generate_shader())
        elif self.op == 'min':
            return "min({}, {})".format(self.arguments[0].generate_shader(), self.arguments[1].generate_shader())
        elif self.op == 'max':
            return "max({}, {})".format(self.arguments[0].generate_shader(), self.arguments[1].generate_shader())
        elif self.op == 'dot':
            return "dot({}, {})".format(self.arguments[0].generate_shader(), self.arguments[1].generate_shader())
        elif self.op == 'const':
            if isinstance(self.arguments[0], (list, tuple)):
                argc = len(self.arguments[0])
                assert(argc <= 4)
                if argc == 1:
                    return "{}".format(float_str(self.arguments[0][0]))
                else:
                    return "vec{}({})".format(argc,
                                              float_str(tuple(self.arguments[0])))
            else:
                return "{}".format(float_str(self.arguments[0]))

    def generate_model(self):
        if self.op == 'add':
            return self.arguments[0].generate_model() + self.arguments[1].generate_model()
        elif self.op == 'sub':
            return self.arguments[0].generate_model() - self.arguments[1].generate_model()
        elif self.op == 'mul':
            return self.arguments[0].generate_model() * self.arguments[1].generate_model()
        elif self.op == 'div':
            return self.arguments[0].generate_model() / self.arguments[1].generate_model()
        elif self.op == 'pow':
            return torch.pow(self.arguments[0].generate_model(), self.arguments[1].generate_model())
        elif self.op == 'min':
            return torch.min(self.arguments[0].generate_model(), self.arguments[1].generate_model())
        elif self.op == 'max':
            return torch.max(self.arguments[0].generate_model(), self.arguments[1].generate_model())
        elif self.op == 'dot':
            return torch.dot(self.arguments[0].generate_model(), self.arguments[1].generate_model())
        elif self.op == 'const':
            if isinstance(self.arguments[0], (list, tuple)):
                return torch.tensor(
                    self.arguments[0], requires_grad='requires_grad' in self.arguments)
            else:
                return torch.tensor([self.arguments[0]], requires_grad='requires_grad' in self.arguments)

    def update(self):
        if self.op == 'const':
            if isinstance(self.arguments[0], (list, tuple)):
                self.arguments[0] = self.model.clone(
                ).detach().numpy().tolist()
            else:
                self.arguments[0] = self.model.clone().detach().item()
        else:
            [arg.update() for arg in self.arguments]

    def __add__(self, other_func):
        return Function('add', self, other_func)

    def __sub__(self, other_func):
        return Function('sub', self, other_func)

    def __mul__(self, other_func):
        return Function('mul', self, other_func)

    def __div__(self, other_func):
        return Function('div', self, other_func)

    def pow(self, other_func):
        return Function('pow', self, other_func)

    def min(self, other_func):
        return Function('min', self, other_func)

    def max(self, other_func):
        return Function('max', self, other_func)

    def dot(self, other_func):
        return Function('dot', self, other_func)


def main():
    a = Function([3, 5])
    b = Function([4, 6])
    print((a + b).generate_shader())
    print((a + b).generate_model())
    c = a + b
    d = Function([5, 7])
    print(c.pow(d).generate_shader())

    b.model = torch.tensor([0, 0])
    b.update()
    print(c.generate_model())
    print(c.generate_shader())


if __name__ == '__main__':
    main()
