#!/usr/bin/env python3

import moderngl
import numpy as np

from example_window import Example, run_example

vs = '''
# version 430 core
layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.);
}
'''

fs = '''
# version 430 core
out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 0.0, 0.0);
}
'''


class SDFVisualizer(Example):
    WINDOW_SIZE = (640, 640)

    def __init__(self):
        self.ctx = moderngl.create_context()

        width, height = self.wnd.size
        canvas = np.array(
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).astype('f4')

        self.prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

        self.vbo = self.ctx.buffer(canvas.tobytes())
        self.vao = self.ctx.simple_vertex_array(
            self.prog, self.vbo, 'position')

    def render(self):
        self.ctx.viewport = self.wnd.viewport
        self.ctx.clear(0.0, 0.0, 0.0)

        self.vao.render(moderngl.TRIANGLE_STRIP)


if __name__ == '__main__':
    run_example(SDFVisualizer)
