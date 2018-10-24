#!/usr/bin/env python3

import pyglet
from pyglet.gl import *

from shader import Shader

vs = '''
#version 430 core
layout(location = 0) in vec3 position;

void main() {
    gl_Position = vec4(position, 1.);
}
'''

fs = '''
#version 430 core
out vec3 color;

void main() {
    color = vec3(1.0, 0.0, 0.0);
}
'''

# create the window, but keep it offscreen until we are done with setup
window = pyglet.window.Window(
    640, 480, resizable=True, visible=False, caption="SDF Visualizer")

window.recompile = True

# create a fullscreen quad
batch = pyglet.graphics.Batch()
batch.add(4, GL_QUADS, None, ('v2i', (0, 0, 1, 0, 1, 1, 0, 1)),
          ('t2f', (0, 0, 1.0, 0, 1.0, 1.0, 0, 1.0)))


def update_gdf(func_string):
    recompile = True


@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)
    return pyglet.event.EVENT_HANDLED


@window.event
def on_draw():
    # clear the screen
    window.clear()

    if window.recompile:
        shader = Shader([vs.encode('ascii'), fs.encode('ascii')])
        recompile = False

    shader.bind()
    batch.draw()
    shader.unbind()


def run():
    window.set_visible(True)
    pyglet.app.run()
