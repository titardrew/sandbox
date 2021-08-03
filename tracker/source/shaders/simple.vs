#version 330 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec4 fragment_color;
uniform mat4 projection_matrix;
out vec4 frag_color;

void main()
{
    frag_color = fragment_color;
    gl_Position = projection_matrix * vec4(position.xy, 0.0f, 1.0f);
}
