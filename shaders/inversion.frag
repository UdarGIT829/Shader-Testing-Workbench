#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;

void main()
{
    vec4 col = texture(screenTexture, TexCoords);
    FragColor = vec4(vec3(1.0 - col.rgb), 1.0);
}
