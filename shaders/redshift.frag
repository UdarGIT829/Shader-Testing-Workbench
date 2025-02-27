#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;

// user‑adjustable parameters:
uniform float redFactor;
uniform float greenFactor;
uniform float blueFactor;

void main()
{
    vec4 col = texture(screenTexture, TexCoords);
    col.r *= redFactor;    // boost or reduce red
    col.g *= greenFactor;  // tweak green
    col.b *= blueFactor;   // tweak blue
    FragColor = col;
}
