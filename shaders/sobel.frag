#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform vec2 texelSize;   // from main code, e.g. (1.0/width, 1.0/height)
uniform float edgeScale;  // user‑adjustable scaling of the edge strength

void main()
{
    float gx[9] = float[](
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1
    );
    float gy[9] = float[](
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    );
    vec3 sample[9];
    int index = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            sample[index++] = texture(screenTexture, TexCoords + vec2(i, j) * texelSize).rgb;
        }
    }

    float sumX = 0.0;
    float sumY = 0.0;
    for (int i = 0; i < 9; i++) {
        float lum = dot(sample[i], vec3(0.2126, 0.7152, 0.0722));
        sumX += gx[i] * lum;
        sumY += gy[i] * lum;
    }
    float edge = length(vec2(sumX, sumY));
    edge *= edgeScale;  // scale the edge intensity

    FragColor = vec4(vec3(edge), 1.0);
}
