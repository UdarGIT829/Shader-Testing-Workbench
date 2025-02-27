#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

// The main image or video frame
uniform sampler2D screenTexture;

// A tileable 512x512 blue noise texture
uniform sampler2D blueNoiseTexture;

void main()
{
    // 1) Read the original color and compute its brightness (luminance).
    vec3 color = texture(screenTexture, TexCoords).rgb;
    float lum  = dot(color, vec3(0.2126, 0.7152, 0.0722));

    // 2) Sample the blue noise texture. We can simply sample at TexCoords,
    //    assuming the texture is set to wrap (GL_REPEAT), or use fract(...) 
    //    if you prefer manual wrapping. For example:
    vec2 noiseUV = fract(TexCoords); // ensures [0..1) even if TexCoords > 1
    float noiseVal = texture(blueNoiseTexture, noiseUV).r;

    // 3) Compare brightness vs. noise to decide black or white output.
    if (lum < noiseVal) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // black
    } else {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0); // white
    }
}
