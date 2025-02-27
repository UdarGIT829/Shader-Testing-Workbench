#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform float threshold;   // e.g. 0.5
uniform bool useGreater;   // if true => keep if brightness > threshold, else keep if brightness < threshold
uniform float setValue;    // brightness to set if the threshold check fails

void main()
{
    vec3 orig = texture(screenTexture, TexCoords).rgb;
    // Calculate perceived luminance (a common formula)
    float lum = dot(orig, vec3(0.2126, 0.7152, 0.0722));

    // Decide whether we pass the threshold
    bool pass = (useGreater) ? (lum > threshold) : (lum < threshold);

    // If not passing, overwrite the color with a uniform brightness
    // (interpreted as grayscale: (setValue, setValue, setValue))
    if (!pass) {
        orig = vec3(setValue);
    }

    FragColor = vec4(orig, 1.0);
}
