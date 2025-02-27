#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform float hueShiftDeg;   // How many degrees to shift hue (0..360)
uniform float satScale;      // How much to scale saturation (0..2, for example)
uniform float valScale;      // How much to scale value (0..2, for example)

// Helper: convert RGB [0..1] to HSV with H in [0..1], S,V in [0..1]
vec3 rgbToHsv(vec3 c) {
    float maxC = max(c.r, max(c.g, c.b));
    float minC = min(c.r, min(c.g, c.b));
    float delta = maxC - minC;
    
    float h = 0.0;
    float s = (maxC == 0.0) ? 0.0 : (delta / maxC);
    float v = maxC;
    
    if (delta < 1e-6) {
        // No color difference => hue is undefined, keep as 0
        h = 0.0;
    } else {
        if (maxC == c.r) {
            h = (c.g - c.b) / delta;
            if (c.g < c.b) {
                h += 6.0;
            }
        } else if (maxC == c.g) {
            h = 2.0 + (c.b - c.r) / delta;
        } else { // maxC == c.b
            h = 4.0 + (c.r - c.g) / delta;
        }
        h /= 6.0; // convert to [0..1]
    }
    
    return vec3(h, s, v);
}

// Helper: convert HSV (H in [0..1], S,V in [0..1]) to RGB
vec3 hsvToRgb(vec3 c) {
    float h = c.x * 6.0;  // sector [0..6)
    float s = c.y;
    float v = c.z;
    
    float i = floor(h);
    float f = h - i;
    
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));
    
    vec3 rgb;
    if (i < 1.0) {
        rgb = vec3(v, t, p);
    } else if (i < 2.0) {
        rgb = vec3(q, v, p);
    } else if (i < 3.0) {
        rgb = vec3(p, v, t);
    } else if (i < 4.0) {
        rgb = vec3(p, q, v);
    } else if (i < 5.0) {
        rgb = vec3(t, p, v);
    } else {
        rgb = vec3(v, p, q);
    }
    
    return rgb;
}

void main()
{
    vec3 orig = texture(screenTexture, TexCoords).rgb;

    // 1. Convert from RGB -> HSV
    vec3 hsv = rgbToHsv(orig);

    // 2. Adjust hue: hueShiftDeg is in degrees, so convert to [0..1] range.
    float hueShift = hueShiftDeg / 360.0;
    hsv.x = fract(hsv.x + hueShift); // fract to wrap around [0..1]

    // 3. Multiply saturation and value
    hsv.y = clamp(hsv.y * satScale, 0.0, 1.0);
    hsv.z = clamp(hsv.z * valScale, 0.0, 1.0);

    // 4. Convert back to RGB
    vec3 finalColor = hsvToRgb(hsv);

    FragColor = vec4(finalColor, 1.0);
}
