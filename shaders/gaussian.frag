#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform vec2 texelSize;    // 1.0 / (width, height)
uniform int kernelSize;    // e.g. 3, 5, 7, etc.
uniform float sigma;       // standard deviation
uniform int randomSeed;    // optional, can use for random offsets if desired

// A very basic naive Gaussian approach
void main()
{
    // We assume kernelSize is odd (3,5,7...). You might clamp or handle even sizes in your code.
    // We'll compute weights using a typical Gaussian formula:
    //   weight = exp( -(x^2 + y^2) / (2*sigma^2) )
    // Then normalize the sum of weights at the end.

    float halfSize = float(kernelSize / 2);  // integer division floors, but it's okay
    float twoSigma2 = 2.0 * sigma * sigma;

    vec3 colorSum = vec3(0.0);
    float weightSum = 0.0;

    // Optionally, you could incorporate randomSeed for offsets, e.g. hashing randomSeed + i,j
    // but for a stable blur, we won't do that by default.

    for(int i = -kernelSize/2; i <= kernelSize/2; i++)
    {
        for(int j = -kernelSize/2; j <= kernelSize/2; j++)
        {
            float r2 = float(i*i + j*j); // squared distance
            float weight = exp(-r2 / twoSigma2);

            // Sample from offset
            vec2 offset = vec2(i, j) * texelSize;
            vec3 sampleCol = texture(screenTexture, TexCoords + offset).rgb;

            colorSum += sampleCol * weight;
            weightSum += weight;
        }
    }

    // Normalize
    colorSum /= weightSum;

    FragColor = vec4(colorSum, 1.0);
}
