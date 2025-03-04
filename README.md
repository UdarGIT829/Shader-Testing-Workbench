# Shader-Testing-Workbench
A Python toolkit and viewer for testing Post-Processing shaders.

## Installation
A virtual environment is recommended, but not required.

### Setup tools
`
pip install -e .
`

## Features

- Dynamic selection of shaders using OpenGL framebuffer.
- Define shader configuration using JSON

Shaders include:
- Edge Detection (Sobel)
- RGB Inversion
- Colorize (Redshift)
- Color Shift
- Brightness threshold
- Gaussian Blur
- Blue Noise Dithering

You can stack shaders in different orders (improved controls coming soon)
![Demo Preview](demo.gif)

## Planned Features

- Extend JSON configuration to API request+update
- Dynamic implementation of texture sampling
- Effect Masking (Limit calculation of shader)
- Alpha Masking (Mix Shaders)
- Visibility grouping (Toggle Visibility/Pipe to Mask)
- Emission shader (Multiple Passes)
- Depth Estimation (Midas) 

# Credits
Blue noise texture source: https://github.com/Calinou/free-blue-noise-textures

Droid object: Unclear Sketchfab artist (@rollingstone ?)