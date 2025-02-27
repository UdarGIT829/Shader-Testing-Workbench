from setuptools import setup, find_packages

setup(
    name="Shader-Testing-Workbench",  # Replace with your project name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "glfw",
        "opencv-python",
        "numpy",
        "imgui",
        "PyOpenGL",
    ],
    python_requires=">=3.6",
    author="Viraat Udar",
    author_email="",
    description="A project using OpenGL, ImGui, and OpenCV",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
