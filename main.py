import os
import glfw
import cv2
import numpy as np
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *
import ctypes
import time

import json

###############################################################################
# 1) File reading & shader compilation
###############################################################################
def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        err = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error:\n{err}")
    return shader

def create_shader_program(vertex_src, fragment_src):
    v = compile_shader(vertex_src, GL_VERTEX_SHADER)
    f = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, v)
    glAttachShader(program, f)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        err = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program link error:\n{err}")
    glDeleteShader(v)
    glDeleteShader(f)
    return program

def get_config():
    _config_raw = None
    with open("config.json","r") as fi:
        _config_raw = json.load(fi)
    return _config_raw

###############################################################################
# 2) Main application
###############################################################################
class shaderViewer:
    def main(self):
        self.CONFIG = get_config()
        # ------------------- GLFW init -------------------
        if not glfw.init():
            print("Failed to initialize GLFW.")
            return

        width, height = 1280, 720
        window = glfw.create_window(width, height, "Multi-Pass Effects with Gaussian Blur", None, None)
        if not window:
            glfw.terminate()
            return
        glfw.make_context_current(window)

        # ------------------- ImGui Setup -------------------
        imgui.create_context()
        impl = GlfwRenderer(window)

        # ------------------- Load & Compile Shaders -------------------
        SHADERS_DIR = "shaders"  # adjust if needed

        effect_programs = {}

        pass_vert = read_file(os.path.join(SHADERS_DIR, "pass_through.vert"))
        pass_frag = read_file(os.path.join(SHADERS_DIR, "pass_through.frag"))
        pass_through_program = create_shader_program(pass_vert, pass_frag)

        for shaderName, _ in self.CONFIG.items():
            shaderFilename = _.get("filename")
            _iterFragmentShader = read_file(os.path.join(SHADERS_DIR, shaderFilename))
            _iterShaderProgram = create_shader_program(pass_vert, _iterFragmentShader)
            effect_programs[shaderName] = _iterShaderProgram

        bluenoise_frag = read_file(os.path.join(SHADERS_DIR, "bluenoise_dither.frag"))
        bluenoise_program = create_shader_program(pass_vert, bluenoise_frag)

        effect_programs.update( {
            "BluenoiseDither": bluenoise_program,
        } )

        self.CONFIG["BluenoiseDither"] = {}

        # ------------------- Fullscreen Quad Setup -------------------
        quad_vertices = np.array([
            # positions   # texCoords
            -1.0,  1.0,   0.0, 1.0,
            -1.0, -1.0,   0.0, 0.0,
            1.0, -1.0,   1.0, 0.0,

            -1.0,  1.0,   0.0, 1.0,
            1.0, -1.0,   1.0, 0.0,
            1.0,  1.0,   1.0, 1.0,
        ], dtype=np.float32)

        VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        glBindVertexArray(VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * quad_vertices.itemsize, ctypes.c_void_p(0))
        # texCoord
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * quad_vertices.itemsize, ctypes.c_void_p(2 * quad_vertices.itemsize))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # ------------------- Create FBOs for Ping-Pong -------------------
        def create_fbo_texture(w, h):
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glBindTexture(GL_TEXTURE_2D, 0)
            return tex

        def create_fbo(w, h):
            fbo = glGenFramebuffers(1)
            tex = create_fbo_texture(w, h)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Framebuffer incomplete!")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return fbo, tex

        fbo1, tex1 = create_fbo(width, height)
        fbo2, tex2 = create_fbo(width, height)

        # ------------------- Video Capture -------------------
        cap = cv2.VideoCapture("video.mp4")
        if not cap.isOpened():
            raise RuntimeError("Failed to open video.mp4 or the path is incorrect.")

        video_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, video_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)


        # Load the blue noise file
        blue_noise_image = cv2.imread("LDR_LLL1_0.png", cv2.IMREAD_GRAYSCALE)
        if blue_noise_image is None:
            raise RuntimeError("Could not load blue noise file!")
        # Flip or otherwise transform if needed, depending on your coordinate conventions
        blue_noise_image = cv2.flip(blue_noise_image, 0)  # typical for OpenGL texture

        blueNoiseTexID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, blueNoiseTexID)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RED,
            blue_noise_image.shape[1], blue_noise_image.shape[0],
            0, GL_RED, GL_UNSIGNED_BYTE, blue_noise_image
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # To tile automatically, set wrapping mode:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glBindTexture(GL_TEXTURE_2D, 0)

        # ------------------- The Effects Stack -------------------
        # Each item: { "type": <string>, "params": { ... } }
        shader_stack = []

        # ------------------- Apply Effect Function -------------------
        def apply_effect(effect, input_tex, output_fbo):
            """
            Render from 'input_tex' into 'output_fbo' using the effect's fragment shader,
            setting any needed uniforms from effect["params"].
            """
            effect_type = effect["type"]
            params      = effect["params"]
            program     = effect_programs[effect_type]
            glUseProgram(program)

            # Common uniform: screenTexture
            loc_tex = glGetUniformLocation(program, "screenTexture")
            glUniform1i(loc_tex, 0)  # texture unit 0


            for iterShaderEffect, _ in self.CONFIG.items():
                iterShaderParameters = _.get("parameters")
                iterShaderExtraParameters = _.get("otherParameters",{})
                if "texelSize" in iterShaderExtraParameters.keys():
                    loc_ts = glGetUniformLocation(program, "texelSize")
                    if loc_ts != -1:
                        glUniform2f(loc_ts, 1.0/width, 1.0/height)
                if effect_type == iterShaderEffect:
                    if iterShaderParameters:
                        for iterParam,paramData in iterShaderParameters.items():
                            loc_var = glGetUniformLocation(program, iterParam)
                            if loc_var != -1:
                                if paramData["type"] == "float":
                                    glUniform1f(loc_var, params.get(iterParam, paramData.get("default_v")))
                                elif paramData["type"] == "int":
                                    glUniform1i(loc_var, params.get(iterParam, paramData.get("default_v")))
                                elif paramData["type"] == "bool":
                                    _val = params.get(iterParam, paramData.get("default_v"))
                                    _val = 1 if _val else 0
                                    glUniform1i(loc_var, _val)



            if effect_type == "BluenoiseDither":
                # screenTexture at unit 0
                loc_screen = glGetUniformLocation(program, "screenTexture")
                glUniform1i(loc_screen, 0)

                # blueNoiseTexture at unit 1
                loc_noise = glGetUniformLocation(program, "blueNoiseTexture")
                glUniform1i(loc_noise, 1)

                # Bind the input_tex to 0
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, input_tex)

                # Bind the blue noise to 1
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, blueNoiseTexID)

                # Render quad to output_fbo
                glBindFramebuffer(GL_FRAMEBUFFER, output_fbo)
                glViewport(0, 0, width, height)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindVertexArray(VAO)
                glDrawArrays(GL_TRIANGLES, 0, 6)
                glBindVertexArray(0)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)

            # Render the quad to output_fbo
            glBindFramebuffer(GL_FRAMEBUFFER, output_fbo)
            glViewport(0, 0, width, height)
            glClear(GL_COLOR_BUFFER_BIT)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, input_tex)

            glBindVertexArray(VAO)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # ------------------- ImGui Popup State -------------------
        show_add_effect_popup = False

        effect_types = list(self.CONFIG.keys())

        selected_effect_idx = 0

        temp_shader_vars = {}
        # Create temporary variables based on list
        for iterShaderEffect, _ in self.CONFIG.items():
            iterShaderParameters = _.get("parameters")
            temp_shader_vars[iterShaderEffect] = {}
            if iterShaderParameters:
                for iterParam,iterParamData in iterShaderParameters.items():
                    temp_shader_vars[iterShaderEffect][iterParam] = iterParamData["default_v"]


        target_frame_time = 1.0 / 30.0  # target frame duration (33.33 ms)


        # Main Render Loop
        while not glfw.window_should_close(window):
            frame_start = time.time()  # record frame start time

            glfw.poll_events()
            impl.process_inputs()

            # 1) Update video frame (loop if needed)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)

            glBindTexture(GL_TEXTURE_2D, video_texture)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB,
                frame.shape[1], frame.shape[0], 0,
                GL_RGB, GL_UNSIGNED_BYTE, frame
            )
            glBindTexture(GL_TEXTURE_2D, 0)

            # 2) Pass-through from video -> fbo1
            glUseProgram(pass_through_program)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo1)
            glViewport(0, 0, width, height)
            glClear(GL_COLOR_BUFFER_BIT)

            passTexLoc = glGetUniformLocation(pass_through_program, "screenTexture")
            glUniform1i(passTexLoc, 0)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, video_texture)

            glBindVertexArray(VAO)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            current_tex = tex1
            src_fbo, dst_fbo = fbo1, fbo2
            src_tex, dst_tex = tex1, tex2

            # 3) Apply each effect in the stack
            for effect in shader_stack:
                apply_effect(effect, src_tex, dst_fbo)
                src_fbo, dst_fbo = dst_fbo, src_fbo
                src_tex, dst_tex = dst_tex, src_tex

            # 4) Final pass to screen
            glUseProgram(pass_through_program)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, width, height)
            glClearColor(0, 0, 0, 1)
            glClear(GL_COLOR_BUFFER_BIT)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, src_tex)

            passTexLoc = glGetUniformLocation(pass_through_program, "screenTexture")
            glUniform1i(passTexLoc, 0)

            glBindVertexArray(VAO)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)

            # ------------------- ImGui UI -------------------
            imgui.new_frame()
            imgui.begin("Shader Stack")

            # Button to open the "Add Effect" popup
            if imgui.button("Add Effect"):
                show_add_effect_popup = True
                imgui.open_popup("AddEffectPopup")

            # Show the popup if triggered
            if show_add_effect_popup:
                # Create a modal popup
                if imgui.begin_popup_modal("AddEffectPopup", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                    # 1) Choose effect type
                    clicked, selected_effect_idx = imgui.combo(
                        "Effect Type",
                        selected_effect_idx,
                        effect_types
                    )


                    effect_type = effect_types[selected_effect_idx]

                    # 2) Show relevant parameters
                    selectedShaderEffect = self.CONFIG.get(effect_type)
                    if selectedShaderEffect.get("parameters"):
                        for iterParam,iterParamData in selectedShaderEffect["parameters"].items():
                            if iterParamData.get("type") == "float":
                                _, temp_shader_vars[effect_type][iterParam] = imgui.slider_float(iterParamData.get("gui_text"), temp_shader_vars[effect_type][iterParam], iterParamData["range"][0], iterParamData["range"][1], "%.2f")  
                            elif iterParamData.get("type") == "int":
                                _, temp_shader_vars[effect_type][iterParam]  = imgui.slider_int(iterParamData.get("gui_text"), temp_shader_vars[effect_type][iterParam], iterParamData["range"][0], iterParamData["range"][1])
                            elif iterParamData.get("type") == "bool":
                                _, temp_shader_vars[effect_type][iterParam] = imgui.checkbox(iterParamData.get("gui_text"), temp_shader_vars[effect_type][iterParam])

                    # 3) Confirm / Cancel
                    if imgui.button("Confirm"):
                        new_effect = {}
                        new_effect["type"] = effect_type
                        new_effect["params"] = {}

                        if selectedShaderEffect.get("parameters"):
                            for iterParam,iterParamData in selectedShaderEffect["parameters"].items(): 
                                new_effect["params"][iterParam] = temp_shader_vars[effect_type][iterParam]

                        # Add to stack
                        shader_stack.append(new_effect)
                        # Close popup
                        imgui.close_current_popup()
                        show_add_effect_popup = False

                    imgui.same_line()
                    if imgui.button("Cancel"):
                        imgui.close_current_popup()
                        show_add_effect_popup = False

                    imgui.end_popup()

            # Show the current stack with parameters
            imgui.text("Effects in Order:")
            for i, effect in enumerate(shader_stack):
                imgui.push_id(str(i))
                effect_type = effect["type"]
                params = effect["params"]

                # Get format string from config
                effect_config = self.CONFIG.get(effect_type, {})
                display_format = effect_config.get("display_format", "{index}: " + effect_type)

                # Prepare the parameters dynamically
                param_values = {"index": i, **params}

                # Special handling for BrightnessThreshold (boolean comparison)
                if effect_type == "BrightnessThreshold":
                    param_values["comparison"] = ">" if params.get("useGreater", True) else "<"

                # Render the effect text using the format from config.json
                try:
                    imgui.text(display_format.format(**param_values))
                except KeyError as e:
                    imgui.text(f"{i}: {effect_type} (Error: Missing key {str(e)})")

                imgui.same_line()
                if imgui.button("Remove"):
                    shader_stack.pop(i)
                imgui.pop_id()
                
            imgui.end()
            imgui.render()
            impl.render(imgui.get_draw_data())

            glfw.swap_buffers(window)
            
            # ------------------- Frame Limiting -------------------
            frame_time = time.time() - frame_start
            sleep_time = target_frame_time - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        # Cleanup
        cap.release()
        glfw.terminate()

if __name__ == "__main__":
    shaderViewer().main()
