#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "GL/gl3w.h"
#include <GLFW/glfw3.h>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <random>

struct RenderData {
    unsigned int compute_program;
    float sky_color[3];
    int sample_count;
    int nsamples;
    float albedo;
};

unsigned int loadShader(std::string path, unsigned int type)
{
    std::ifstream file(path);
    std::stringstream buf;
    buf << file.rdbuf();
    std::string src = buf.str();
    const char* source = src.c_str();

    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (success != GL_TRUE) {
        GLint log_size;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_size);
        char* shader_log = static_cast<char*>(malloc(log_size));
        glGetShaderInfoLog(shader, log_size, NULL, shader_log);
        if (type == GL_VERTEX_SHADER) {
            std::cerr << "Vertex:" << shader_log << std::endl;
        } else {
            std::cerr << "Fragment:" << shader_log << std::endl;
        }
    }

    return shader;
}

unsigned int loadCompute()
{
    unsigned int compute_shader = loadShader("compute.glsl", GL_COMPUTE_SHADER);
    unsigned int compute_program = glCreateProgram();
    glAttachShader(compute_program, compute_shader);
    glLinkProgram(compute_program);
    glDeleteShader(compute_shader);

    return compute_program;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        RenderData* ptr = (RenderData*)glfwGetWindowUserPointer(window);
        glDeleteProgram(ptr->compute_program);
        ptr->compute_program = loadCompute();
        ptr->sample_count = 0;
    }
}

float* loadVolume(std::string path, unsigned int& size_x, unsigned int& size_y, unsigned int& size_z, float& sigma_hat)
{
    std::ifstream f(path);

    int min_x, min_y, min_z, max_x, max_y, max_z;
    f.read((char*)&min_x, 4);
    f.read((char*)&min_y, 4);
    f.read((char*)&min_z, 4);
    f.read((char*)&max_x, 4);
    f.read((char*)&max_y, 4);
    f.read((char*)&max_z, 4);

    printf("(%u, %u, %u) -> (%u, %u, %u)\n", min_x,min_y, min_z, max_x, max_y, max_z);
    size_x = (unsigned)(max_x - min_x + 1);
    size_y = (unsigned)(max_y - min_y + 1);
    size_z = (unsigned)(max_z - min_z + 1);
    float* buf = (float*)malloc(size_x*size_y*size_z*sizeof(float));

    sigma_hat = 0.;

    float* ptr = buf;
    for (int z = min_z; z <= max_z; z++)
    {
        for (int y = min_y; y <= max_y; y++)
        {
            for (int x = min_x; x <= max_x; x++)
            {
                f.read((char*)ptr, 4);
                sigma_hat = std::max(sigma_hat, *ptr);
                ptr++;
            }
        }
    }
    std::cout << sigma_hat << std::endl;
    return buf;
}

int main()
{
    GLFWwindow* window;
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(1024, 1024, "GPU Raytracing", NULL, NULL);
    glfwSetKeyCallback(window, key_callback);

    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    gl3wInit();

    glViewport(0, 0, 1024, 1024);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");

    float vertices[] = {
        1.f,  1.f, 0.0f,  1.f, 1.f, // top right
        1.f, -1.f, 0.0f,  1.f, 0.f, // bottom right
        -1.f, -1.f, 0.0f, 0.f, 0.f, // bottom left
        -1.f,  1.f, 0.0f, 0.f, 1.f // top left 
    };

    unsigned int indices[] = {  // note that we start from 0!
        0, 1, 3,   // first triangle
        1, 2, 3    // second triangle
    };  

    unsigned int size_x, size_y, size_z;
    float sigma_hat;
    float* density = loadVolume("out.dat", size_x, size_y, size_z, sigma_hat);

    unsigned int vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    unsigned int program = glCreateProgram();

    unsigned int vertex_shader = loadShader("base.vs", GL_VERTEX_SHADER);
    unsigned int fragment_shader = loadShader("base.fs", GL_FRAGMENT_SHADER);

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);  
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);


    unsigned int density_tex;
    glGenTextures(1, &density_tex);
    glBindTexture(GL_TEXTURE_3D, density_tex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, size_x, size_y, size_z, 0, GL_RED, GL_FLOAT, density);
    glGenerateMipmap(GL_TEXTURE_3D);

    unsigned int compute_shader = loadShader("compute.glsl", GL_COMPUTE_SHADER);
    RenderData data;
    data.compute_program = glCreateProgram();
    data.sky_color[0] = .6;
    data.sky_color[1] = .7;
    data.sky_color[2] = .8;
    data.sample_count = 0;
    data.nsamples = 30;
    data.albedo = .5f;
    float density_scale = 10.f;

    glfwSetWindowUserPointer(window, &data);
    glAttachShader(data.compute_program, compute_shader);
    glLinkProgram(data.compute_program);
    glDeleteShader(compute_shader);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Hello");
        if (ImGui::ColorEdit3("Sky Color", data.sky_color)) data.sample_count = 0;
        ImGui::Text("Samples: %d", data.sample_count);
        ImGui::SliderInt("Samples / Frame", &data.nsamples, 1, 100);
        if (ImGui::SliderFloat("albedo", &data.albedo, 0., 1.)
        ||  ImGui::SliderFloat("density scale", &density_scale, 1., 100.)) {
            data.sample_count = 0;
        }
        ImGui::End();
        ImGui::Render();

        if (data.sample_count < 300) {
            glUseProgram(data.compute_program);
            glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
            glUniform3fv(0, 1, data.sky_color);
            glUniform1i(1, data.sample_count);
            glUniform1i(2, data.nsamples);
            glUniform1f(3, data.albedo);
            glUniform1f(4, sigma_hat);
            glUniform1f(5, density_scale);
            glDispatchCompute(64, 64, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            data.sample_count += data.nsamples;
        }

        glUseProgram(program);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
    
    glfwTerminate();
    return 0;
}
