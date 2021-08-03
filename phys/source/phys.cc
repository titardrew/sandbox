#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define OFFSETOF(s, m) ((size_t)&(((s *)0)->m))

const char *FragmentShader =
    "#version 330 core\n"
    "\n"
    "in vec4 frag_color;\n"
    "out vec4 color;\n"
    "\n"
    "void main()\n"
    "{\n"
    "   color = frag_color;\n"
    "}\n";

const char *VertexShader =
    "#version 330 core\n"
    "\n"
    "layout (location = 0) in vec3 position;\n"
    "layout (location = 1) in vec4 fragment_color;\n"
    "uniform mat4 proj_matr;\n"
    "out vec4 frag_color;\n"
    "\n"
    "void main()\n"
    "{\n"
    "   frag_color = fragment_color;\n"
    "   gl_Position = proj_matr * vec4(position, 1.0f);\n"
    "}\n";


GLint CompileShader(GLuint shader_id, const char *shader_source)
{
    glShaderSource(shader_id, 1, &shader_source, nullptr);
    glCompileShader(shader_id);

    GLint result = GL_FALSE;
    int info_log_length = 0;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        char *msg = new char[info_log_length + 1];
        glGetShaderInfoLog(shader_id, info_log_length, nullptr, msg);
        printf("Shader compilation error: %s\n", msg);
        delete[] msg;
    }
    return result;
}

GLuint LoadShaders()
{
    GLuint vs_id = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs_id = glCreateShader(GL_FRAGMENT_SHADER);

    CompileShader(vs_id, VertexShader);
    CompileShader(fs_id, FragmentShader);

    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vs_id);
    glAttachShader(program_id, fs_id);
    glLinkProgram(program_id);

    GLint result = GL_FALSE;
    int info_log_length = 0;
    glGetProgramiv(program_id, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        char *msg = new char[info_log_length + 1];
        glGetProgramInfoLog(program_id, info_log_length, nullptr, msg);
        printf("Program linking error: %s\n", msg);
        delete[] msg;
    }

    glDetachShader(program_id, vs_id);
    glDetachShader(program_id, fs_id);

    glDeleteShader(vs_id);
    glDeleteShader(fs_id);

    return program_id;
}

void PrintError(const char *msg = "")
{
#if 1
    const char *error;
    auto err_code = glGetError();
    switch (err_code) {
        case 0:                                error = "OK"; break;
        case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
        case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
        case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
        case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
        case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
        case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
        case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        default:                               error = "unk code";
    }
    printf("GL STATUS %s: %u (%s)\n", msg, err_code, error);
#endif
}


struct RenderingContext {
    GLuint program_id;
    GLuint attr_position_location;
    GLuint attr_color_location;
    GLuint proj_matr_location;
    GLuint vbo, vao;
};

struct Vec3f {
    float x, y, z;
};

struct Color {
    uint8_t r, g, b, a;
};

struct Vertex {
    Vec3f pos;
    Color col;
};

int main(int, char **)
{
    GLFWwindow *window;
    if (!glfwInit()) {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

    window = glfwCreateWindow( 768, 768, "OpenGL", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, "
                "they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = true;  // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    PrintError();

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

    RenderingContext ctx;
    ctx.program_id = LoadShaders();
    ctx.attr_position_location = glGetAttribLocation(ctx.program_id, "position");
    ctx.attr_color_location    = glGetAttribLocation(ctx.program_id, "fragment_color");
    ctx.proj_matr_location     = glGetUniformLocation(ctx.program_id, "proj_matr");
    glUseProgram(ctx.program_id);
    glGenVertexArrays(1, &ctx.vao);
    glBindVertexArray(ctx.vao);
    float foc = 1.0f;
    const float persp_projection[4][4] =
    {
        {foc,  0.0f, 0.0f, 0.0f},
        {0.0f, foc,  0.0f, 0.0f},
        {0.0f, 0.0f, foc,  0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    };
    glUniformMatrix4fv(ctx.proj_matr_location, 1, GL_FALSE, &persp_projection[0][0]);
    glGenBuffers(1, &ctx.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo);
    glEnableVertexAttribArray(ctx.attr_position_location);
    glEnableVertexAttribArray(ctx.attr_color_location);
    glVertexAttribPointer(ctx.attr_position_location, 3, GL_FLOAT, GL_FALSE,
                          sizeof(Vertex), (GLvoid *)OFFSETOF(Vertex, pos));
    glVertexAttribPointer(ctx.attr_color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE,
                          sizeof(Vertex), (GLvoid *)OFFSETOF(Vertex, col));
    printf("Program id: %u\n", ctx.program_id);

    int64_t t = 0;
    for (;;)
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE ) == GLFW_PRESS
                || glfwWindowShouldClose(window) != 0) {
            break;
        }
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.1f, 0.1f, 0.1f, 0.0f);

        Vertex triangle[3] = {
            {.pos=Vec3f{.x=1.0f, .y=0.0f, .z=0.0f}, .col=Color{.r=255, .g=255, .b=255, .a=255}},
            {.pos=Vec3f{.x=0.5f, .y=0.5f, .z=0.0f}, .col=Color{.r=0  , .g=255, .b=255, .a=255}},
            {.pos=Vec3f{.x=0.0f, .y=0.0f, .z=0.5f}, .col=Color{.r=255, .g=0  , .b=255, .a=255}},
        };
        glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(Vertex), (const GLvoid *)&triangle[0], GL_STREAM_DRAW);
        // glDrawElements(GL_TRIANGLES, (GLsizei)3, GL_UNSIGNED_INT, (void *)(intptr_t)0);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)3);

        glfwSwapBuffers(window);
        glfwPollEvents();

        ++t;
    }

    printf("Hello, World!\n");
    return 0;
}
