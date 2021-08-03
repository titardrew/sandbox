#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "shader.h"

#define DO_ONCE for (static int s__i = 0; s__i < 1; ++s__i)
#define OFFSETOF(TYPE, ELEMENT) ((size_t)&(((TYPE *)0)->ELEMENT))

GLFWwindow *window;

template <typename T>
T max(T x, T y)
{
    return x > y ? x : y;
}

template <typename T>
T min(T x, T y)
{
    return x < y ? x : y;
}


inline double GetClockSeconds()
{
    struct timeval current_time;
    gettimeofday(&current_time, nullptr);
    return (double)current_time.tv_sec  + (double)current_time.tv_usec / 1000000.0f;
}

struct FrameClock {
    double  time_last_s   = 0.0;
    double  dt_s          = 0.0;
    int64_t n_frames_acc  = 0;
    bool    is_paused     = true;

    FrameClock &SetSecondsPerFrame(double dt_s)
    {
        assert(dt_s > 0);
        n_frames_acc = Frame();
        time_last_s = GetClockSeconds();

        this->dt_s = dt_s;
        return *this;
    }

    void Reset()
    {
        time_last_s = GetClockSeconds();
        is_paused = false;
    }

    void Pause()
    {
        n_frames_acc = Frame();
        time_last_s = GetClockSeconds();
        is_paused = true;
    }

    void UnPause()
    {
        time_last_s = GetClockSeconds();
        is_paused = false;
    }

    float Seconds()
    {
        return (float)Frame() * dt_s;
    }

    int64_t Frame()
    {
        if (is_paused) {
            return n_frames_acc;
        } else {
            double clk = GetClockSeconds();
            return (int64_t)floor((clk - time_last_s) / dt_s) + n_frames_acc;
        }
    }

} frame_clock;

static void PrintMatrix(int n, int m, const float *matr, const char *name = nullptr);

struct Color {
    uint8_t r, g, b, a;
};

struct Vec2f {
    float x, y;
};

void operator +=(Vec2f &lhs, const Vec2f &rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
}

void operator +=(Vec2f &lhs, float rhs)
{
    lhs.x += rhs;
    lhs.y += rhs;
}

void operator -=(Vec2f &lhs, const Vec2f &rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
}

void operator -=(Vec2f &lhs, float rhs)
{
    lhs.x -= rhs;
    lhs.y -= rhs;
}

void operator *=(Vec2f &lhs, float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
}

Vec2f operator +(const Vec2f &lhs, const Vec2f &rhs)
{
    Vec2f result;
    result.x = lhs.x + rhs.x;
    result.y = lhs.y + rhs.y;
    return result;
}

Vec2f operator +(const Vec2f &lhs, float rhs)
{
    Vec2f result;
    result.x = lhs.x + rhs;
    result.y = lhs.y + rhs;
    return result;
}

Vec2f operator +(float lhs, const Vec2f &rhs)
{
    Vec2f result;
    result.x = lhs + rhs.x;
    result.y = lhs + rhs.y;
    return result;
}

Vec2f operator -(const Vec2f &lhs, float rhs)
{
    Vec2f result;
    result.x = lhs.x - rhs;
    result.y = lhs.y - rhs;
    return result;
}

Vec2f operator -(const Vec2f &lhs, const Vec2f &rhs)
{
    Vec2f result;
    result.x = lhs.x - rhs.x;
    result.y = lhs.y - rhs.y;
    return result;
}

Vec2f operator -(float lhs, const Vec2f &rhs)
{
    Vec2f result;
    result.x = lhs - rhs.x;
    result.y = lhs - rhs.y;
    return result;
}

Vec2f operator *(const Vec2f &lhs, float rhs)
{
    Vec2f result;
    result.x = lhs.x * rhs;
    result.y = lhs.y * rhs;
    return result;
}

Vec2f operator *(float lhs, const Vec2f &rhs)
{
    Vec2f result;
    result.x = lhs * rhs.x;
    result.y = lhs * rhs.y;
    return result;
}

struct Vertex {
    Vec2f pos;
    Color col;
};

constexpr Color ColorWhite  = {.r=255, .g=255, .b=255, .a=255};
constexpr Color ColorGray   = {.r=125, .g=125, .b=125, .a=255};
constexpr Color ColorBlack  = {.r=0,   .g=0,   .b=0,   .a=255};
constexpr Color ColorRed    = {.r=255, .g=0,   .b=0,   .a=255};
constexpr Color ColorGreen  = {.r=0,   .g=180, .b=0,   .a=255};
constexpr Color ColorBlue   = {.r=0,   .g=0,   .b=255, .a=255};
constexpr Color ColorYellow = {.r=200, .g=200, .b=0,   .a=255};

struct RenderingContext {
    std::vector<Vertex> verts;
    std::vector<uint32_t> inds;
    uint32_t vertex_buffer_object;
    uint32_t element_buffer_object;
    uint32_t shader;
    GLuint attr_position_location;
    GLuint attr_color_location;
    GLuint shader_location;
    GLuint vertex_array_object;
};


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


RenderingContext CreateRenderingContext()
{
    RenderingContext ctx;
    ctx.shader = LoadShaders("shaders/simple.vs", "shaders/simple.fs");
    glGenBuffers(1, &ctx.vertex_buffer_object);
    glGenBuffers(1, &ctx.element_buffer_object);
    ctx.attr_position_location = glGetAttribLocation(ctx.shader, "position");
    ctx.attr_color_location = glGetAttribLocation(ctx.shader, "fragment_color");
    ctx.shader_location = glGetUniformLocation(ctx.shader, "projection_matrix");
    glUseProgram(ctx.shader);
    glGenVertexArrays(1, &ctx.vertex_buffer_object);
    glBindVertexArray(ctx.vertex_buffer_object);
    return ctx;
}

void SetupRenderingContext(RenderingContext *ctx)
{
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    // @Todo(aty): try [-1;1] and use glViewport(...)
    // @Todo(aty): denormalize this.
    float L = 0;
    float R = 1; // 768;
    float B = 1; // 768;
    float T = 0;
    const float ortho_projection[4][4] =
    {
        { 2.0f/(R-L),   0.0f,         0.0f,   0.0f },
        { 0.0f,         2.0f/(T-B),   0.0f,   0.0f },
        { 0.0f,         0.0f,        -1.0f,   0.0f },
        { (R+L)/(L-R),  (T+B)/(B-T),  0.0f,   1.0f },
    };

    DO_ONCE {
        PrintMatrix(4, 4, &ortho_projection[0][0], "Projection");
    }

    glUniformMatrix4fv(ctx->shader_location, 1, GL_FALSE, &ortho_projection[0][0]);
    glBindBuffer(GL_ARRAY_BUFFER, ctx->vertex_buffer_object);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ctx->element_buffer_object);
    glUseProgram(ctx->shader);

    glEnableVertexAttribArray(ctx->attr_position_location);
    glEnableVertexAttribArray(ctx->attr_color_location);
    glVertexAttribPointer(ctx->attr_position_location, 2, GL_FLOAT, GL_FALSE,
                          sizeof(Vertex), (GLvoid *)OFFSETOF(Vertex, pos));
    glVertexAttribPointer(ctx->attr_color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE,
                          sizeof(Vertex), (GLvoid *)OFFSETOF(Vertex, col));
}

void Render(RenderingContext *ctx)
{
    glUseProgram(ctx->shader);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)ctx->verts.size() * (int)sizeof(Vertex),
                 (const GLvoid*)ctx->verts.data(), GL_STREAM_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)ctx->inds.size() * (int)sizeof(uint32_t),
                 (const GLvoid *)ctx->inds.data(), GL_STREAM_DRAW);

    glDrawElements(GL_TRIANGLES, (GLsizei)ctx->inds.size(), GL_UNSIGNED_INT, (void*)(intptr_t)0);
    // printf("n_verts=%zu, n_inds=%zu\n", ctx->verts.size(), ctx->inds.size());

    ctx->verts.clear();
    ctx->inds.clear();
}

void PrintMatrix(int n, int m, const float *matr, const char *name)
{
    if (name) {
        printf("Matrix %s:\n", name);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("%.2f ", matr[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static inline float Rsqrt(float x)
{
    return 1.0f / sqrtf(x);
}

constexpr float kMaxNorm = 100.0f;
inline void Normalize(float &dx, float &dy)
{
    float d2 = dx*dx + dy*dy;
    if (d2 > 0.0f) {
        float inv_d = Rsqrt(d2);
        dx *= inv_d;
        dy *= inv_d;
    }
}

void AddLine(RenderingContext *ctx, Vec2f p0, Vec2f p1, float w, Color col)
{
    uint32_t first_id = ctx->inds.size();
    uint32_t first_v  = ctx->verts.size();

    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    w /= 2;

    Normalize(dx, dy);
    float nx = dy;
    float ny = -dx;

    ctx->verts.reserve(first_v + 4);
    ctx->verts.push_back({.pos={p0.x + nx*w, p0.y + ny*w}, .col=col});
    ctx->verts.push_back({.pos={p0.x - nx*w, p0.y - ny*w}, .col=col});
    ctx->verts.push_back({.pos={p1.x + nx*w, p1.y + ny*w}, .col=col});
    ctx->verts.push_back({.pos={p1.x - nx*w, p1.y - ny*w}, .col=col});

    ctx->inds.reserve(first_id + 6);
    ctx->inds.push_back(first_v + 0);
    ctx->inds.push_back(first_v + 1);
    ctx->inds.push_back(first_v + 2);
    ctx->inds.push_back(first_v + 1);
    ctx->inds.push_back(first_v + 2);
    ctx->inds.push_back(first_v + 3);
}

/// @Todo: change triangulation.
void AddRect(RenderingContext *ctx, Vec2f p0, Vec2f size, float w, Color col)
{
    float hw = w / 2;
    Vec2f pt0, pt1;

    pt0 = p0 + Vec2f{-hw, 0.0f};
    pt1 = p0 + Vec2f{size.x + hw, 0.0f};
    AddLine(ctx, pt0, pt1, w, col);

    pt0 = p0 + Vec2f{size.x, hw};
    pt1 = p0 + Vec2f{size.x, size.y - hw};
    AddLine(ctx, pt0, pt1, w, col);

    pt0 = p0 + Vec2f{size.x + hw, size.y};
    pt1 = p0 + Vec2f{-hw, size.y};
    AddLine(ctx, pt0, pt1, w, col);

    pt0 = p0 + Vec2f{0.0f, size.y - hw};
    pt1 = p0 + Vec2f{0.0f, hw};
    AddLine(ctx, pt0, pt1, w, col);
}

void AddArc(
    RenderingContext *ctx, Vec2f center, float r, float a_min,
    float a_max, float w, Color col, int n_segments = 12)
{
    float seg_angle = (a_max - a_min) / n_segments;

    Vec2f pos0, pos1;
    pos0.x = center.x + r * cos(a_min);
    pos0.y = center.y + r * sin(a_min);

    for (int seg = 1; seg <= n_segments; ++seg) {
        float angle = a_min + seg * seg_angle;
        pos1.x = center.x + r * cos(angle);
        pos1.y = center.y + r * sin(angle);
        AddLine(ctx, pos0, pos1, w, col);
        pos0 = pos1;
    }
}

constexpr float kPi = 3.1415;
void AddFilledCirle(RenderingContext *ctx, Vec2f center, float r, Color col, int n_segments = 12)
{
    uint32_t first_id = ctx->inds.size();
    uint32_t first_v  = ctx->verts.size();

    ctx->verts.reserve(first_v + n_segments + 1);
    ctx->verts.push_back({.pos=center, .col=col});
    float seg_angle = 2*kPi / n_segments;

    Vec2f pos;
    for (int seg = 0; seg < n_segments; ++seg) {
        float angle = seg * seg_angle;
        pos.x = center.x + r * cos(angle);
        pos.y = center.y + r * sin(angle);
        ctx->verts.push_back({.pos=pos, .col=col});
    }

    ctx->inds.reserve(first_id + n_segments * 3);
    for (int seg = 0; seg < n_segments - 1; ++seg) {
        ctx->inds.push_back(first_v);            // center
        ctx->inds.push_back(first_v + seg + 1);  // seg start
        ctx->inds.push_back(first_v + seg + 2);  // seg end
    }

    // Last triangle
    ctx->inds.push_back(first_v);               // center
    ctx->inds.push_back(first_v + n_segments);  // seg start
    ctx->inds.push_back(first_v + 1);           // seg end
}


template <typename T, size_t BufferSize = 50>
struct CircularBuffer {
    using DataType = T;

    static constexpr size_t kBufferSize = BufferSize;

    T buffer[kBufferSize] = {0};

    size_t ptr_end   = 0;
    size_t ptr_start = 0;
    size_t size      = 0;

    inline void Push(const T &x)
    {
        buffer[ptr_end] = x;
        ptr_end = (ptr_end + 1) % kBufferSize;

        if (size == kBufferSize) {
            ptr_start = (ptr_start + 1) % kBufferSize;
        } else {
            ++size;
        }
    }

    inline size_t MaxSize()
    {
        return kBufferSize;
    }

    inline size_t Size()
    {
        return size;
    }

    inline T &Back()
    {
        return buffer[(ptr_end - 1) % kBufferSize];
    }

    inline T &operator[](size_t i)
    {
        return buffer[(ptr_start + i) % kBufferSize];
    }
};


struct TrackedObject {
    Vec2f position;

    CircularBuffer<Vec2f, 50> history;
};

void DrawTrackedObject(RenderingContext *ctx, TrackedObject *obj, Vec2f new_pos)
{
    AddFilledCirle(ctx, new_pos, 0.015f, ColorYellow);

    obj->position = new_pos;
    Vec2f pos1 = new_pos;
    Vec2f pos0 = {0};
    Color traj_col = ColorBlack;
    size_t max_sz = obj->history.MaxSize();
    for (int i = (int)obj->history.Size() - 1; i >= 0; --i) {
        pos0 = obj->history[i];
        traj_col.a = (uint8_t)((float)i / max_sz * 255.0f);
        AddLine(ctx, pos1, pos0, 0.003f, traj_col);
        pos1 = pos0;
    }

    Vec2f diff = new_pos - obj->history.Back();
    if (diff.x * diff.x + diff.y * diff.y > 0.01f * 0.01f) {
        obj->history.Push(new_pos);
    }
}

void ToggleWireframeMode()
{
    static bool is_toggled = false;

    is_toggled = !is_toggled;
    if (is_toggled) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}

void InputKeyCallback(
    GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_W) {
        if (action == GLFW_RELEASE) {
            ToggleWireframeMode();
        }
    }
    if (key == GLFW_KEY_SPACE) {
        if (action == GLFW_RELEASE) {
            if (!frame_clock.is_paused) {
                frame_clock.Pause();
            } else {
                frame_clock.UnPause();
            }
        }
    }

    static double spfs[] = {1.0, 0.5, 0.1, 0.01, 0.001, 0.0001};
    static int cursor = 4;
    if (key == GLFW_KEY_UP) {
        if (action == GLFW_RELEASE) {
            cursor = min(cursor + 1, (int)(sizeof(spfs) / sizeof(spfs[0])) - 1);
            frame_clock.SetSecondsPerFrame(spfs[cursor]);
        }
    } else if (key == GLFW_KEY_DOWN) {
        if (action == GLFW_RELEASE) {
            cursor = max(cursor - 1, 0);
            frame_clock.SetSecondsPerFrame(spfs[cursor]);
        }
    }
}

struct Trajectory {
    std::vector<Vec2f>    points;
    // std::vector<uint32_t> timestamps;
};

Trajectory LinearTrajectory(Vec2f p0, Vec2f v0, Vec2f a, float duration)
{
    Trajectory traj;
    constexpr float dt = 0.001;  // 1 ms
    constexpr float dt_sq = dt * dt;
    float t = 0;
    traj.points.reserve(duration / dt + 1);
    Vec2f p_i = p0;
    Vec2f v_i = v0;
    while (t < duration) {
        v_i += a * dt;
        p_i += v_i * dt + a * (dt_sq / 2.0f);

        if (p_i.x < -0.45 || p_i.x > 0.45) {
            a.x = -a.x;
            v_i.x = -v_i.x;
        }
        if (p_i.y < -0.45 || p_i.y > 0.45) {
            a.y = -a.y;
            v_i.y = -v_i.y;
        }

        t += dt;
        traj.points.push_back(p_i);
    }
    return traj;
}

namespace kalman2d {

    using EigMat44f   = Eigen::Matrix<float, 4, 4>;
    using EigMat42f   = Eigen::Matrix<float, 4, 2>;
    using EigMat24f   = Eigen::Matrix<float, 2, 4>;
    using EigMat22f   = Eigen::Matrix<float, 2, 2>;
    using EigVec4f    = Eigen::Matrix<float, 4, 1>;
    using EigVec2fRef = Eigen::Map<Eigen::Matrix<float, 2, 1>>;

    inline EigMat44f ComputeF(float dt)
    {
        EigMat44f F;
        F << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;
        return F;
    }

    inline EigMat44f ComputeQ(float dt)
    {
        EigMat42f Ga;
        Ga << dt * dt / 2, 0,
              0, dt * dt / 2,
              dt, 0,
              0, dt;
        EigMat22f Cov_a;
        Cov_a << 1.1 * 1.1 / 100, 0,
                 0, 1.1 * 1.1 / 100;
        return Ga * Cov_a * Ga.transpose();
    }

    struct State {
        EigVec4f  x;
        EigMat44f P;
        float last_t;
    };

    EigMat24f H = (EigMat24f() <<
            1, 0, 0, 0,
            0, 1, 0, 0
        ).finished();

    EigMat22f R = (EigMat22f() <<
            0.001,   0,
            0,   0.001
        ).finished();

    constexpr float kSigmaX  = 0.1f;
    constexpr float kSigmaY  = 0.1f;
    constexpr float kSigmaVX = 2.0f;
    constexpr float kSigmaVY = 2.0f;

    constexpr float kGatingMultiplierX = 1.0f;
    constexpr float kGatingMultiplierY = 1.0f;
    constexpr float kMinGatingSigmaX   = 0.05f;
    constexpr float kMinGatingSigmaY   = 0.05f;

    State CreateFilter(Vec2f position0, float t0)
    {
        State s = {};
        s.x << position0.x, position0.y, 0, 0;
        s.P << kSigmaX*kSigmaX, 0, 0, 0,
               0, kSigmaY*kSigmaY, 0, 0,
               0, 0, kSigmaVX*kSigmaVX, 0,
               0, 0, 0, kSigmaVY*kSigmaVY;
        s.last_t = t0;
        return s;
    }

    State Predict(State *s, float t)
    {
        float dt = t - s->last_t;
        const auto &F = ComputeF(dt);
        State state_new;
        state_new.x = F * s->x;
        state_new.P = F * s->P * F.transpose() + ComputeQ(dt);
        return state_new;
    }

    void PredictAndUpdate(State *s, float t)
    {
        float dt = t - s->last_t;
        const auto &F = ComputeF(dt);
        s->x = F * s->x;
        s->P = F * s->P * F.transpose() + ComputeQ(dt);
        s->last_t = t;
    }

    void UpdateFromMeasurement(State *s, EigVec2fRef z)
    {
        const auto& S = H * s->P * H.transpose() + R;
        const auto& y = z - H * s->x;
        const auto& K = s->P * H.transpose() * S.inverse();
        const auto& A = EigMat44f::Identity() - K * H;
        s->x = s->x + K * y;
        // Using Joseph's form update for stability.
        s->P = A * s->P * A.transpose() + K * R * K.transpose();
    }

    inline void Update(State *s, Vec2f meas)
    {
        UpdateFromMeasurement(s, EigVec2fRef((float *)&meas));
    }

    struct GatingWindow {
        float x0, y0, x1, y1;
    };

    GatingWindow ComputeGatingWindow(State *s, float t)
    {
        float dt = t - s->last_t;
        State next_s = Predict(s, t);
        float pred_sigma_x  = sqrtf(next_s.P(0, 0));
        float pred_sigma_y  = sqrtf(next_s.P(1, 1));
        float pred_sigma_vx = sqrtf(next_s.P(2, 2));
        float pred_sigma_vy = sqrtf(next_s.P(3, 3));

        float gating_coef_x = max(pred_sigma_x, kMinGatingSigmaX)
            + dt * pred_sigma_vx;
        float gating_coef_y = max(pred_sigma_y, kMinGatingSigmaY)
            + dt * pred_sigma_vy;

        GatingWindow w;
        w.x0 = next_s.x(0) - kGatingMultiplierX * gating_coef_x;
        w.y0 = next_s.x(1) - kGatingMultiplierY * gating_coef_y;
        w.x1 = next_s.x(0) + kGatingMultiplierX * gating_coef_x;
        w.y1 = next_s.x(1) + kGatingMultiplierY * gating_coef_y;
        return w;
    }

    inline GatingWindow ComputeGatingWindow(State *s)
    {
        return ComputeGatingWindow(s, s->last_t);
    }

}  // namespace kalman2d

int main(void)
{
    if (!glfwInit()) {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
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

    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

    RenderingContext ctx = CreateRenderingContext();

    double last_time = glfwGetTime();
    int num_frames = 0;

    Trajectory traj1 = LinearTrajectory(Vec2f{0, 0}, Vec2f{0.10, 0.1}, Vec2f{-0.05, -0.01}, 20.0f);
    Trajectory traj2 = LinearTrajectory(Vec2f{0, 0}, Vec2f{0.15, 0.1}, Vec2f{ 0.15, -0.12}, 20.0f);
    Trajectory traj3 = LinearTrajectory(Vec2f{0, 0}, Vec2f{0.11, 0.0}, Vec2f{-0.24,  0.20}, 20.0f);
    TrackedObject obj1, obj2, obj3;
    Vec2f meas1;

    glfwSetKeyCallback(window, InputKeyCallback);

    Vec2f g_offset = {0.5f, 0.5f};
    int64_t t   = 0;
    float   t_s = frame_clock.Seconds();

    kalman2d::State s = kalman2d::CreateFilter(Vec2f{0.0f, 0.0f}, t);

    for (;;) {
        t = frame_clock.Frame();
        t_s = frame_clock.Seconds();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE ) == GLFW_PRESS
                || glfwWindowShouldClose(window) != 0) {
            break;
        }

        // Measure speed
        double cur_time= glfwGetTime();
        ++num_frames;
        if (cur_time - last_time >= 1.0) {
            printf("%f ms/frame   fps:%d\n", 1000.0/double(num_frames), num_frames);
            num_frames = 0;
            last_time += 1.0;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        SetupRenderingContext(&ctx);
        AddRect(&ctx, {0.05f, 0.05f}, {0.9f, 0.9f}, 0.004f, ColorBlack);
        Vec2f meas = traj1.points[t % traj1.points.size()];

        DrawTrackedObject(&ctx, &obj1, meas + g_offset);
        // DrawTrackedObject(&ctx, &obj2, traj2.points[t % traj2.points.size()] + g_offset);
        // DrawTrackedObject(&ctx, &obj3, traj3.points[t % traj3.points.size()] + g_offset);

        kalman2d::GatingWindow w0 = kalman2d::ComputeGatingWindow(&s);
        kalman2d::GatingWindow w = kalman2d::ComputeGatingWindow(&s, t_s);

        AddRect(
            &ctx, Vec2f{w0.x0, w0.y0} + g_offset,
            Vec2f{w0.x1, w0.y1} - Vec2f{w0.x0, w0.y0}, 0.004f, ColorGray);
        AddLine(
            &ctx, Vec2f{(w0.x0 + w0.x1) / 2, (w0.y0 + w0.y1) / 2} + g_offset,
            Vec2f{(w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2} + g_offset, 0.004f,
            ColorGray);
        AddRect(
            &ctx, Vec2f{w.x0, w.y0} + g_offset,
            Vec2f{w.x1, w.y1} - Vec2f{w.x0, w.y0}, 0.004f, ColorBlack);

        if (w.x0 <= meas.x && meas.x <= w.x1 && w.y0 <= meas.y && meas.y <= w.y1) {
            static int t_last = t;
            static int i = 0;
            if (t != t_last) {
                 ++i;
                if (i % 10 == 0) {
                    AddFilledCirle(&ctx, meas + g_offset, 0.013, ColorRed, 12);
                    kalman2d::PredictAndUpdate(&s, t_s);
                    kalman2d::Update(&s, meas);
                }
            }
        }

        Render(&ctx);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        if (frame_clock.dt_s == 0.0) {
            frame_clock.SetSecondsPerFrame(0.001).Reset();
        }
    }

    glfwTerminate();

    return 0;
}

// Render to texture. <<<
// Real benchmark.
// Viewports
// Grid angular
// Text
