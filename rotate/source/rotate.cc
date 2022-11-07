#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Image
{
    int32_t width;
    int32_t height;
    int32_t depth;

    uint8_t *data;
};

Image LoadImage(const char *filename)
{
    Image img;
    img.data = stbi_load(filename, &img.width, &img.height, &img.depth, 4);
    img.depth = 4;
    if (!img.data) {
        printf("Can't load image: %s.\n", filename);
        exit(1);
    }
    if (img.depth != 4 && img.depth != 3) {
        printf("Only 4/3-channel images are supported. Found: %d.\n", img.depth);
        exit(1);
    }
    return img;
}

struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

enum class InterType {
    NONE,
    NEAREST,
    BILINEAR
};

struct RotationContext {
    int32_t canvas_width;
    int32_t canvas_height;
    int32_t canvas_linesz;
    uint8_t *buffer;

    Image *img;
};

RotationContext FromImage(Image *img)
{
    assert(img->depth == 4 || img->depth == 3);
    int32_t diag = (int)ceil(sqrtf(img->width * img->width + img->height * img->height));
    uint8_t *canvas_data = (uint8_t *)aligned_alloc(64, diag * diag * img->depth);
    memset(canvas_data, 0, img->width * img->height * img->depth);
    return RotationContext{
        .canvas_width = diag,
        .canvas_height = diag,
        .canvas_linesz = diag * img->depth,
        .buffer = canvas_data,
        .img = img,
    };
}

Pixel Pick(Image *img, int32_t x, int32_t y, float *coef)
{
    if ((x >= img->width) || (x < 0) || (y >= img->height) || (y < 0)) {
        *coef = 0;
        return Pixel{0,0,0,0};
    }
    *coef = 1;
    uint8_t *pix = img->data + y * img->width * img->depth + x * img->depth;

    return *(Pixel *)pix;
}

Pixel Sample(Image *img, float x, float y, InterType inter = InterType::NEAREST)
{
    int32_t floor_x = (int32_t)x;
    int32_t floor_y = (int32_t)y;
    float frac_x = x - floor_x;
    float frac_y = y - floor_y;

    switch (inter) {
        case InterType::NONE: {
            int32_t inter_x = floor_x;
            int32_t inter_y = floor_y;
            float coef;
            return Pick(img, inter_x, inter_y, &coef);
        } break;

        case InterType::NEAREST: {
            int32_t inter_x = floor_x + round(frac_x);
            int32_t inter_y = floor_y + round(frac_y);
            float coef;
            return Pick(img, inter_x, inter_y, &coef);
        } break;

        case InterType::BILINEAR: {
            int32_t inter_x0 = floor_x;
            int32_t inter_y0 = floor_y;
            int32_t inter_x1 = floor_x + 1;
            int32_t inter_y1 = floor_y + 1;

            float c00, c01, c10, c11;
            Pixel pix00 = Pick(img, inter_x0, inter_y0, &c00);
            Pixel pix01 = Pick(img, inter_x0, inter_y1, &c01);
            Pixel pix10 = Pick(img, inter_x1, inter_y0, &c10);
            Pixel pix11 = Pick(img, inter_x1, inter_y1, &c11);

            float w00 = c00 * (1 - frac_x) * (1 - frac_y);
            float w01 = c01 * (1 - frac_x) * frac_y;
            float w10 = c10 * frac_x * (1 - frac_y);
            float w11 = c11 * frac_x * frac_y;

            float s = w00 + w01 + w10 + w11;

            w00 /= s;
            w01 /= s;
            w10 /= s;
            w11 /= s;

            return Pixel{
                .r = uint8_t(pix00.r * w00 + pix01.r * w01 + pix10.r * w10 + pix11.r * w11),
                .g = uint8_t(pix00.g * w00 + pix01.g * w01 + pix10.g * w10 + pix11.g * w11),
                .b = uint8_t(pix00.b * w00 + pix01.b * w01 + pix10.b * w10 + pix11.b * w11),
                .a = uint8_t(pix00.a * w00 + pix01.a * w01 + pix10.a * w10 + pix11.a * w11)
            };
        } break;
        default: {
            assert(0);
        } break;
    }

}

void Rotate(RotationContext *ctx, float angle_rad, InterType inter)
{
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);

    float inv_rot_mat[2][2] = {{ cos_a, sin_a},
                               {-sin_a, cos_a}};
    for (int32_t x = 0; x < ctx->canvas_width; ++x) {
        for (int32_t y = 0; y < ctx->canvas_height; ++y) {
            // [0, D] -> [0, +1] -> [-0.5, 0.5]
            float x_ = x / (float)ctx->canvas_width  - 0.5f;
            float y_ = y / (float)ctx->canvas_height - 0.5f;

            float tex_x_ = inv_rot_mat[0][0] * x_ + inv_rot_mat[0][1] * y_;
            float tex_y_ = inv_rot_mat[1][0] * x_ + inv_rot_mat[1][1] * y_;

            // To texture space. [-0.5, 0.5] -> [0, +1]
            float tex_x = tex_x_ * (float)ctx->canvas_width  + 0.5f * ctx->img->width;
            float tex_y = tex_y_ * (float)ctx->canvas_height + 0.5f * ctx->img->height;

            uint8_t *pix = ctx->buffer + y * ctx->canvas_linesz + x * ctx->img->depth;
            *(Pixel *)pix = Sample(ctx->img, tex_x, tex_y, inter);
        }
    }
}

struct Profile {
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

    Profile(const char *name)
        : m_name(name)
    {
        m_start_time = clock::now();
    }

    ~Profile()
    {
        using std::chrono::duration_cast;
        using std::chrono::microseconds;

        auto total_us = duration_cast<microseconds>(clock::now() - m_start_time).count();

        int us = total_us % 1000;
        int ms = total_us / 1000;

        printf("%s time: %d ms. %d us.\n", m_name, ms, us);
    }

    const char *m_name;
    time_point  m_start_time;
};

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s /path/to/image.png <angle_deg>\n", argv[0]);
        return 1;
    }
    char *path = argv[1];
    float deg  = atof(argv[2]);

    Image img = LoadImage(path);
    RotationContext ctx = FromImage(&img);
    float a_rad = deg / 180.0f * M_PI;
    {
        Profile p("Nearest");
        Rotate(&ctx, a_rad, InterType::NEAREST);
    }
    stbi_write_png("out_nearest.png", ctx.canvas_width, ctx.canvas_height, img.depth, ctx.buffer, ctx.canvas_linesz);

    {
        Profile p("Bilinear");
        Rotate(&ctx, a_rad, InterType::BILINEAR);
    }
    stbi_write_png("out_bilinear.png", ctx.canvas_width, ctx.canvas_height, img.depth, ctx.buffer, ctx.canvas_linesz);

    // stbi_image_free(img.data);
}
