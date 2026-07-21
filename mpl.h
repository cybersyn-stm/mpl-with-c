#ifndef MPL_H
#define MPL_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define ALLOC_OR_FAIL(ptr, n, type, label) \
    do {                                   \
        (ptr) = calloc((n), sizeof(type)); \
        if (!(ptr))                        \
            goto label;                    \
    } while (0)

/* ===== 网络超参数 ===== */

#define WIDTH 20
#define HEIGHT 20

/* ===== 训练超参数 ===== */

#define TRAIN_PASSES 500000
#define TRAIN_SEED 42
#define LEARNING_RATE 0.001f

/* ===== 类型定义 ===== */

typedef float Image[HEIGHT][WIDTH];

typedef struct {
    int in_dim, out_dim;
    float *weights, *bias, *z, *a, *gradient;
} Layer;

typedef struct {
    int num_layers;
    Layer **layers;
} MLP;

/* ===== 内联工具函数 ===== */

static inline float rand_norm(void) {
    float u1 = (float)rand() / RAND_MAX + 1e-7f;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static inline int rand_range(int low, int high) {
    assert(low < high);
    return rand() % (high - low) + low;
}

static inline int clampi(int x, int low, int high) {
    if (x < low)
        x = low;
    if (x > high)
        x = high;
    return x;
}

/* ===== 激活函数 ===== */

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float derivative_relu(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float derivative_sigmoid(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

/* ===== 损失函数 ===== */

static inline float bce_loss(float y_hat, float y) {
    float eps = 1e-7f;
    y_hat = fmaxf(eps, fminf(1.0f - eps, y_hat));
    return -(y * logf(y_hat) + (1.0f - y) * logf(1.0f - y_hat));
}

static inline float bce_loss_vec(const float *y_hat, const float *y, int dim) {
    float loss = 0.0f, eps = 1e-7f;
    for (int i = 0; i < dim; i++) {
        float p = fmaxf(eps, fminf(1.0f - eps, y_hat[i]));
        loss += -(y[i] * logf(p) + (1.0f - y[i]) * logf(1.0f - p));
    }
    return loss;
}

static inline void output_gradient(Layer *output_layer, const float *label) {
    for (int i = 0; i < output_layer->out_dim; i++)
        output_layer->gradient[i] = output_layer->a[i] - label[i];
}

/* ===== 函数声明 ===== */

int clampi(int x, int low, int high);
int ensure_dir_exist(const char *dir_path);

void layer_fill_rect(Image layer, int x, int y, int w, int h, float value);
void layer_fill_circle(Image layer, int cx, int cy, int r, float value);

void create_random_rect(Image inputs);
void create_random_circle(Image inputs);

Layer *layer_malloc(int in_dim, int out_dim);
void layer_init_he(Layer *layer);
void layer_free(Layer *layer);

float *layer_forward_prop(Layer *layer, float *inputs, float (*active)(float));
void layer_back_prop(Layer *layer, Layer *next_layer, float (*derivative)(float));
void layer_update(Layer *layer, float *input, float lr);

MLP *mlp_create(int layer_dim[], int dim);
void mlp_free(MLP *net);

void train(MLP net);
void test(MLP net);
void save_model(MLP net, const char *file_path);

#endif /* MPL_H */
