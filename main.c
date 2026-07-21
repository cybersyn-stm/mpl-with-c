#include "mpl.h"
#include <unistd.h>

/* ===== 目录工具 ===== */

int ensure_dir_exist(const char *dir_path) {
    if (strlen(dir_path) >= 255) {
        fprintf(stderr, "[ERROR] directory path is too long\n");
        return -1;
    }
    char path[256], mk_path[256];
    int slash_num = 0, i;
    for (i = 0; dir_path[i] != '\0'; ++i) {
        path[i] = dir_path[i];
        if (path[i] == '/')
            slash_num++;
    }
    path[i] = '\0';
    if (dir_path[0] == '/' || dir_path[0] == '.') {
        fprintf(stderr, "[ERROR] directory must be a relative path without leading '.' or '/'\n");
        return -1;
    }
    for (int j = 0; j < slash_num; j++) {
        int slash_idx = 0, len = 0;
        while (slash_idx <= j) {
            if (path[len] == '/')
                slash_idx++;
            len++;
        }
        memcpy(mk_path, path, (size_t)len);
        mk_path[len] = '\0';
        if (access(mk_path, F_OK) == -1) {
            if (mkdir(mk_path, 0777) == -1) {
                fprintf(stderr, "[ERROR] failed to create directory %s\n", mk_path);
                return -1;
            }
        }
    }
    return 0;
}

/* ===== 图形绘制 ===== */

void layer_fill_rect(Image layer, int x, int y, int w, int h, float value) {
    assert(w > 0);
    assert(h > 0);
    int x0 = clampi(x, 0, WIDTH - 1);
    int y0 = clampi(y, 0, HEIGHT - 1);
    int x1 = clampi(x0 + w - 1, 0, WIDTH - 1);
    int y1 = clampi(y0 + h - 1, 0, HEIGHT - 1);
    for (int i = y0; i <= y1; i++)
        for (int j = x0; j <= x1; j++)
            layer[i][j] = value;
}

void layer_fill_circle(Image layer, int cx, int cy, int r, float value) {
    assert(r > 0);
    int x0 = clampi(cx - r, 0, WIDTH - 1);
    int y0 = clampi(cy - r, 0, HEIGHT - 1);
    int x1 = clampi(cx + r, 0, WIDTH - 1);
    int y1 = clampi(cy + r, 0, HEIGHT - 1);
    for (int x = x0; x <= x1; x++)
        for (int y = y0; y <= y1; y++) {
            int dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy <= r * r)
                layer[y][x] = value;
        }
}

/* ===== 数据生成 ===== */

void create_random_rect(Image inputs) {
    layer_fill_rect(inputs, 0, 0, WIDTH, HEIGHT, 0.0f);
    int x = rand_range(0, WIDTH), y = rand_range(0, HEIGHT);
    int w = WIDTH - x;
    if (w < 2)
        w = 2;
    w = rand_range(1, w);
    int h = HEIGHT - y;
    if (h < 2)
        h = 2;
    h = rand_range(1, h);
    layer_fill_rect(inputs, x, y, w, h, 1.0f);
}

void create_random_circle(Image inputs) {
    layer_fill_rect(inputs, 0, 0, WIDTH, HEIGHT, 0.0f);
    int cx = rand_range(0, WIDTH), cy = rand_range(0, HEIGHT);
    int r = WIDTH / 2;
    if (r > cx)
        r = cx;
    if (r > WIDTH - cx)
        r = WIDTH - cx;
    if (r > cy)
        r = cy;
    if (r > HEIGHT - cy)
        r = HEIGHT - cy;
    if (r < 2)
        r = 2;
    r = rand_range(1, r);
    layer_fill_circle(inputs, cx, cy, r, 1.0f);
}

Layer *layer_malloc(int in_dim, int out_dim) {
    if (in_dim <= 0 || out_dim <= 0) {
        fprintf(stderr, "[ERROR] layer_malloc: in_dim and out_dim must be positive\n");
        return NULL;
    }
    // alloc Layer memories and parameters
    Layer *layer = malloc(sizeof(Layer));
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    ALLOC_OR_FAIL(layer->weights, in_dim * out_dim, float, fail);
    ALLOC_OR_FAIL(layer->bias, out_dim, float, fail);
    ALLOC_OR_FAIL(layer->z, out_dim, float, fail);
    ALLOC_OR_FAIL(layer->a, out_dim, float, fail);
    ALLOC_OR_FAIL(layer->gradient, out_dim, float, fail);
    return layer;
// error handling
fail:
    free(layer);
    return NULL;
}

void layer_init_he(Layer *layer) {
    float std = sqrtf(2.0f / layer->in_dim);
    for (int i = 0; i < layer->in_dim * layer->out_dim; i++)
        layer->weights[i] = rand_norm() * std;
    /* bias 已由 calloc 置零，无需额外操作 */
}

void layer_free(Layer *layer) {
    if (layer == NULL)
        return;
    free(layer->weights);
    free(layer->bias);
    free(layer->z);
    free(layer->a);
    free(layer->gradient);
    free(layer);
}

float *layer_forward_prop(Layer *layer, float *inputs, float (*active)(float)) {
    if (layer == NULL || inputs == NULL || active == NULL) {
        fprintf(stderr, "[ERROR] layer_forward_prop: layer, inputs and active must not be NULL\n");
        return NULL;
    }
    int o_dim = layer->out_dim;
    int i_dim = layer->in_dim;

    for (int i = 0; i < o_dim; i++) {
        layer->z[i] = 0.0f;
        for (int j = 0; j < i_dim; j++) {
            layer->z[i] += layer->weights[i * i_dim + j] * inputs[j];
        }
        layer->z[i] += layer->bias[i];
        // active_function
        layer->a[i] = active(layer->z[i]);
    }
    return layer->a;
}

void layer_back_prop(Layer *layer, Layer *next_layer, float (*derivative)(float)) {
    int n = next_layer->in_dim;  // 本层神经元数 = 下层输入维
    int m = next_layer->out_dim; // 下层神经元数

    for (int i = 0; i < n; i++) {
        layer->gradient[i] = 0.0f;
        for (int j = 0; j < m; j++)
            layer->gradient[i] += next_layer->gradient[j] * next_layer->weights[j * n + i];
        layer->gradient[i] *= derivative(layer->z[i]);
    }
}

void layer_update(Layer *layer, float *input, float lr) {
    for (int i = 0; i < layer->out_dim; i++) {
        layer->bias[i] -= lr * layer->gradient[i];
        for (int j = 0; j < layer->in_dim; j++)
            layer->weights[i * layer->in_dim + j] -= lr * layer->gradient[i] * input[j];
    }
}
void mlp_free(MLP *net) {
    if (net == NULL)
        return;
    for (int i = 0; i < net->num_layers; i++) {
        layer_free(net->layers[i]);
    }
    free(net->layers);
    free(net);
}

MLP *mlp_create(int layer_dim[], int dim) {
    MLP *net = malloc(sizeof(MLP));
    if (!net)
        return NULL;
    net->num_layers = dim - 1;
    ALLOC_OR_FAIL(net->layers, net->num_layers, Layer *, fail_net);
    for (int i = 0; i < net->num_layers; i++) {
        if ((*(net->layers + i) = layer_malloc(layer_dim[i], layer_dim[i + 1])) == NULL)
            goto fail;
        layer_init_he(*(net->layers + i));
    }
    return net;
fail:
    mlp_free(net);
    return NULL;
fail_net:
    free(net);

    fprintf(stderr, "[ERROR] mlp_create: failed to create MLP\n");
    return NULL;
}

void train(MLP net) {
    float inputs[HEIGHT][WIDTH], loss_sum = 0.0f;
    int label;
    for (int i = 0; i < TRAIN_PASSES; i++) {
        if ((label = rand() % 2)) {
            create_random_circle(inputs);
        } else {
            create_random_rect(inputs);
        }
        layer_forward_prop(net.layers[0], *inputs, relu);
        for (int j = 1; j < net.num_layers - 1; j++)
            layer_forward_prop(net.layers[j], net.layers[j - 1]->a, relu);
        layer_forward_prop(net.layers[net.num_layers - 1], net.layers[net.num_layers - 2]->a, sigmoid);
        net.layers[net.num_layers - 1]->gradient[0] = net.layers[net.num_layers - 1]->a[0] - (float)label;
        float pred = net.layers[net.num_layers - 1]->a[0];
        loss_sum += bce_loss(pred, (float)label);
        for (int j = net.num_layers - 1; j > 0; j--)
            layer_back_prop(net.layers[j - 1], net.layers[j], derivative_relu);
        for (int j = 0; j < net.num_layers; j++)
            layer_update(net.layers[j], j == 0 ? *inputs : net.layers[j - 1]->a, LEARNING_RATE);
        if ((i + 1) % 50000 == 0)
            printf("pass %d/%d  avg_loss=%.6f\n", i + 1, TRAIN_PASSES, loss_sum / 50000.0f), loss_sum = 0.0f;
    }
    printf("Training completed after %d passes\n", TRAIN_PASSES);
}
void test(MLP net) {
    float inputs[HEIGHT][WIDTH];
    int label;
    int correct = 0;
    for (int i = 0; i < 1000; i++) {
        if ((label = rand() % 2)) {
            create_random_circle(inputs);
        } else {
            create_random_rect(inputs);
        }
        layer_forward_prop(net.layers[0], *inputs, relu);
        for (int j = 1; j < net.num_layers - 1; j++) {
            layer_forward_prop(net.layers[j], net.layers[j - 1]->a, relu);
        }
        layer_forward_prop(net.layers[net.num_layers - 1], net.layers[net.num_layers - 2]->a, sigmoid);
        int prediction = net.layers[net.num_layers - 1]->a[0] > 0.5f ? 1 : 0;
        if (prediction == label)
            correct++;
    }
    printf("Testing completed after %d passes\n", 1000);
    printf("Accuracy: %.2f%%\n", (float)correct / 1000 * 100.0f);
}
void save_model(MLP net, const char *file_path) {
    FILE *fp = fopen(file_path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "[ERROR] save_model: failed to open file %s\n", file_path);
        return;
    }
    fwrite(&net.num_layers, sizeof(int), 1, fp);
    for (int i = 0; i < net.num_layers; i++) {
        Layer *layer = net.layers[i];
        fwrite(&layer->in_dim, sizeof(int), 1, fp);
        fwrite(&layer->out_dim, sizeof(int), 1, fp);
        fwrite(layer->weights, sizeof(float), layer->in_dim * layer->out_dim, fp);
        fwrite(layer->bias, sizeof(float), layer->out_dim, fp);
    }
    fclose(fp);
}
int main() {
    MLP *net;
    int dims[] = {400, 128, 64, 1};
    net = mlp_create(dims, 4);
    srand(TRAIN_SEED);
    printf("MLP created with %d layers\n", net->num_layers);
    train(*net);
    save_model(*net, "model.dat");
    test(*net);
    mlp_free(net);
}
