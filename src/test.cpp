#include "config.hpp"
#include "task.hpp"

float embedded  [TOKEN_NUM * EMBED_CHAN];

float W_Q       [EMBED_CHAN * EMBED_CHAN];
float W_K       [EMBED_CHAN * EMBED_CHAN];
float W_V       [EMBED_CHAN * EMBED_CHAN];

float Q         [TOKEN_NUM * EMBED_CHAN];
float K         [TOKEN_NUM * EMBED_CHAN];
float V         [TOKEN_NUM * EMBED_CHAN];
float QK_T      [TOKEN_NUM * TOKEN_NUM * HEAD_NUM];

float SHA       [TOKEN_NUM * EMBED_CHAN];
float W_O       [EMBED_CHAN * EMBED_CHAN];
float MHA       [TOKEN_NUM * EMBED_CHAN];

float W_FFN1    [EMBED_CHAN * EMBED_CHAN * 4];
float FFN1      [TOKEN_NUM * EMBED_CHAN * 4];
float W_FFN2    [EMBED_CHAN * 4 * EMBED_CHAN];
float FFN2      [TOKEN_NUM * EMBED_CHAN];

void init_data_rand() {

    for(int i=0; i<TOKEN_NUM*EMBED_CHAN; i++) embedded[i] = rand();

    for(int i=0; i<EMBED_CHAN * EMBED_CHAN; i++) {
        W_Q[i] = rand();
        W_K[i] = rand();
        W_V[i] = rand();
        W_O[i] = rand();
    }

    for(int i=0; i<EMBED_CHAN * EMBED_CHAN * 4; i++) {
        W_FFN1[i] = rand();
        W_FFN2[i] = rand();
    }
    
}

MMTask<float> QKVTasks[DEVICE_NUM * SPLIT_NUM];
MMTask<float> SATasks[DEVICE_NUM * DEVICE_NUM * HEAD_NUM];
MMTask<float> SHATasks[DEVICE_NUM * DEVICE_NUM * HEAD_NUM];

