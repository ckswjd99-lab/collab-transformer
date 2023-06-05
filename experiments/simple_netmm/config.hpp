/**
 * This is hard-coded toy example!
*/
#ifndef __CONFIG_HPP
#define __CONFIG_HPP

#include <semaphore.h>
#include <cblas.h>
#include <vector>
#include <pthread.h>
#include <stdlib.h>
#include <algorithm>
#include <stdio.h>

#define __DEBUG     1

#define RAND_SEED   42
#define WORKER_NUM  4

#define DEVICE_NUM  2
#define SPLIT_NUM   12

#define TOKEN_NUM   128
#define EMBED_CHAN  768
#define HEAD_NUM    12

#define DUMMY_M     128
#define DUMMY_N     (768*4)
#define DUMMY_K     768

enum MatrixState {
    MAT_EMPTY,
    MAT_COMPUTING,
    MAT_RECEIVING,
    MAT_FILLED
};

enum EncoderLayer {
    ENCO_QKV,
    ENCO_MHA,
    ENCO_ADDNORM1,
    ENCO_FFN1,
    ENCO_FFN2,
    ENCO_ADDNORM2,
    ENCO_DUMMY
};

class Matrix {

public:
    Matrix(EncoderLayer layer, int idx)
     : layer(layer), idx(idx) { };
    
    EncoderLayer layer;
    int idx;

};

#endif