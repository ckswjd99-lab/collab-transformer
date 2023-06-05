#include <semaphore.h>
#include <stdio.h>
#include <cblas.h>


#define SPLIT_NUM   12

#define MULT_M      768
#define MULT_K1     768
#define MULT_K2     (768*4)
#define MULT_N      768

#define MULT_SUBM   (MULT_M / SPLIT_NUM)

class Task {

private:
    int layer;
    int op_idx;
    Task *next;

public:
    Task(int layer, int op_idx);
    ~Task();

    int get_layer();
    int get_op_idx();
    Task *get_next();
    void set_next(Task *task);
    int same(int layer, int op_idx);
    void compute();
};

class TaskQueue {

private:
    Task *head;
    sem_t sem;

public:
    TaskQueue(int seed);
    ~TaskQueue();

    Task *find(int layer, int op_idx);
    void push_back(Task *task);
    void push_front(Task *task);
    Task *pop_front();
    void print();
};