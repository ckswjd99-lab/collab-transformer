#ifndef __TASK_HPP
#define __TASK_HPP

#include <semaphore.h>
#include <vector>
#include <cstdarg>
#include <stdlib.h>
#include <cblas.h>

enum TaskState {
    TASK_EMPTY,
    TASK_COMPUTING,
    TASK_RECEIVING,
    TASK_FILLED
};

enum Activation {
    ACTI_NONE,
    ACTI_RELU,
    ACTI_GELU
};

template <typename T>
class Task {

private:
    static int task_count = 0;
    
    int task_id;

    TaskState prefered_state;
    TaskState state;
    sem_t state_sem;
    
    int requirements;
    sem_t req_sem;

    vector<Task<T> *> providings;

    T *data;
    int row;
    int col;
    int incRow;    

    int is_data_dynamic;

public:
    Task(int req_num, TaskState prefered_state, T* data, int row, int col, int incRow);
    ~Task();

    int get_task_id() { return this->task_id; }

    TaskState try_fetch(TaskState new_state);
    
    int decrease_req();

    Task<T> *push_providings(Task<T> *first, Task<T> *...);
    
    T *get_data();
    int get_row();
    int get_col();
    int get_incRow();

    void compute();
    void receive(T *data);

};


template <typename T>
class MMTask : public Task {

private:
    int M, N, K;

    Task *A, *B;

    T *dataA, *dataB;

    int incRowA, incRowB;

    float alpha, beta;

    Activation activation;

public:
    MMTask(int req_num, TaskState prefered_state, float alpha, Task<T> *A, Task<T> *B, float beta, T *C, int incRowC, Activation acti);
    ~MMTask() {};

    void compute();
};

#endif