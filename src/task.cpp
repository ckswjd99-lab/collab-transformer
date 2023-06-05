#include "task.hpp"

template <typename T>
Task<T>::Task(int req_num, TaskState prefered_state, T* data, int row, int col, int incRow) {
    this->task_id = (this->task_count)++;
    sem_init(&(this->state_sem), 0, 1);
    sem_init(&(this->req_sem), 0, 1);

    this->prefered_state = prefered_state;
    this->state = TASK_EMPTY;
    this->requirements = req_num;

    if (!data) {
        this->data = malloc(sizeof(T) * row * col);
        this->is_data_dynamic = true;
    }
    else {
        this->data = data;
    }

    this->row = row;
    this->col = col;
    this->incRow = incRow;
    
}

template <typename T>
Task<T>::~Task() {
    if (this->is_data_dynamic) free(this->data);
}

template <typename T>
TaskState Task<T>::try_fetch(TaskState new_state) {
    sem_wait(&this->state_sem);

    if (this->state == TASK_EMPTY) {
        this->state = new_state;

        sem_post(&this->state_sem);
        return TASK_EMPTY;
    }
    else {
        sem_post(&this->state_sem);
        return this->state;
    }
}

template <typename T>
int Task<T>::decrease_req() {
    int result;

    sem_wait(&this->req_sem);

    result = --this->requirements;

    sem_post(&this->req_sem);

    return result;
}

template <typename T>
Task<T> *Task<T>::push_providings(Task<T> *first, Task<T> *...) {
    va_list list;

    va_start(list, first);

    Task *val = first;
    this->providings.push_back(first);

    while (val != NULL) {
        val = va_arg(list, Task *);
        this->providings.push_back(val);
    }

    va_end(list);
}

template <typename T>
T *Task<T>::get_data() {
    return this->data;
}

template <typename T>
int Task<T>::get_row() {
    return this->row;
}

template <typename T>
int Task<T>::get_col() {
    return this->col;
}

template <typename T>
int Task<T>::get_incRow() {
    return this->incRow;
}

template <typename T>
void Task<T>::compute() {
    // NOP
}

template <typename T>
void Task<T>::receive(T *data) {
    for (int i=0; i<this->row; i++) {
        memcpy(this->data + i * this->incRow, data + i * this->row, this->row);
    }
}

template <typename T>
MMTask<T>::MMTask(int req_num, TaskState prefered_state, float alpha, Task<T> *A, Task<T> *B, float beta, T *C, int incRowC, Activation acti)
:Task<T>(req_num, TaskState prefered_state, C, A->get_row(), B->get_col(), incRowC)
{
    this->M = A->get_row();
    this->N = B->get_col();
    this->K = A->get_col();

    this->A = A;
    this->B = B;

    this->dataA = A->get_data();
    this->dataB = B->get_data();

    this->alpha = alpha;
    this->beta = beta;

    this->activation = acti;

    if (!C) {
        this->data = malloc(sizeof(T) * this->M * this->N);
        this->is_data_dynamic = true;
    }
    else {
        this->data = C;
    }

}

template <typename T>
void MMTask<T>::compute() {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        this->M, this->N, this->K, 
        this->alpha, 
        this->dataA, this->incRowA, 
        this->dataB, this->incRowB, 
        this->beta, 
        this->data, this->incRow
    );
}
