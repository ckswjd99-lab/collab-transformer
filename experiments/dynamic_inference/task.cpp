#include "task.hpp"

float mult_input     [MULT_M * MULT_K1];
float mult_weight1   [MULT_K1 * MULT_K2];
float mult_interm    [MULT_M * MULT_K2];
float mult_weight2   [MULT_K2 * MULT_N];
float mult_output    [MULT_M * MULT_N];

Task::Task(int layer, int op_idx) {
    this->layer = layer;
    this->op_idx = op_idx;
    this->next = NULL;
}

Task::~Task() {}

int Task::get_layer() {
    return this->layer;
}

int Task::get_op_idx() {
    return this->op_idx;
}

Task *Task::get_next() {
    return this->next;
}

void Task::set_next(Task *task) {
    this->next = task;
}

int Task::same(int layer, int op_idx) {
    return (this->layer == layer && this->op_idx == op_idx);
}

void Task::compute() {
    if (this->layer == 0) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            MULT_SUBM, MULT_K2, MULT_K1, 
            1.0, 
            mult_input + this->op_idx * MULT_K1 * MULT_SUBM, MULT_K1, 
            mult_weight1, MULT_K2, 
            1.0, 
            mult_interm + this->op_idx * MULT_K2 * MULT_SUBM, MULT_K2
        );
    }
    else if (this->layer == 1) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            MULT_SUBM, MULT_N, MULT_K2,
            1.0, 
            mult_interm + this->op_idx * MULT_K2 * MULT_SUBM, MULT_K2,
            mult_weight1, MULT_N,
            1.0, 
            mult_output + this->op_idx * MULT_N * MULT_SUBM, MULT_N
        );
    }
}

TaskQueue::TaskQueue(int seed) {
    sem_init(&(this->sem), 0, 1);

    this->head = new Task(-1, -1);

    if (seed >= 0) {
        for (int i=0; i<SPLIT_NUM; i++) {
            this->push_back(new Task(0, i));
        }
    }
    else {
        for (int i=SPLIT_NUM-1; i>=0; i--) {
            this->push_back(new Task(0, i));
        }
    }
    
}

TaskQueue::~TaskQueue() {
    Task *temp;

    while (this->head->get_next() != NULL) {
        temp = this->head;
        this->head = this->head->get_next();
        delete temp;
    }

    delete this->head;
}

Task *TaskQueue::find(int layer, int op_idx) {
    Task *temp = this->head;

    while(temp->get_next() != NULL) {
        temp = temp->get_next();
        if (temp->same(layer, op_idx)) return temp;
    }

    return NULL;
}

void TaskQueue::push_back(Task *task) {
    sem_wait(&(this->sem));

    Task *temp = this->head;

    while(temp->get_next() != NULL) {
        temp = temp->get_next();
    }

    temp->set_next(task);

    sem_post(&(this->sem));
}

void TaskQueue::push_front(Task *task) {
    sem_wait(&(this->sem));

    Task *temp = this->head->get_next();
    this->head->set_next(task);
    task->set_next(temp);

    sem_post(&(this->sem));
}

Task *TaskQueue::pop_front() {
    sem_wait(&(this->sem));

    Task *temp = this->head->get_next();
    this->head->set_next(temp ? temp->get_next() : NULL);

    sem_post(&(this->sem));

    return temp;
}

void TaskQueue::print() {
    printf("[ ");

    Task *temp = this->head;

    while(temp->get_next()) {
        temp = temp->get_next();
        printf("(%d, %d) ", temp->get_layer(), temp->get_op_idx());
    }

    printf("]\n");
}