//
// Created by psaw on 9/16/16.
//


#include <stdio.h>

void *getStruct(int age);

typedef struct base_test_1 {
    char *name;
} test_1;

typedef struct base_test_2 {
    int age;
} test_2;

typedef struct container_test{
    test_2 test2;

};

int main(int argc, char **argv) {
    void *test = getStruct(10);
    printf("Age is %d \n", ((test_2 *) test)->age);
}

void *getStruct(int age) {
    test_2 test2 = {age};
    return &test2;
}