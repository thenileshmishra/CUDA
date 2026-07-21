#include <stdio.h>

int main(){
    int value = 42;
    int *ptr1 = &value;
    int ** ptr2 = &ptr1;
    int ***ptr3 = &ptr2;


    printf("value: %d/n", ***ptr3);
    return 0;
}