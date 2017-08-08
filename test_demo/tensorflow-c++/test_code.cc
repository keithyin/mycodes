#include <tensorflow/c/c_api.h>
#include <stdio.h>
using namespace std;
int main(){
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}