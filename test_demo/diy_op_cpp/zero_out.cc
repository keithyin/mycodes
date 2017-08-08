#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include <iostream>
using namespace std;
using namespace tensorflow;

typedef FunctionDefHelper FDH;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .Output("indice: int32")
    ;

//REGISTER_OP("ZeroOut")
//    .Input("to_zero: int32")
//    .Output("zeroed: int32")
//    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//      c->set_output(0, c->input(0));
//      return Status::OK();
//    });

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
      cout<<"hello, there"<<endl;
    }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();
    cout<< "the first input value = " << typeid(input).name() <<endl;

    cout<<"cpu threads "<<context->device()->tensorflow_cpu_worker_threads()->num_threads<<endl;
    cout<<"workers " << context->device()->tensorflow_cpu_worker_threads()->workers <<endl;
    //cout<<"workers' num thread "<<context->device()->tensorflow_cpu_worker_threads()->workers->NumThreads()<<endl;

    // Create an output tensor
    Tensor* output_tensor = NULL;
    Tensor* output_tensor_indice = NULL;
    TensorShape indice_shape;
    int dims[] = {1};
    TensorShapeUtils::MakeShape(dims, 1, &indice_shape);


    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, indice_shape,
                                                     &output_tensor_indice));
    auto output_flat = output_tensor->flat<int32>();
    auto indice_flat = output_tensor_indice->flat<int32>();
    indice_flat(0) = 3;
    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};



REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);

Status ZeroOutGrad(const AttrSlice& attrs, FunctionDef* g){
//clang-format off

*g = FDH::Define(
"ZeroOutGrad",
// arg defs
{"x: T", "grad_softmax: T" },

{"grad_x: T"},

{{"T:{float, double}"}},

//Nodes

);

}

//REGISTER_OP_GRADIENT("ZeroOut", ZeroOutGradOp)
//
// TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
// g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

