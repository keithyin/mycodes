#include <iostream>

using namespace std;

class ZhangSB{
public:
    ZhangSB(int age){
        this->age = age;
    }

    int& operator() (){
        return age; 
    }

    int get_age(){
        return this->age;
    }
private:
    int age;
};

void test_lambda(){
    int i = 1;
    cout<<"before lambda, the value of i is "<< i <<endl;
    auto lam = [&i]{ i = 2;};
    lam();
    cout<<"after lambda, the value of i is "<<i<<endl;
}

void test_mutable(){
  int v1 = 2;
  auto f = [v1]() mutable {return ++v1;};
  auto val = f();
  cout<<"after lambda v1  "<<val<<endl;
}

int main(){
    ZhangSB zsb(2);
    cout<<"age:" <<zsb.get_age()<<endl;
    zsb() = 10;
    cout<<"age:" <<zsb.get_age()<<endl;
    test_mutable();
}


