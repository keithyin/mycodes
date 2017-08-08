#include <iostream>

void printf(){
	std::cout<<"hello, world."<<std::endl;
}

int main(int argc, char const *argv[])
{
	void (*fp)();
	fp = &printf;
	(*fp)();
	return 0;
}