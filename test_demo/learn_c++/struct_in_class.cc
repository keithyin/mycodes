#include <iostream>
#include <string>
using namespace std;

class People{
public:
	struct Attr
	{
		string name;
		int age;
	};

	People(string name, int age){
		attr.name = name;
		attr.age = age;
	}

	void print_attr(){
		cout<<"name: "<<attr.name<<" age: "<<attr.age<<endl;
	}
	

private:
	Attr attr;

};

int main(int argc, char const *argv[])
{
	People peo("yin", 18);
	peo.print_attr();
	People::Attr attr;
	attr.name = "yinpeng";
	attr.age = 19;
	cout<<"age "<<attr.age<<endl;
	
	People peo1 = People("hhh", 20);
	peo1.print_attr();
	return 0;
}