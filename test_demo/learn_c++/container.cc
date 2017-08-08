#include <utility>
#include <map>
#include <set>
#include <iostream>

using namespace std;

void test_set()
{

}

void test_pair()
{

}

void test_map()
{
	map<string,int> string_int = {{"a", 1}, {"b",2}, {"v",3}, {"c",4}};
	map<string,int>::iterator cursor = string_int.begin();
	map<string,int>::iterator end = string_int.end();

	for(;cursor!=end; ++cursor){
		cout<< "key: "<<cursor->first<<", value: "<<cursor->second<<endl;
	}
}

int main(int argc, char const *argv[])
{
	test_map();
	
	return 0;
}