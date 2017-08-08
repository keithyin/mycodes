// #include <iostream>
// #include <thread>
// #include <vector>
// #include <mutex>

// static std::mutex barrier;
  
// void dot_product(const std::vector<int> &v1, const std::vector<int> &v2, int &result, int L, int R){
//      int patial_sum = 0;
//      for(int i = L; i < R; ++i){
//          patial_sum += v1[i] * v2[i];
//      }
//      std::lock_guard<std::mutex> block_thread(barrier);
//      result += patial_sum;
//  }
 
//  int main(){
//      int nr_elements = 100000;
//      int nr_threads = 4;
//      int result = 0;
//      std::vector<std::thread> threads;
 
//      //Fill two vectors with some constant values for a quick verification
//      // v1={1,1,1,1,...,1}
//      // v2={2,2,2,2,...,2}
//      // The result of the dot_product should be 200000 for this particular case
//      std::vector<int> v1(nr_elements,1), v2(nr_elements,2);
 
//      //Split nr_elements into nr_threads parts
//      //std::vector<int> limits = bounds(nr_threads, nr_elements);
 
//      //Launch nr_threads threads:
//      for (int i = 0; i < nr_threads; ++i) {
//          threads.push_back(std::thread(dot_product, std::ref(v1), std::ref(v2), std::ref(result), i*25000, (i+1)*25000));
//      }
 
 
//      //Join the threads with the main thread
//      for(auto &t : threads){
//          t.join();
//      }
 
//      //Print the result
//      std::cout<<result<<std::endl;
 
//      return 0;
// }

/*****************************************************************************/


// #include <iostream>
// #include <thread>

// void foo() { std::cout << "foo()\n"; }
// void bar() { std::cout << "bar()\n"; }

// int main()
// {
// 	std::thread t([]{
// 		        foo();
// 			bar();						 
// 	                });
// 	t.join();
// 	std::cout<<"num concurrency "<<std::thread::hardware_concurrency()<<std::endl;
// 	{
// 		int i = 0;
// 	}
// 	std::cout<< i <<std::endl;
// 	return 0;
// }

/****************************************************************************/
#include <initializer_list>
#include <iostream>
using namespace std;

int main(){
	initializer_list<int> lst({1,2,3,2,1});
	auto cursor = lst.begin();
	auto end = lst.end();
	for (;cursor != end; cursor++)
		cout<<*cursor<<endl;
	return 0;
}