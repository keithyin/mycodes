#include <iostream>
#include <vector>
using namespace std;


class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        auto column_cursor = array.begin();
        auto column_end = array.end();
        for (; column_cursor!=column_end; ++column_cursor){
            if ((*column_cursor)[0]<target){
            	if (column_cursor != (column_end-1))
                	continue;
            }
            if (column_cursor != (column_end-1))
            	--column_cursor;
            auto row_cursor = (*column_cursor).begin();
            auto row_end = (*column_cursor).end();
            for (;row_cursor!=row_end; ++row_cursor){
                if ((*row_cursor)==target)
                    return true;
            }
            return false;
            
        }
        return false;
    }
};

int main(){
	vector<vector<int> > array;
	vector<int> col1 = {1,2,3,4,7};
	vector<int> col2 = {2,3,4,5,8};
	vector<int> col3 = {3,4,5,6,9};
	array.push_back(col1);
	array.push_back(col2);
	array.push_back(col3);
	Solution s;
	cout<<s.Find(6, array)<<endl;
	return 0;
}