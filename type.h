// for determining var types
#include <cxxabi.h>

template <typename T>
void type(T var){
    int st;
    cout << abi::__cxa_demangle(typeid( var ).name(), 0, 0, &st) << endl;
}