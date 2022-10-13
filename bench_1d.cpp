#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <array>
#include <vector>
#include <complex>


int main()
{
    int nfft = 3, i;
    std::vector<std::complex<double>> a(5);
    // for (i = 0; i < nfft; i++) {
        // a.push_back({static_cast<double>(i*7), 8*10.0});
    // }

    for(int i=0; i < a.size(); i++){
        // a[i] = {i*i*1.0 + 100.0, 2.0*i + 20};
        std::cout << a.at(i) << ' ';
    }    
    std::cout<< "\n";
}