#include <fstream>
#include <cmath>

int main(){
    double r1=10.0, r2=12.0;
    double theta = 0;
    std::ofstream fout("build/test.txt");
    for(;theta<2*M_PI;theta+=0.0001){
        fout << (r1 * cos(theta)) << ' ' << (r1 * sin(theta)) << std::endl;
        fout << (r2 * cos(theta)) << ' ' << (r2 * sin(theta)) << std::endl;
    }
    fout.close();
    return 0;
}
