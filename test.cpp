#include <fstream>
#include <cmath>

int main(){
    double sq1=5,sq2=10,dmin=0.05;
    std::ofstream fout("build/test.txt");
    for(double x=-sq1;x<=sq1;x+=dmin){
        for(double y=-sq1;y<=sq1;y+=dmin){
            fout << x << ' ' << y << std::endl;
        }
    }
    for(double x=-sq2;x<=sq2;x+=dmin){
        for(double y=-sq2;y<=-(sq1+sq2)/2;y+=dmin){
            fout << x << ' ' << y << std::endl;
        }
        for(double y=(sq1+sq2)/2;y<=sq2;y+=dmin){
            fout << x << ' ' << y << std::endl;
        }
    }
    for(double y=-sq2;y<=sq2;y+=dmin){
        for(double x=-sq2;x<=-(sq1+sq2)/2;x+=dmin){
            fout << x << ' ' << y << std::endl;
        }
        for(double x=(sq1+sq2)/2;x<=sq2;x+=dmin){
            fout << x << ' ' << y << std::endl;
        }
    }
    
    fout.close();
    return 0;
}
