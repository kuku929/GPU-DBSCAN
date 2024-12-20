#include <fstream>
#include "dbscan2.cuh"
#include <sstream>
#include <string>


int main(){
    std::ifstream fin("test.txt");
    std::vector<Point<double>> points;
    std::string s;
    while(getline(fin,s)){
        std::istringstream sin(s);
        Point<double> p;
        sin >> p.x >> p.y;
        points.push_back(p);
    }
    DBSCAN<double> scanner(0.1, 10);
    int size=0;
    for(int i=0;i < 100; ++i){
        size = scanner.identify_cluster(points);
        std::cout << i << std::endl;
    }
    std::ofstream fout("clustered.txt");
    for(int i=0;i < size; ++i){
        if(scanner.label(i) == 0){fout << points[i].x << ' ' << points[i].y << std::endl;}
    }
    // scanner.show_labels();
}
