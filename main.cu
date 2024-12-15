#include <fstream>
#include "dbscan.cuh"
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
    DBSCAN<double> scanner(points, 0.1, 100);
    scanner.identify_cluster();
    scanner.show_labels();
    std::ofstream fout("clustered.txt");
    for(int i=0;i < scanner.size(); ++i){
        if(scanner.label(i) == 0){fout << points[i].x << ' ' << points[i].y << std::endl;}
    }
}
