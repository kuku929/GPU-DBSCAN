#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#define SIZE 64
typedef unsigned char uchar;

__device__ __managed__ int true_count = 1;

template<typename T>
struct Point{
    Point() = default;
    Point(T i, T j) : x(i) , y(j) {}
    T x, y;

    __device__ __host__ T euclidean_distance(Point<T> &other){
            return sqrt((float)((this->x - other.x)*(this->x - other.x) + (this->y - other.y)*(this->y - other.y)));
    }
    __device__ __host__ T manhattan_distance(Point<T> &other){
            return abs(this->x - other.x) + abs(this->y - other.y);
    }
};

__global__ void search(int *adj_list, int *offset, uchar *frontier, uchar *v, float eps, int min_pts, int no_of_nodes){
    int index = blockIdx.x*SIZE + threadIdx.x;
    if(index < no_of_nodes){
        if(frontier[index]){  //if node is a frontier
            frontier[index] = 0; 
            for(int neighbor=offset[index];neighbor < offset[index+1]; ++neighbor){ //set all its neighbors as frontiers
                if(!v[adj_list[neighbor]]){
                    //if border point
                    if(offset[adj_list[neighbor]+1]-offset[adj_list[neighbor]] >= min_pts){
                        frontier[adj_list[neighbor]] = 1;
                    }
                    v[adj_list[neighbor]] = 1;
                }
            }
        }
        //the first thread sums the frontier array
        int sum=0;
        if(index == 0){
            for(int i=0;i < no_of_nodes; ++i){
                sum+=frontier[i];
            }
            true_count = sum;
        }
    }
}


__global__ void reset(int node, uchar *frontier, uchar *v){
    /*
    * kernel code to do bfs
    */
    int index = blockIdx.x*SIZE + threadIdx.x;
    int val=0;
    if(index == node)
        val=1;
    frontier[index] = val;
    v[index] = val;

    true_count = 1;
}

template<typename T>
__global__ void num_neighbors(int *count_list, Point<T> *points, int no_of_nodes, float eps){
    int index = blockIdx.x*SIZE + threadIdx.x;
    if(index < no_of_nodes){
        int temp=0;
        for(int i=0;i < no_of_nodes; ++i){
            if(i == index)
                continue;
            if(points[index].euclidean_distance(points[i]) <= eps){
                temp++;
            }
        }
        count_list[index]=temp;
    }
}

template<typename T>
__global__ void make_graph(int *adj_list, int *offset, Point<T> *points, int no_of_nodes, float eps){
    int index = blockIdx.x*SIZE + threadIdx.x;

    // // debug
    // if(index == 0){
    //     printf("testing : %f\n", eps);//points[0].euclidean_distance(points[1]));
    // }

    if(index < no_of_nodes){
        int curr_ind = 0;
        for(int i=0;i < no_of_nodes; ++i){
            if(i == index)
                continue;
            if(points[index].euclidean_distance(points[i]) <= eps){
                adj_list[offset[index] + curr_ind] = i;
                curr_ind++;
            }
        }
    }
}

template<typename T>
class DBSCAN{
    public:
        DBSCAN(std::vector<Point<T>> &white_pixel_indices, float eps, int min_pts) :  eps(eps), min_pts(min_pts)
        {
            // NOTE: copying the cluster
            nodes = white_pixel_indices;
            std::vector<int> neighbor_list(white_pixel_indices.size(), 0);
            int *dev_neighbor_list;
        
            //allocate nodes on device
            Point<T> *dev_nodes;
            cudaMalloc(&dev_nodes, sizeof(Point<T>)*nodes.size());
            cudaMemcpy(dev_nodes, nodes.data(), sizeof(Point<T>)*nodes.size(), cudaMemcpyHostToDevice);
            cudaMalloc(&dev_neighbor_list, sizeof(int)*neighbor_list.size());
        
            //find neighbors
            dim3 dim_block(SIZE, 1);
            dim3 dim_grid((nodes.size() + SIZE-1)/SIZE, 1);
            num_neighbors<T><<<dim_grid, dim_block>>>(dev_neighbor_list, dev_nodes, nodes.size(), eps);
        
            //back to host
            cudaMemcpy(neighbor_list.data(), dev_neighbor_list, sizeof(int)*neighbor_list.size(), cudaMemcpyDeviceToHost);
        
			 //debug
			 for(int i=0;i < nodes.size(); ++i){
				 std::cout << neighbor_list[i] << ' ';
			 }
			 std::cout << '\n';
		
            //allocating memory to adjacency list
            prefix_sum = new int[nodes.size()+1];
            prefix_sum[0] = 0;
            for(int i=1;i < nodes.size()+1; ++i){
                prefix_sum[i] = prefix_sum[i-1] + neighbor_list[i-1];
            }
            cudaMalloc(&adj_list, sizeof(int)*(prefix_sum[nodes.size()]));
            cudaMalloc(&dev_prefix, sizeof(int)*(nodes.size()+1)); 
            cudaMemcpy(dev_prefix, prefix_sum, sizeof(int)*(nodes.size()+1), cudaMemcpyHostToDevice);
            make_graph<T><<<dim_grid, dim_block>>>(adj_list, dev_prefix, dev_nodes, nodes.size(), eps);
        
            // //debug
            // std::cout << "adj list :\n";
            // int adj[prefix_sum[nodes.size()]];
            // cudaMemcpy(adj, adj_list, sizeof(int)*prefix_sum[nodes.size()], cudaMemcpyDeviceToHost);
            // for(int i=0;i < nodes.size(); ++i){
            //     for(int j=prefix_sum[i]; j < prefix_sum[i+1]; ++j){
            //         std::cout << adj[j] << ' ';
            //     }
            //     std::cout << '\n';
            // }
        
            cudaFree(dev_nodes);
            cudaFree(dev_neighbor_list);
            //unified memory
            // cudaMallocManaged(&visited, sizeof(int)*no_nodes);
            // cudaMallocManaged(&labels, sizeof(int)*no_nodes);
            no_nodes = nodes.size();
            visited = new uchar[no_nodes];
            labels = new uchar[no_nodes];
            // cudaMallocManaged(&true_count, sizeof(int));
        }
        
        ~DBSCAN()
        {
            cudaFree(adj_list);
            cudaFree(dev_prefix);
            delete []prefix_sum;
            delete []visited;
            delete []labels;
        }

        void identify_cluster(){
            int cluster_id = 1;
            for(int i=0;i < no_nodes; ++i){
                visited[i] = 0;
                labels[i]=0;
            }        
            // allocating memory
            uchar *frontier; 
            uchar *v; 
            cudaMalloc(&frontier, sizeof(uchar)*no_nodes);
            cudaMalloc(&v, sizeof(uchar)*no_nodes);
            int neighbors;
            for(int node=0;node < no_nodes; ++node){
                neighbors = prefix_sum[node+1] - prefix_sum[node];
                if(!visited[node] && neighbors >= min_pts){
                    visited[node] = 1;
                    labels[node] = cluster_id; 
                    bfs(frontier, v, node, eps, min_pts, cluster_id++);
                }
            }
            cudaFree(frontier);
            cudaFree(v);
        }
        
        void show_labels(){
            std::cout << "labels :\n";
            for(int i=0;i < no_nodes; ++i){
                std::cout << (int)labels[i] << ' ';
            }
            std::cout << '\n';
        }

        uchar label(int index){
            return labels[index];
        }
        
        size_t size(){
            return nodes.size();
        }

        Point<T> node(int index){
            return nodes[index];
        }
    private:
        int no_nodes;
        uchar *visited;
        uchar *labels;
        float eps;
        int min_pts;
        std::vector<Point<T>> nodes;
        int *adj_list; 
        int *dev_prefix;
        int *prefix_sum;
        
        void find_nodes(std::vector<Point<T>> &white_pixel_indices){
            for (auto p: white_pixel_indices) {
                nodes.push_back(Point(p.x,p.y));
            }
        }
        
        void bfs(uchar *frontier, uchar *v, int node, float eps, int min_pts, int cluster_id)
        {
            /*
            * start from a node and do bfs
            */
        
            // //debug
            // int adj[prefix_sum[nodes.size()]];
            // cudaMemcpy(adj, adj_list, sizeof(int)*prefix_sum[nodes.size()], cudaMemcpyDeviceToHost);
            // for(int i=0;i < nodes.size(); ++i){
            //     for(int j=prefix_sum[i]; j < prefix_sum[i+1]; ++j){
            //         std::cout << adj[j] << ' ';
            //     }
            //     std::cout << '\n';
            // }
        
        
            dim3 dim_block(SIZE, 1);
            dim3 dim_grid((no_nodes + SIZE-1)/SIZE);
            reset<<<dim_grid,dim_block>>>(node, frontier, v);
            // cudaDeviceSynchronize();
            // *true_count = 1;
        
            //debug
            // int counter=1;
        
            while(true_count){
                search<<<dim_grid, dim_block>>>(adj_list, dev_prefix, frontier, v, eps, min_pts, no_nodes);
                cudaDeviceSynchronize();
        
                //debug
                // if(counter == 10){
                //     break;
                // }
                // counter++;
                
            }
            //back to host
            uchar V[no_nodes];
            cudaMemcpy(V, v, sizeof(uchar)*no_nodes, cudaMemcpyDeviceToHost);        
        
            for(int node=0;node < no_nodes; ++node){
                if(V[node]){
                    labels[node] = cluster_id;
                    visited[node] = 1;
                }
            }
        }
};
