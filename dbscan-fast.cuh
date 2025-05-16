/*
 *@Author: Krutarth Patel
 *@Date: 20th December 2024
 *@Description : GDBSCAN header only library
 * using CUDA for speeeeeeeeeeeeeeed.
 */

#include <cwchar>
#include <algorithm>
#include <iostream>
#include <vector>
#include <exception>
#include <cuda_runtime.h>
#include <queue>
#include <fstream>
#include <unistd.h>
#define SIZE 64
#define MAX_NODES 1000000
typedef unsigned char uchar;
//#define DEBUG

__device__ __managed__ int true_count = 1;
#define __START_TIMER__                                                                            \
    float gpu_elapsed_time_ms;                                                                     \
    cudaEvent_t start, stop;                                                                       \
    cudaEventCreate(&start);                                                                       \
    cudaEventCreate(&stop);                                                                        \
    cudaEventRecord(start, 0);

#define __END_TIMER__                                                                              \
    cudaEventRecord(stop, 0);                                                                      \
    cudaEventSynchronize(stop);                                                                    \
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);                                       \
    std::cout << gpu_elapsed_time_ms << std::endl;


template<typename T>
__global__ void num_neighbors_1(int *count_list, T *points, int *adj_list, int offset, int no_of_nodes, float eps){
    int index = blockIdx.x*SIZE + threadIdx.x;
    if(index < no_of_nodes * no_of_nodes){
        int i = index%no_of_nodes;
        int j = index/no_of_nodes;
        atomicAdd(count_list+i,1);
        int neigh = count_list[i];
        T dist = fabs(points[i<<1]-points[j<<1]) + fabs(points[(i<<1)+1]-points[(i<<1)+1]);
        if(dist <= eps) adj_list[neigh + i * offset] = j;
    }
}

template<typename T>
__global__ void num_neighbors(int *count_list, T *points, int *adj_list, int offset, int no_of_nodes, float eps){
    int index = blockIdx.x*SIZE + threadIdx.x;
    int blk_ind = threadIdx.x;
    int glob_stride = offset*index;
    float x,y;
    int temp=0;
    T dist;
    if(index < no_of_nodes){
        x=points[index<<1];
        y=points[(index<<1) + 1];
    }
    __shared__ T l1pts[SIZE*2];
    for(int blk_off=0; blk_off < no_of_nodes;blk_off+=SIZE){
        // for each block load those SIZE no of
        // points. Done by each block
        int tmp = blk_ind + blk_off;
        __syncthreads();
        if(tmp < no_of_nodes){
            l1pts[blk_ind<<1] = points[(tmp)<<1];
            l1pts[(blk_ind<<1) + 1] = points[(tmp<<1) + 1];
        }
        __syncthreads();
        int size = min((no_of_nodes - blk_off), SIZE);
        if(index < no_of_nodes){
            // if a point has more neighbors than max we cut
            // I think this is valid since in my testing I haven't found
            // a case where we actually have a lot of neighbors
            // the pointcloud is pretty uniformly spaced
            for(int i=0;i<size && (temp < offset);++i){
                // should  all  threads access the same
                // memory  at a time  or  is  a cyclic
                // access better?
                dist = fabs(x-l1pts[i<<1])+fabs(y-l1pts[(i<<1) + 1]);
                if(dist <= eps){
                    // this should be shared memory
                    adj_list[glob_stride + temp] = i + blk_off;
                    temp++;
                }
            }
        }
    }
    if(index < no_of_nodes){
        count_list[index]=temp;
    }
}

template<typename T>
class DBSCAN{
private:
	float eps;
	int min_pts;
	int no_nodes;
	int MAX_NEIGHBORS; // thereotical limit on maximum total neighbor
	int OFFSET;        // max neighbor for one node
	double dmin;       // min distance between two nodes
	int *labels;
	uchar *v; 
	T *dev_nodes;
	int *adj_list; 
	int *dev_adj_list;
	int *dev_neighbor_count;
	std::vector<int> neighbor_count;
public:
    DBSCAN(float eps, int min_pts) :  eps(eps), min_pts(min_pts)
    {
        this->dmin = 0.008;
        this->OFFSET = 4 * eps * eps/(dmin * dmin);
        this->MAX_NEIGHBORS = (MAX_NODES * OFFSET) + 1; //+1 for the point itself
        if(eps > 1){
            std::cerr << "WARNING: Are you sure you want to keep this big of an eps? " << eps << std::endl;
            std::cerr << "Yours truly, DBSCAN" << std::endl;
            // now, your scale might be different, so we will adjust to
            // that, the expectation is that eps/dmin is about 100-500
            // more may result in too much memory being allocated on the GPU
            this->dmin = 0.05;
        }
        //allocate nodes on device
        cudaMalloc(&dev_nodes, 2 * sizeof(T)*MAX_NODES);
        cudaMalloc(&dev_neighbor_count, sizeof(int)*MAX_NODES);
        // allocating memory to adjacency list
        // dmin is the minimum distance between two points,
        // eps is the distance considered for two points to be neighboring
        // The maximum neighbors a point could thus have is roughly 2 * eps/dmin
        // I measured dmin in rviz, and this is very rough, but 0.005 should be enough
        // trust...
        adj_list = new int[MAX_NEIGHBORS];
        cudaMalloc(&dev_adj_list, sizeof(int)*(MAX_NEIGHBORS));
        labels = new int[MAX_NODES];
        neighbor_count = std::vector<int>(MAX_NODES, 0);
    }
    ~DBSCAN()
    {
        cudaFree(dev_adj_list);
        cudaFree(dev_nodes);
        cudaFree(dev_neighbor_count);
        delete []labels;
        delete []adj_list;
    }

    int identify_cluster(std::vector<T> &nodes)
    {
    // this will be the effective size, We ignore all points 
    // after MAX_NODES!
#if defined(DEBUG)
        if(nodes.size()/2 > MAX_NODES){
            std::cerr << "Ignoring " << nodes.size() - MAX_NODES << " points!" << std::endl;
            std::cerr << "Yours Truly, DBSCAN" << std::endl;
        }
#endif

        no_nodes = min(MAX_NODES,(int)nodes.size()/2);

        //// Alright, I am going to do something very cool
        //// I know that distance between points increases as 
        //// the distance increases. So I want to account for that
        //double min_dist1 = 1e5;
        //double dist1 = fabs(nodes[0]) + fabs(nodes[1]);
        //for(int i=1;i<no_nodes;++i){
            //min_dist1 = min(min_dist1, fabs(nodes[i<<1]-nodes[0]) + fabs(nodes[i<<1+1]-nodes[1]));
        //}
        //double min_dist2 = 1e5;
        //double dist2 = fabs(nodes[nodes.size()-1]) + fabs(nodes[nodes.size()-2]);
        //for(int i=0;i<no_nodes-1;++i){
            //min_dist2 = min(min_dist2, fabs(nodes[i<<1]-nodes[nodes.size()-2]) + fabs(nodes[i<<1+1]-nodes[nodes.size()-1]));
        //}
        //double c1 = (min_dist1 - min_dist2) / (dist1 - dist2 + 1e-5);
        //double c2 = min_dist1 - c1 * dist1;

        //allocate nodes on device
        cudaMemcpy(dev_nodes, nodes.data(), 2 * sizeof(T)*no_nodes, cudaMemcpyHostToDevice);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cerr << cudaGetErrorString(err) << std::endl;
        }
        //find neighbors
        dim3 dim_block(SIZE, 1, 1);
        dim3 dim_grid((no_nodes + SIZE-1)/SIZE, 1, 1);
        //num_neighbors_1<T><<<(no_nodes*no_nodes+512-1)/512, 512>>>(dev_neighbor_count, dev_nodes, 
            //dev_adj_list, this->OFFSET, no_nodes, eps);
        num_neighbors<T><<<dim_grid, dim_block>>>(dev_neighbor_count, dev_nodes, 
            dev_adj_list, this->OFFSET, no_nodes, eps);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cerr << "ERROR:" << cudaGetErrorString(err) << std::endl;
        }
        cudaMemcpy(neighbor_count.data(), dev_neighbor_count, sizeof(int)*no_nodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(adj_list, dev_adj_list, sizeof(int) * MAX_NEIGHBORS, cudaMemcpyDeviceToHost);
#if defined(DEBUG)
        // allocating memory to adjacency list
        // this is for testing, I am not sure 
        // about the theoretical limit so I will
        // keep this for now. 
        std::cout << "no of nodes: " << no_nodes << std::endl; 
        for(int i=0;i < no_nodes; ++i){
             std::cout << neighbor_count[i] << ' ';
        }
        std::cout << std::endl;
        long long acc = 0;
        for(int i=0;i < no_nodes; ++i){
            acc += (long long)neighbor_count[i];
        }
        // NOTE: this will never happen since I changed
        // num_neighbors()
        //if(acc >= MAX_NEIGHBORS){
            //std::cerr << "Well, this is unexpected. There are more neighbors than theoretically possible!" << std::endl;
            //std::cerr << "Theoretical limit: " << MAX_NEIGHBORS << ", total: " << acc << std::endl;
            //std::cerr << "Two options, tune DBSCAN to match your reality OR tune eps to match DBSCAN's reality" << std::endl;
            //std::cerr << "Yours Truly, DBSCAN" << std::endl;
            //return 0;
        //}

         ////debug
         //std::cout << "adj list :\n";
         //for(int i=0;i < no_nodes; ++i){
         //for(int j=i * OFFSET; j < i * OFFSET + neighbor_count[i]; ++j){
             //std::cout << adj_list[j] << ' ';
         //}
         //std::cout << std::endl;
         //}

#endif
        int cluster_id = 1;
        for(int i=0;i < no_nodes; ++i){
            labels[i]=0;
        }
        int neighbors;
        // bfs to find clusters
        std::queue<int> q;
        for(int node=0;node < no_nodes; ++node){
            if(!labels[node] && neighbor_count[node] >= min_pts){
                q.push(node);
                labels[node] = cluster_id;
                while(!q.empty()){
                    int curr_node = q.front();q.pop();
                    for(int i=curr_node*this->OFFSET;i<curr_node*this->OFFSET+neighbor_count[curr_node];++i){
                        int neighbor = adj_list[i];
                        if(!labels[neighbor]){ 
                            labels[neighbor]=cluster_id;
                            if(neighbor_count[neighbor] >= min_pts)q.push(neighbor); 
                        }
                    }
                }
                cluster_id++;
            }
        }
        for(int node=0;node<no_nodes;++node){
            if(labels[node] == 0){labels[node]=cluster_id++;}
        }
        // we return the number of nodes
        // considered in the algorithm
        return no_nodes;
    }

        void show_labels(){
            std::cout << "labels :\n";
            for(int i=0;i < no_nodes; ++i){
                std::cout << (int)labels[i] << ' ';
            }
            std::cout << '\n';
        }

        uchar label(int index){
                if(index >= MAX_NODES){
                        // we need to think what to do
                        // here, because it could be the case
                        // that we have too many points and we do 
                        // not consider some of them, I could
                        // give a random value each time to make
                        // sure no FALSE POSITIVES are considered 
                        // at least
                        return no_nodes + rand()%no_nodes;
                }
            return labels[index];
        }
};
