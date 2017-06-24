__kernel void euclidean_dist(__global const float4 *data, __global const float4 *weights, __global float *dist) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    dist[i] = length(data[i] - weights);
}