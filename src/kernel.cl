__kernel void add_one(__global const float *A, __global float *B) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
	// Print out which CU you are
	fprintf("CU id is: %d", i);

    // Do the operation
    B[i] = A[i] + 1.0;
}