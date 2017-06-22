
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Used when reading the kernel file
#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

struct dataStruct {
public:
	int data_size;
	int data_dimensions;
	float *data;
};

/* Generic functions */

/*
@brief Outputs any error messages to the terminal and returns -1
in order to terminate the program's execution
message: the string of error message to be put on screen
*/
int CERR(string message) {
	cout << message << endl;
	return (-1);
}

/*
@brief Reads the input data from a text file.
filename: the name of the file to read the data from
data: the data structure to hold the data it reads
*/
int LoadData(string filename, dataStruct &data)
{
	ifstream in(filename);
	if (in.is_open()) {
		// first line holds the data size
		in >> data.data_size;
		// second line holds the data dimension
		in >> data.data_dimensions;
		// The two above are used to allocate the proper memory, since OpenCL requires arrays
		data.data = (float *)malloc(data.data_dimensions*data.data_size*sizeof(float));

		for (int i = 0; i < data.data_size; i++) {
			for (int j = 0; j < data.data_dimensions; j++) {
				in >> data.data[data.data_dimensions*i + j];
			}
		}

		return 1;
	}
	else {
		return (CERR("Error openning the file"));
	}
}

/*
@brief Writes the resulting data to a text file.
filename: the name of the file to write the data
data: the data structure of the data to write
*/
int WriteData(string filename, dataStruct data)
{
	ofstream out(filename);
	if (out.is_open()) {
		// first line holds the data size
		out << data.data_size << "\n";
		// second line holds the data dimension
		out << data.data_dimensions << "\n";
		// The rest is data
		for (int i = 0; i < data.data_size; i++) {
			for (int j = 0; j < data.data_dimensions; j++) {
				out << data.data[data.data_dimensions*i + j] << "\t";
			}
			out << "\n";
		}
		out.close();
		return 1;
	}
	else {
		return (CERR("Error writing to file"));
	}
}


/*
main
*/
int main(int argc, char** argv) {

	if (argc != 3) {
		return(CERR("Usage: SOM_parallel <path to filename in> <path to filename out>"));
	}

	// Read the input file
	string filein = argv[1];
	dataStruct data_in;
	if (LoadData(filein, data_in) == -1) {
		return(CERR("Error reading file"));
	}

	int data_size = data_in.data_dimensions * data_in.data_size;

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("C:\\Users\\dzerm\\Documents\\GitHubProjects\\SOM\\build\\Debug\\kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
		&device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		data_size * sizeof(float), NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		data_size * sizeof(float), NULL, &ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
		data_size * sizeof(float), data_in.data, 0, NULL, NULL);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
		(const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "add_one", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);

	// Execute the OpenCL kernel on the list
	size_t global_item_size = data_size; // Process the entire lists
	size_t local_item_size = 64; // Divide work items into groups of 64
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, &local_item_size, 0, NULL, NULL);

	// Read the memory buffer C on the device to the local variable C
	float *data_out_temp = (float*)malloc(sizeof(float)*data_size);
	ret = clEnqueueReadBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
		data_size * sizeof(float), data_out_temp, 0, NULL, NULL);

	string fileout = argv[2];
	dataStruct data_out;
	data_out.data_dimensions = 4;
	data_out.data_size = 12;
	data_out.data = (float *)malloc(data_out.data_dimensions*data_out.data_size*sizeof(float));
	data_out.data = data_out_temp;

	if (WriteData(fileout, data_out) == -1) {
		return(CERR("Error writing file"));
	}

	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	return 0;
}