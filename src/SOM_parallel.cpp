
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
int CERR(string message) 
{
	cout << message << endl;
	return (-1);
}

/*
	@brief Checks the OpenCL functions' return value and outputs any error messages 
	to the terminal. If an error occurs, the process terminates.
	error: integer holding the error number, or 0 if no error occurs
*/
int CheckOpenCLError(cl_int error)
{
	if (error != CL_SUCCESS) {
		cerr << "OpenCL call failed with error " << error << endl;
		return (-1);
	}
	return 1;
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
	@brief Get the name of the platform providing its ID
	id: the ID of the platform
*/
string GetPlatformName(cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);
	
	string result;
	result.resize(size);
	clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char*> (result.data()), nullptr);
	return result;
}

/*
@brief Get the name of the device providing its ID
id: the ID of the device
*/
string GetDeviceName(cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);
	
	string result;
	result.resize(size);
	clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char*> (result.data()), nullptr);

	return result;
}

/*
	main
*/
int main(int argc, char** argv) {

	if (argc != 3) {
		return(CERR("Usage: SOM_parallel <path to filename in> <path to filename out>"));
	}

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("C:\\Users\\dzerm\\Documents\\GithubProjects\\SOM\\src\\kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/*
		Get platform information
	*/ 
	cl_int ret; // all error codes are stored here
	cl_uint platformIdCount = 0;
	// Ask for the number of platforms available
	ret = clGetPlatformIDs(0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		return(CERR("No OpenCL platform found"));
	}
	else {
		cout << "Found " << platformIdCount << " platform(s)" << endl;
	}

	// Ask again for the platform IDs now that you know their number
	vector<cl_platform_id> platformIds(platformIdCount);
	ret = clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		cout << "\t (" << (i + 1) << ") : " << GetPlatformName(platformIds[i]) << endl;
	}

	// Let the user choose which platform
	cout << "Select a platform. (1, 2, ...) :\t";
	int platformSelection;
	cin >> platformSelection;

	if (platformSelection < 1 || platformSelection > platformIds.size()) {
		return((CERR("The platform selected does not exist")));
	}

	/*
		Get device information
	*/
	cl_uint deviceIdCount;
	platformSelection--;
	// Ask for the number of devices available
	ret = clGetDeviceIDs(platformIds[platformSelection], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

	if (deviceIdCount == 0) {
		return(CERR("No OpenCL devices found"));
	}
	else {
		cout << "Found " << deviceIdCount << " device(s)" << endl;
	}
	
	// Ask again for the device IDs now that you know their number
	vector<cl_device_id> deviceIds(deviceIdCount);
	ret = clGetDeviceIDs(platformIds[platformSelection], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		cout << "\t (" << (i + 1) << ") : " << GetDeviceName(deviceIds[i]) << endl;
	}

	// Let the user choose which platform
	cout << "Select a device. (1, 2, ...) :\t";
	int deviceSelection;
	cin >> deviceSelection;

	if (deviceSelection < 1 || deviceSelection > deviceIds.size()) {
		return((CERR("The device selected does not exist")));
	}

	deviceSelection--;
	cl_device_id deviceID = deviceIds[deviceSelection];

	/*
		Create an OpenCL context
	*/
	const cl_context_properties contextProperties[] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[platformSelection]), 0, 0
	};
	cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), nullptr, nullptr, &ret);
	CheckOpenCLError(ret);

	/* 
		Create a command queue
	*/
	cl_command_queue command_queue = clCreateCommandQueue(context, deviceID, 0, &ret);
	CheckOpenCLError(ret);

	// Read the input file
	string filein = argv[1];
	dataStruct data_in;
	if (LoadData(filein, data_in) == -1) {
		return(CERR("Error reading file"));
	}

	int data_size = data_in.data_dimensions * data_in.data_size;

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size * sizeof(float), NULL, &ret);
	CheckOpenCLError(ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size * sizeof(float), NULL, &ret);
	CheckOpenCLError(ret);

	// Copy the data_in to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_FALSE, 0, data_size * sizeof(float), data_in.data, 0, NULL, NULL);
	CheckOpenCLError(ret);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	CheckOpenCLError(ret);

	// Build the program
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	if (ret != CL_SUCCESS) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
	}

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "add_one", &ret);
	CheckOpenCLError(ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	CheckOpenCLError(ret);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	CheckOpenCLError(ret);

	// Execute the OpenCL kernel on the list
	size_t local_item_size = 16; // Divide work items into groups of 64
	size_t numLocalGroups = ceil(data_size / local_item_size);
	size_t global_item_size = local_item_size * numLocalGroups;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, nullptr, 0, NULL, NULL);

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