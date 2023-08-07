#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "SmartRedis.hpp"
#include "client.h"
#include <string>
#include <vector>

smartredis_data *sr = new smartredis_data;
SmartRedis::Client *client_ptr;

void smartredis::init_client()
{
  std::string logger_name("Client");
  bool cluster_mode = false;
  std::cout<<"\n"<<"Initializing client ..."<<std::endl;
  sr->ranks_per_db = 1;
  //SmartRedis::Client client(cluster_mode, logger_name); // allocates on stack, goes out of scope outside this function
  //client_ptr = &client; // produces dangling reference
  client_ptr = new SmartRedis::Client(cluster_mode, logger_name); // allocates on heap
  std::cout<<"Done \n"<<std::endl;

  std::cout<<"Testing client ..."<<std::endl;
  size_t dim1 = 3;
  size_t dim2 = 2;
  size_t dim3 = 5;
  std::vector<size_t> dims = {3, 2, 5};
  size_t n_values = dim1 * dim2 * dim3;
  std::vector<double> input_tensor(n_values, 0);
  for(size_t i=0; i<n_values; i++)
    input_tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
  std::string key = "3d_tensor";
  std::cout<<client_ptr<<std::endl;
  client_ptr->put_tensor(key, input_tensor.data(), dims,
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  std::cout<<"Done\n"<<std::endl;
}

void smartredis::put_tensor()
{
  size_t dim1 = 3;
  size_t dim2 = 2;
  size_t dim3 = 5;
  std::vector<size_t> dims = {3, 2, 5};
  size_t n_values = dim1 * dim2 * dim3;
  std::vector<double> input_tensor(n_values, 0);
  for(size_t i=0; i<n_values; i++)
    input_tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
  double total = 0;
  for(size_t i=0; i<n_values; i++)
    total = total+input_tensor[i];
  std::string key = "3d_tensor";
  std::cout<<"\n"<<"Sending tensor ..."<<std::endl;
  client_ptr->put_tensor(key, input_tensor.data(), dims,
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  std::cout<<"Vector sum = "<<total<<std::endl;
  std::cout<<"Done"<<std::endl;
}

void smartredis::get_tensor()
{
  size_t dim1 = 3;
  size_t dim2 = 2;
  size_t dim3 = 5;
  std::vector<size_t> dims = {3, 2, 5};
  size_t n_values = dim1 * dim2 * dim3;
  std::vector<double> unpack_tensor(n_values, 0);
  std::string key = "3d_tensor";
  std::cout<<"\n"<<"Retrieving tensor ..."<<std::endl;
  client_ptr->unpack_tensor(key, unpack_tensor.data(), {n_values},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  double total = 0;
  for(size_t i=0; i<n_values; i++)
    total = total+unpack_tensor[i];
  std::cout<<"Vector sum = "<<total<<std::endl;
  std::cout<<"Done"<<std::endl;
}