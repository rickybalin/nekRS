#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "SmartRedis.hpp"
#include "client.h"
#include <string>
#include <vector>

smartredis_data *sr = new smartredis_data;

void smartredis::init_client()
{
  std::string logger_name("Client");
  bool cluster_mode = false;
  std::cout<<"\n"<<"Initializing client ..."<<std::endl;
  sr->ranks_per_db = 1;
  SmartRedis::Client client(cluster_mode, logger_name);
  sr->client = &client;
  std::cout<<"Done \n"<<std::endl;
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
  std::string key = "3d_tensor";
  std::cout<<"\n"<<"Sending tensor ..."<<std::endl;
  std::cout<<sr->ranks_per_db<<std::endl;
  sr->client->put_tensor(key, input_tensor.data(), dims,
                    SRTensorTypeDouble, SRMemLayoutContiguous);
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
  sr->client->unpack_tensor(key, unpack_tensor.data(), {n_values},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  for(size_t i=0; i<n_values; i++)
    std::cout<<"Received: "<<unpack_tensor[i]<<std::endl;
}