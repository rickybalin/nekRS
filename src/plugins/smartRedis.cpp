#ifdef ENABLE_SMARTREDIS

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
  // Replace this with variable in .par file
  sr->ranks_per_db = 1;
  sr-> db_nodes = 1;

  // Initialize SR client
  if(platform->comm.mpiRank == 0)
    printf("\nInitializing client ...\n");
  bool cluster_mode;
  if (sr->db_nodes > 1)
    cluster_mode = true;
  else
    cluster_mode = false;
  //SmartRedis::Client client(cluster_mode, logger_name); // allocates on stack, goes out of scope outside this function
  //client_ptr = &client; // produces dangling reference
  std::string logger_name("Client");
  client_ptr = new SmartRedis::Client(cluster_mode, logger_name); // allocates on heap
  if(platform->comm.mpiRank == 0)
    printf("Done\n");
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


void smartredis::put_data(nrs_t *nrs, dfloat time, int tstep)
{
  int rank = platform->comm.mpiRank;
  std::string key = "u_" + std::to_string(rank);
  if(rank == 0)
    printf("\nSending field with key %s \n",key.c_str());
  client_ptr->put_tensor(key, nrs->U, {nrs->fieldOffset,3},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  if(rank == 0)
    printf("Done\n\n");

  if(rank == 0)
    printf("Checking array ...\n");
  dfloat *u = new dfloat[nrs->fieldOffset * 3]();
  client_ptr->unpack_tensor(key, u, {nrs->fieldOffset * 3},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  double error = 0.0;
  for (int n=0; n<nrs->fieldOffset*3; n++) {
    error = error + (u[n] - nrs->U[n])*(u[n] - nrs->U[n]);
  }
  if(rank == 0)
    printf("Error in fields = %f\n",error);
    printf("Done\n\n");

}

#endif