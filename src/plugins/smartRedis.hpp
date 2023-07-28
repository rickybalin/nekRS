#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "client.h"

struct smartredis_data {
  SmartRedis::Client *client;
  int ranks_per_db;
};
extern smartredis_data sr;

namespace smartredis
{
  void init_client();
  void put_tensor();
  void get_tensor();
}