#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "client.h"

struct smartredis_data {
  int ranks_per_db;
};

namespace smartredis
{
  void init_client();
  void put_tensor();
  void get_tensor();
}