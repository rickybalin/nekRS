#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

struct smartredis_data {
  int ranks_per_db;
  int db_nodes;
};

namespace smartredis
{
  void init_client();
  void put_tensor();
  void get_tensor();
  void put_data(nrs_t *nrs, dfloat time, int tstep);
}