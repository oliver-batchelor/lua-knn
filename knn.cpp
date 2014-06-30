#include <TH.h>
#include <luaT.h>

#include <stdexcept>

template<typename T> struct TensorType;


template<typename T> typename TensorType<T>::Tensor *check(lua_State *L, int i);



extern void knn(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);

static int knn(lua_State* L) {
  THFloatTensor *distances = NULL;
  THIntTensor *indices = NULL;
  
  try {
    
    int k = lua_tonumber(L, 1);
    THFloatTensor *ref =   (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    THFloatTensor *query =   (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");  
    
    if(ref->nDimension != 2 || query->nDimension != 2) 
      throw std::invalid_argument("knn: expected 2d tensor of reference and query points");
    
    size_t features = ref->size[0];
    if(features != query->size[0])
      throw std::invalid_argument("knn: query and reference points must have the same size");
    
    distances = THFloatTensor_newWithSize2d(k, query->size[1]);
    indices = THIntTensor_newWithSize2d(k, query->size[1]);      

  
    knn(THFloatTensor_data(ref), ref->size[1], THFloatTensor_data(query), query->size[1], features, k, THFloatTensor_data(distances), THIntTensor_data(indices));
    
    THFloatTensor_retain(distances);
    THIntTensor_retain(indices);
    
    luaT_pushudata(L, distances, "torch.FloatTensor");
    luaT_pushudata(L, indices, "torch.IntTensor");
    
    return 2;
  
  } catch (std::exception const &e) {
    luaL_error(L, e.what());
  }  
  
  
  return 1;
}

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor        TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define libknn_(NAME) TH_CONCAT_3(libknn_, Real, NAME)

#include "generic/knn.cpp"
#include "THGenerateAllTypes.h"




//============================================================
// Register functions in LUA
//
static const luaL_reg libknn_init [] =
{  
  {"knn",   knn},
  {NULL,NULL}
};


extern "C" {

  DLL_EXPORT int luaopen_libknn(lua_State *L)
  {

    luaL_register(L, "libknn", libknn_init);
    
    libknn_ByteMain_init(L);
    libknn_CharMain_init(L);
    libknn_IntMain_init(L);
    libknn_LongMain_init(L);
    libknn_FloatMain_init(L);
    libknn_DoubleMain_init(L);    

    return 1;
  }

}