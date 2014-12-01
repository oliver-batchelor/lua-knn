#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/knn.cpp"
#else


#include <luaT.h>
#include <TH.h>


inline void libknn_(push)(lua_State *L, THTensor *tensor) {
  THTensor_(retain)(tensor);
  luaT_pushudata(L, tensor, torch_Tensor);
}


inline THTensor *libknn_(checkTensor)(lua_State* L, int arg) {
  return (THTensor*)luaT_checkudata(L, arg, torch_Tensor);  
}

template<> struct TensorType<real> { typedef THTensor Tensor; };
inline real *data(THTensor *tensor) { return THTensor_(data)(tensor); }

inline THLongStorage *newSizeOf(THTensor *tensor) { return THTensor_(newSizeOf)(tensor); }


template<>
inline THTensor *check<real>(lua_State *L, int i) { return libknn_(checkTensor)(L, i); }


template<typename T>
inline int libknn_(lookup) (lua_State *L) {
  try {

    THTensor *table = libknn_(checkTensor)(L, 1);
    typename TensorType<T>::Tensor *index = check<T>(L, 2);
    
    THTensor *dest = THTensor_(newWithSize)(newSizeOf(index), NULL);
    real *tableData = (real*)data(table);
    
    TH_TENSOR_APPLY2(real, dest, T, index,  *dest_data = tableData[*index_data - 1]; );
    
    libknn_(push)(L, dest);
    
    return 1;       
    
  } catch (std::exception const &e) {
    luaL_error(L, e.what());
  }    
}

static int libknn_(lookupByte) (lua_State *L) { return libknn_(lookup)<unsigned char>(L); }
static int libknn_(lookupShort) (lua_State *L) { return libknn_(lookup)<short>(L); }
static int libknn_(lookupInt) (lua_State *L) { return libknn_(lookup)<int>(L); }
static int libknn_(lookupLong) (lua_State *L) { return libknn_(lookup)<long>(L); }


static const luaL_reg libknn_(Main__) [] =
{
  {"lookup_byte",   libknn_(lookupByte)   },
  {"lookup_short",   libknn_(lookupShort)   },
  {"lookup_int",   libknn_(lookupInt)   },
  {"lookup_long",   libknn_(lookupLong)   },
  {NULL, NULL}  /* sentinel */
};


extern "C" {

  DLL_EXPORT int libknn_(Main_init) (lua_State *L) {
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, libknn_(Main__), "libknn");
    lua_pop(L,1); 
    return 1;
  }

}

#endif