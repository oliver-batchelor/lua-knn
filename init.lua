
require 'torch'
require 'sys'
require 'paths'
require 'dok'


-- load C lib
require 'libknn'


local knn = {}

function knn.knn(...)

   local _, ref, query, k = dok.unpack(
      {...},
      'knn.knn',
      [[K-Nearest Neighbours]],
           
      {arg='ref', type='torch.FloatTensor',
       help='reference points (m x h) 2d tensor', req=true},
       
      {arg='query', type='torch.FloatTensor',
       help='query point(s) (n x h) 2d tensor or (h) 1d tensor', req=true},
             
      {arg='k', type='number',
       help='number of results returned per query point', default=1}
   )
   
   
   if(query:dim() == 1) then
     query = query:resize(1, query:size(1))
   end
   
   assert(query:dim() == 2 and ref:dim() == 2, "query must be 1d or 2d tensor (h or n x h), ref must be a 2d (h x m) tensor")
   assert(query:size(2) == ref:size(2), "query and ref must have equal size features")
--    assert(query:size(1) <= 65535 and ref:size(1) <= 65535, "maximum size permitted is 65535")
   
   k = math.min(k, ref:size(1))
   
   return libknn.knn(k, ref:t():contiguous(), query:t():contiguous())
end



return knn