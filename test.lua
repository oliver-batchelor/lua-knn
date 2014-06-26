
local knn = require 'knn'

local function time(name, f)
  local t = torch.Timer()
  local r = f()
  print(string.format("%s = %s: %.2f", name, tostring(r), t:time().real))
  
  collectgarbage()
end

local test = {}


local makeData = function(n, size)
  return torch.FloatTensor():range(1, size * n):reshape(n, size)
end

test.benchmark = function(k, size, q, n)
  local data = makeData(n, size)
  local query = makeData(q, size)

  time(string.format("knn (k: %d) (query: %s) (data: %s)", k, tostring {query}, tostring {data}), function ()
    knn.knn(data, query, k)
  end)

end


test.benchmark(2, 128, 10000, 10000)
test.benchmark(4, 128, 10000, 50000)
test.benchmark(8, 128, 50000, 50000)
test.benchmark(24, 128, 100000, 100000)
test.benchmark(16, 1024, 50000, 50000)



return test