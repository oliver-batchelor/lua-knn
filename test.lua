
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

  time(string.format("knn (k: %d) (features: %d) (query: %d) (data: %d)", k, query:size(2), query:size(1), data:size(1)), function ()
    knn.knn(data, query, k)
  end)

end


test.test = function(k, size, q, n)
  local data = torch.FloatTensor():rand(n, size)
  local query = torch.FloatTensor(q, size):zero()
  
  
  local inds = {}
  for i = 1, q do
    local index = torch.random(n)
    inds[i] = index
    query[i] = data[index] 
  end

  local dists, indices = knn.knn(data, query, k)
  
--   print(data, query, dists, indices)

  for i = 1, q do
    assert(dists[i][1] == 0, "distance should be zero, was: "..tostring(dists[i][1]))
    assert(indices[i][1] == inds[i], "indices aren't correct, was: "..tostring(indices[i][1]).." should be: "..inds[i])
  end

  print(string.format("test passed, knn (k: %d) (features: %d) (query: %d) (data: %d)", k, query:size(2), query:size(1), data:size(1)))
end

test.test(2, 5, 10, 10)
test.test(4, 1280, 100, 10000)
test.test(4, 100, 200, 10000)

test.test(16, 1024, 2000, 70000)

test.benchmark(2, 128, 10000, 10000)
test.benchmark(4, 128, 10000, 50000)
test.benchmark(8, 128, 50000, 50000)
test.benchmark(24, 128, 10000, 100000)
test.benchmark(16, 1024, 10000, 50000)

return test