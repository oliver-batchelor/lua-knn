package = "knn"
version = "0-1"

source = {
   url = "git://github.com/torch/nn.git",
}

description = {
   summary = "KNN cuda bindings for Torch",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/nn",
   license = "Attribution-Noncommercial-Share Alike 3.0 Unported"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}