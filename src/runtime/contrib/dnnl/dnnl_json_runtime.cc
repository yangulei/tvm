/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"
#include <map>
#include <sys/time.h>

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;
class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      SetBuffer(eid);
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      SetBuffer(eid);
    }
    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      const auto& prim = net_.at(i);
      
      const auto& mem_args = net_args_.at(i);
            static const std::unordered_map<dnnl::primitive::kind, std::string> prim_name{
          {dnnl::primitive::kind::reorder, "reorder"},
          {dnnl::primitive::kind::convolution, "conv"},
          {dnnl::primitive::kind::deconvolution, "deconv"},
          {dnnl::primitive::kind::inner_product, "Dense"},
          {dnnl::primitive::kind::pooling_v2, "pool_v2"}};
      const auto prim_kind = prim.get_kind();
      if (prim_name.find(prim_kind) != prim_name.end()) {
        std::cout << "executing " << prim_name.at(prim_kind)
                  << " primitiv ===============================" << std::endl;
      } else {
        std::cout << "executing unknown primitiv ===============================" << std::endl;
      }
      std::map<int, dnnl::memory> args;
      for (const auto& kvp : mem_args) args[kvp.first] = kvp.second;
      for (const auto& arg : args) {
        std::cout << "address for arg #." << arg.first << ": " << arg.second.get_data_handle()
                  << std::endl;
      }

      net_.at(i).execute(stream_, net_args_.at(i));
    }
    stream_.wait();
  }

 private:
  tag layout2tag(std::string layout, bool is_weight = false) {
    static const std::map<std::string, tag> str2tag = {
        {"nc", tag::nc},
        {"cn", tag::cn},
        {"tn", tag::tn},
        {"nt", tag::nt},
        {"ncw", tag::ncw},
        {"nwc", tag::nwc},
        {"nchw", tag::nchw},
        {"nhwc", tag::nhwc},
        {"chwn", tag::chwn},
        {"ncdhw", tag::ncdhw},
        {"ndhwc", tag::ndhwc},
        {"oi", tag::oi},
        {"io", tag::io},
        {"oiw", tag::oiw},
        {"owi", tag::owi},
        {"wio", tag::wio},
        {"iwo", tag::iwo},
        {"oihw", tag::oihw},
        {"hwio", tag::hwio},
        {"ohwi", tag::ohwi},
        {"ihwo", tag::ihwo},
        {"iohw", tag::iohw},
        {"oidhw", tag::oidhw},
        {"dhwio", tag::dhwio},
        {"odhwi", tag::odhwi},
        {"iodhw", tag::iodhw},
        {"idhwo", tag::idhwo},
        {"goiw", tag::goiw},
        {"gowi", tag::gowi},
        {"wigo", tag::wigo},
        {"gohwi", tag::gohwi},
        {"goihw", tag::goihw},
        {"hwigo", tag::hwigo},
        {"giohw", tag::giohw},
        {"goidhw", tag::goidhw},
        {"giodhw", tag::giodhw},
        {"godhwi", tag::godhwi},
        {"dhwigo", tag::dhwigo},
        {"tnc", tag::tnc},
        {"ntc", tag::ntc},
        {"ldnc", tag::ldnc},
        {"ldigo", tag::ldigo},
        {"ldgoi", tag::ldgoi},
        {"ldio", tag::ldio},
        {"ldoi", tag::ldoi},
        {"ldgo", tag::ldgo},
        {"format_tag_last", tag::format_tag_last},
        {"nCdhw16c", tag::nCdhw16c},
        {"nCdhw4c", tag::nCdhw4c},
        {"nCdhw8c", tag::nCdhw8c},
        {"nChw16c", tag::nChw16c},
        {"nChw4c", tag::nChw4c},
        {"nChw8c", tag::nChw8c},
        {"nCw16c", tag::nCw16c},
        {"nCw4c", tag::nCw4c},
        {"nCw8c", tag::nCw8c},
        {"NCw16n16c", tag::NCw16n16c},
        {"NChw16n16c", tag::NChw16n16c},
        {"NCdhw16n16c", tag::NCdhw16n16c},
        {"NCdhw32n32c", tag::NCdhw32n32c},
        {"NChw32n32c", tag::NChw32n32c},
        {"IOhw16i16o", tag::IOhw16i16o},
        {"OI16i16o", tag::OI16i16o},
        {"OI16i32o", tag::OI16i32o},
        {"OI16i64o", tag::OI16i64o},
        {"OI8i16o2i", tag::OI8i16o2i},
        {"OI8i32o2i", tag::OI8i32o2i},
        {"OI8i64o2i", tag::OI8i64o2i},
        {"OI4i16o4i", tag::OI4i16o4i},
        {"OI4i32o4i", tag::OI4i32o4i},
        {"OI4i64o4i", tag::OI4i64o4i},
        {"Ohwi32o", tag::Ohwi32o},
        {"IOdhw16i16o", tag::IOdhw16i16o},
        {"gIOhw16i16o", tag::gIOhw16i16o},
        {"gOhwi32o", tag::gOhwi32o},
        {"Goidhw16g", tag::Goidhw16g},
        {"IOw16o16i", tag::IOw16o16i},
        {"OIw16i16o", tag::OIw16i16o},
        {"OIw16i32o", tag::OIw16i32o},
        {"OIw16i64o", tag::OIw16i64o},
        {"IOw16i16o", tag::IOw16i16o},
        {"gIOw16i16o", tag::gIOw16i16o},
        {"OIw16o16i", tag::OIw16o16i},
        {"Oiw16o", tag::Oiw16o},
        {"OIw4i16o4i", tag::OIw4i16o4i},
        {"OIw4i32o4i", tag::OIw4i32o4i},
        {"OIw4i64o4i", tag::OIw4i64o4i},
        {"OIw2i8o4i", tag::OIw2i8o4i},
        {"OIw4i4o", tag::OIw4i4o},
        {"OIw4o4i", tag::OIw4o4i},
        {"Oiw4o", tag::Oiw4o},
        {"OIw8i16o2i", tag::OIw8i16o2i},
        {"OIw8i32o2i", tag::OIw8i32o2i},
        {"OIw8i64o2i", tag::OIw8i64o2i},
        {"OIw8i8o", tag::OIw8i8o},
        {"OIw8o16i2o", tag::OIw8o16i2o},
        {"OIw8o8i", tag::OIw8o8i},
        {"OIw8o4i", tag::OIw8o4i},
        {"OIw16i16o4i", tag::OIw16i16o4i},
        {"OIw16i32o4i", tag::OIw16i32o4i},
        {"OIw16i48o4i", tag::OIw16i48o4i},
        {"OIw16i64o4i", tag::OIw16i64o4i},
        {"OIw16i16o2i", tag::OIw16i16o2i},
        {"OIw16i32o2i", tag::OIw16i32o2i},
        {"OIw16i48o2i", tag::OIw16i48o2i},
        {"OIw16i64o2i", tag::OIw16i64o2i},
        {"OIw16o16i2o", tag::OIw16o16i2o},
        {"Owi16o", tag::Owi16o},
        {"OwI16o2i", tag::OwI16o2i},
        {"Owi4o", tag::Owi4o},
        {"Owi8o", tag::Owi8o},
        {"IOhw16o16i", tag::IOhw16o16i},
        {"Ohwi16o", tag::Ohwi16o},
        {"OhwI16o2i", tag::OhwI16o2i},
        {"Ohwi4o", tag::Ohwi4o},
        {"Ohwi8o", tag::Ohwi8o},
        {"OIhw16i16o", tag::OIhw16i16o},
        {"OIhw16i32o", tag::OIhw16i32o},
        {"OIhw16i64o", tag::OIhw16i64o},
        {"OIhw16o16i", tag::OIhw16o16i},
        {"Oihw16o", tag::Oihw16o},
        {"OIhw4i16o4i", tag::OIhw4i16o4i},
        {"OIhw4i32o4i", tag::OIhw4i32o4i},
        {"OIhw4i64o4i", tag::OIhw4i64o4i},
        {"OIhw4i4o", tag::OIhw4i4o},
        {"OIhw4o4i", tag::OIhw4o4i},
        {"Oihw4o", tag::Oihw4o},
        {"OIhw8i16o2i", tag::OIhw8i16o2i},
        {"OIhw8i32o2i", tag::OIhw8i32o2i},
        {"OIhw8i64o2i", tag::OIhw8i64o2i},
        {"OIhw8i8o", tag::OIhw8i8o},
        {"OIhw8o16i2o", tag::OIhw8o16i2o},
        {"OIhw8o8i", tag::OIhw8o8i},
        {"OIhw8o4i", tag::OIhw8o4i},
        {"OIhw2i8o4i", tag::OIhw2i8o4i},
        {"IOdhw16o16i", tag::IOdhw16o16i},
        {"Odhwi16o", tag::Odhwi16o},
        {"OdhwI16o2i", tag::OdhwI16o2i},
        {"Odhwi4o", tag::Odhwi4o},
        {"Odhwi8o", tag::Odhwi8o},
        {"OIdhw16i16o", tag::OIdhw16i16o},
        {"OIdhw16i32o", tag::OIdhw16i32o},
        {"OIdhw16i64o", tag::OIdhw16i64o},
        {"OIdhw16o16i", tag::OIdhw16o16i},
        {"Oidhw16o", tag::Oidhw16o},
        {"OIdhw4i4o", tag::OIdhw4i4o},
        {"OIdhw4o4i", tag::OIdhw4o4i},
        {"Oidhw4o", tag::Oidhw4o},
        {"OIdhw8i16o2i", tag::OIdhw8i16o2i},
        {"OIdhw8i32o2i", tag::OIdhw8i32o2i},
        {"OIdhw8i64o2i", tag::OIdhw8i64o2i},
        {"OIdhw4i16o4i", tag::OIdhw4i16o4i},
        {"OIdhw16i16o4i", tag::OIdhw16i16o4i},
        {"OIdhw16i32o4i", tag::OIdhw16i32o4i},
        {"OIdhw16i48o4i", tag::OIdhw16i48o4i},
        {"OIdhw16i64o4i", tag::OIdhw16i64o4i},
        {"OIdhw16i16o2i", tag::OIdhw16i16o2i},
        {"OIdhw16i32o2i", tag::OIdhw16i32o2i},
        {"OIdhw16i48o2i", tag::OIdhw16i48o2i},
        {"OIdhw16i64o2i", tag::OIdhw16i64o2i},
        {"OIdhw4i32o4i", tag::OIdhw4i32o4i},
        {"OIdhw4i64o4i", tag::OIdhw4i64o4i},
        {"OIdhw2i8o4i", tag::OIdhw2i8o4i},
        {"OIdhw8i8o", tag::OIdhw8i8o},
        {"OIdhw8o8i", tag::OIdhw8o8i},
        {"OIdhw8o4i", tag::OIdhw8o4i},
        {"gIOw16o16i", tag::gIOw16o16i},
        {"gOIw16i16o", tag::gOIw16i16o},
        {"gOIw16o16i", tag::gOIw16o16i},
        {"gOiw16o", tag::gOiw16o},
        {"gOIw4i16o4i", tag::gOIw4i16o4i},
        {"gOIw2i8o4i", tag::gOIw2i8o4i},
        {"gOIw4i4o", tag::gOIw4i4o},
        {"gOIw4o4i", tag::gOIw4o4i},
        {"gOiw4o", tag::gOiw4o},
        {"gOIw8i16o2i", tag::gOIw8i16o2i},
        {"gOIw8i8o", tag::gOIw8i8o},
        {"gOIw8o16i2o", tag::gOIw8o16i2o},
        {"gOIw8o8i", tag::gOIw8o8i},
        {"gOIw8o4i", tag::gOIw8o4i},
        {"gOIw16i16o4i", tag::gOIw16i16o4i},
        {"gOIw16i16o2i", tag::gOIw16i16o2i},
        {"gOIw16o16i2o", tag::gOIw16o16i2o},
        {"gOwi16o", tag::gOwi16o},
        {"gOwI16o2i", tag::gOwI16o2i},
        {"gOwi4o", tag::gOwi4o},
        {"gOwi8o", tag::gOwi8o},
        {"Goiw8g", tag::Goiw8g},
        {"Goiw16g", tag::Goiw16g},
        {"gIOhw16o16i", tag::gIOhw16o16i},
        {"gOhwi16o", tag::gOhwi16o},
        {"gOhwI16o2i", tag::gOhwI16o2i},
        {"gOhwi4o", tag::gOhwi4o},
        {"gOhwi8o", tag::gOhwi8o},
        {"Goihw16g", tag::Goihw16g},
        {"gOIhw16i16o", tag::gOIhw16i16o},
        {"gOIhw16o16i", tag::gOIhw16o16i},
        {"gOihw16o", tag::gOihw16o},
        {"gOIhw4i16o4i", tag::gOIhw4i16o4i},
        {"gOIhw2i8o4i", tag::gOIhw2i8o4i},
        {"gOIhw4i4o", tag::gOIhw4i4o},
        {"gOIhw4o4i", tag::gOIhw4o4i},
        {"gOihw4o", tag::gOihw4o},
        {"Goihw8g", tag::Goihw8g},
        {"gOIhw8i16o2i", tag::gOIhw8i16o2i},
        {"gOIhw8i8o", tag::gOIhw8i8o},
        {"gOIhw8o16i2o", tag::gOIhw8o16i2o},
        {"OIw4o8i8o4i", tag::OIw4o8i8o4i},
        {"OIdhw4o8i8o4i", tag::OIdhw4o8i8o4i},
        {"OIhw4o8i8o4i", tag::OIhw4o8i8o4i},
        {"OIhw2o8i8o2i", tag::OIhw2o8i8o2i},
        {"gOIw4o8i8o4i", tag::gOIw4o8i8o4i},
        {"gOIdhw4o8i8o4i", tag::gOIdhw4o8i8o4i},
        {"gOIhw4o8i8o4i", tag::gOIhw4o8i8o4i},
        {"gOIhw2o8i8o2i", tag::gOIhw2o8i8o2i},
        {"OIhw16i16o4i", tag::OIhw16i16o4i},
        {"OIhw16i32o4i", tag::OIhw16i32o4i},
        {"OIhw16i48o4i", tag::OIhw16i48o4i},
        {"OIhw16i64o4i", tag::OIhw16i64o4i},
        {"OIhw16i16o2i", tag::OIhw16i16o2i},
        {"OIhw16i32o2i", tag::OIhw16i32o2i},
        {"OIhw16i48o2i", tag::OIhw16i48o2i},
        {"OIhw16i64o2i", tag::OIhw16i64o2i},
        {"OIhw16o16i2o", tag::OIhw16o16i2o},
        {"gOIhw16i16o4i", tag::gOIhw16i16o4i},
        {"gOIhw16i16o2i", tag::gOIhw16i16o2i},
        {"gOIhw16o16i2o", tag::gOIhw16o16i2o},
        {"gOIhw8o8i", tag::gOIhw8o8i},
        {"gOIhw8o4i", tag::gOIhw8o4i},
        {"gIOdhw16i16o", tag::gIOdhw16i16o},
        {"gIOdhw16o16i", tag::gIOdhw16o16i},
        {"gOdhwi16o", tag::gOdhwi16o},
        {"gOdhwI16o2i", tag::gOdhwI16o2i},
        {"gOdhwi4o", tag::gOdhwi4o},
        {"gOdhwi8o", tag::gOdhwi8o},
        {"gOIdhw16i16o", tag::gOIdhw16i16o},
        {"gOIdhw16o16i", tag::gOIdhw16o16i},
        {"gOidhw16o", tag::gOidhw16o},
        {"gOIdhw4i4o", tag::gOIdhw4i4o},
        {"gOIdhw4o4i", tag::gOIdhw4o4i},
        {"gOidhw4o", tag::gOidhw4o},
        {"gOIdhw8i16o2i", tag::gOIdhw8i16o2i},
        {"gOIdhw4i16o4i", tag::gOIdhw4i16o4i},
        {"gOIdhw16i16o4i", tag::gOIdhw16i16o4i},
        {"gOIdhw16i16o2i", tag::gOIdhw16i16o2i},
        {"gOIdhw2i8o4i", tag::gOIdhw2i8o4i},
        {"gOIdhw8i8o", tag::gOIdhw8i8o},
        {"gOIdhw8o8i", tag::gOIdhw8o8i},
        {"gOIdhw8o4i", tag::gOIdhw8o4i},
        {"gOIw2i4o2i", tag::gOIw2i4o2i},
        {"gOIhw2i4o2i", tag::gOIhw2i4o2i},
        {"gOIdhw2i4o2i", tag::gOIdhw2i4o2i},
        {"gOIw2o4i2o", tag::gOIw2o4i2o},
        {"gOIhw2o4i2o", tag::gOIhw2o4i2o},
        {"gOIdhw2o4i2o", tag::gOIdhw2o4i2o},
        {"gOIw4i8o2i", tag::gOIw4i8o2i},
        {"gOIhw4i8o2i", tag::gOIhw4i8o2i},
        {"gOIdhw4i8o2i", tag::gOIdhw4i8o2i},
        {"gOIw4o8i2o", tag::gOIw4o8i2o},
        {"gOIhw4o8i2o", tag::gOIhw4o8i2o},
        {"gOIdhw4o8i2o", tag::gOIdhw4o8i2o},
        {"ldOi32o", tag::ldOi32o},
        {"ldOI32o4i", tag::ldOI32o4i},
        {"ldgOi32o", tag::ldgOi32o},
        {"ldgOI32o2i", tag::ldgOI32o2i},
        {"ldgOI32o4i", tag::ldgOI32o4i},
        {"OwI16o4i", tag::OwI16o4i},
        {"OhwI16o4i", tag::OhwI16o4i},
        {"gOwI16o4i", tag::gOwI16o4i},
        {"gOhwI16o4i", tag::gOhwI16o4i},
        {"OdhwI16o4i", tag::OdhwI16o4i},
        {"gOdhwI16o4i", tag::gOdhwI16o4i},
        {"Owi32o", tag::Owi32o},
        {"OwI32o2i", tag::OwI32o2i},
        {"OwI32o4i", tag::OwI32o4i},
        {"Owi48o", tag::Owi48o},
        {"OwI48o2i", tag::OwI48o2i},
        {"OwI48o4i", tag::OwI48o4i},
        {"Owi64o", tag::Owi64o},
        {"OwI64o2i", tag::OwI64o2i},
        {"OwI64o4i", tag::OwI64o4i},
        {"wIo2i", tag::wIo2i},
        {"wIo4i", tag::wIo4i},
        {"gOwi32o", tag::gOwi32o},
        {"gOwI32o2i", tag::gOwI32o2i},
        {"gOwI32o4i", tag::gOwI32o4i},
        {"gOwi48o", tag::gOwi48o},
        {"gOwI48o2i", tag::gOwI48o2i},
        {"gOwI48o4i", tag::gOwI48o4i},
        {"gOwi64o", tag::gOwi64o},
        {"gOwI64o2i", tag::gOwI64o2i},
        {"gOwI64o4i", tag::gOwI64o4i},
        {"gwio", tag::gwio},
        {"gwIo2i", tag::gwIo2i},
        {"gwIo4i", tag::gwIo4i},
        {"OhwI32o", tag::OhwI32o},
        {"OhwI32o2i", tag::OhwI32o2i},
        {"OhwI32o4i", tag::OhwI32o4i},
        {"Ohwi48o", tag::Ohwi48o},
        {"OhwI48o2i", tag::OhwI48o2i},
        {"OhwI48o4i", tag::OhwI48o4i},
        {"Ohwi64o", tag::Ohwi64o},
        {"OhwI64o2i", tag::OhwI64o2i},
        {"OhwI64o4i", tag::OhwI64o4i},
        {"hwIo2i", tag::hwIo2i},
        {"hwIo4i", tag::hwIo4i},
        {"gOhwI32o", tag::gOhwI32o},
        {"gOhwI32o2i", tag::gOhwI32o2i},
        {"gOhwI32o4i", tag::gOhwI32o4i},
        {"gOhwi48o", tag::gOhwi48o},
        {"gOhwI48o2i", tag::gOhwI48o2i},
        {"gOhwI48o4i", tag::gOhwI48o4i},
        {"gOhwi64o", tag::gOhwi64o},
        {"gOhwI64o2i", tag::gOhwI64o2i},
        {"gOhwI64o4i", tag::gOhwI64o4i},
        {"ghwio", tag::ghwio},
        {"ghwIo2i", tag::ghwIo2i},
        {"ghwIo4i", tag::ghwIo4i},
        {"Odhwi32o", tag::Odhwi32o},
        {"OdhwI32o2i", tag::OdhwI32o2i},
        {"OdhwI32o4i", tag::OdhwI32o4i},
        {"Odhwi48o", tag::Odhwi48o},
        {"OdhwI48o2i", tag::OdhwI48o2i},
        {"OdhwI48o4i", tag::OdhwI48o4i},
        {"Odhwi64o", tag::Odhwi64o},
        {"OdhwI64o2i", tag::OdhwI64o2i},
        {"OdhwI64o4i", tag::OdhwI64o4i},
        {"dhwIo2i", tag::dhwIo2i},
        {"dhwIo4i", tag::dhwIo4i},
        {"gOdhwi32o", tag::gOdhwi32o},
        {"gOdhwI32o2i", tag::gOdhwI32o2i},
        {"gOdhwI32o4i", tag::gOdhwI32o4i},
        {"gOdhwi48o", tag::gOdhwi48o},
        {"gOdhwI48o2i", tag::gOdhwI48o2i},
        {"gOdhwI48o4i", tag::gOdhwI48o4i},
        {"gOdhwi64o", tag::gOdhwi64o},
        {"gOdhwI64o2i", tag::gOdhwI64o2i},
        {"gOdhwI64o4i", tag::gOdhwI64o4i},
        {"gdhwio", tag::gdhwio},
        {"gdhwIo2i", tag::gdhwIo2i},
        {"gdhwIo4i", tag::gdhwIo4i},
        {"ldIo32i", tag::ldIo32i},
        {"ldgIo32i", tag::ldgIo32i},
        {"ldgIO32i2o", tag::ldgIO32i2o},
        {"nCdhw32c", tag::nCdhw32c},
        {"nChw32c", tag::nChw32c},
        {"nCw32c", tag::nCw32c},
        {"NCw32n16c", tag::NCw32n16c},
        {"NChw32n16c", tag::NChw32n16c},
        {"NCdhw32n16c", tag::NCdhw32n16c},
        {"NCw32n32c", tag::NCw32n32c},
        {"OI16i16o4i", tag::OI16i16o4i},
        {"IOw8o16i2o", tag::IOw8o16i2o},
        {"IOhw8o16i2o", tag::IOhw8o16i2o},
        {"Owhi16o", tag::Owhi16o},
        {"OIdhw8o16i2o", tag::OIdhw8o16i2o},
        {"IOdhw8o16i2o", tag::IOdhw8o16i2o},
        {"Goiw4g", tag::Goiw4g},
        {"gIOw8o16i2o", tag::gIOw8o16i2o},
        {"Goiw32g", tag::Goiw32g},
        {"Goihw4g", tag::Goihw4g},
        {"gIOhw8o16i2o", tag::gIOhw8o16i2o},
        {"Goihw32g", tag::Goihw32g},
        {"gOwhi16o", tag::gOwhi16o},
        {"IOw4i8o8i4o", tag::IOw4i8o8i4o},
        {"IOhw4i8o8i4o", tag::IOhw4i8o8i4o},
        {"IOdhw4i8o8i4o", tag::IOdhw4i8o8i4o},
        {"gIOw4i8o8i4o", tag::gIOw4i8o8i4o},
        {"gIOhw4i8o8i4o", tag::gIOhw4i8o8i4o},
        {"gIOdhw4i8o8i4o", tag::gIOdhw4i8o8i4o},
        {"gOIdhw8o16i2o", tag::gOIdhw8o16i2o},
        {"gIOdhw8o16i2o", tag::gIOdhw8o16i2o},
        {"Goidhw32g", tag::Goidhw32g},
        {"OI16i32o4i", tag::OI16i32o4i},
        {"OI16i48o4i", tag::OI16i48o4i},
        {"OI16i64o4i", tag::OI16i64o4i},
        {"OI16i16o2i", tag::OI16i16o2i},
        {"OI16i32o2i", tag::OI16i32o2i},
        {"OI16i48o2i", tag::OI16i48o2i},
        {"OI16i64o2i", tag::OI16i64o2i},
        {"OwI16i16o2i", tag::OwI16i16o2i},
        {"gOwI16i16o2i", tag::gOwI16i16o2i},
        {"OhwI16i16o2i", tag::OhwI16i16o2i},
        {"gOhwI16i16o2i", tag::gOhwI16i16o2i},
        {"OdhwI16i16o2i", tag::OdhwI16i16o2i},
        {"gOdhwI16i16o2i", tag::gOdhwI16i16o2i},
        {"OwI16i16o4i", tag::OwI16i16o4i},
        {"gOwI16i16o4i", tag::gOwI16i16o4i},
        {"OhwI16i16o4i", tag::OhwI16i16o4i},
        {"gOhwI16i16o4i", tag::gOhwI16i16o4i},
        {"OdhwI16i16o4i", tag::OdhwI16i16o4i},
        {"gOdhwI16i16o4i", tag::gOdhwI16i16o4i},
        {"OwI16i32o2i", tag::OwI16i32o2i},
        {"OwI16i32o4i", tag::OwI16i32o4i},
        {"OwI16i48o2i", tag::OwI16i48o2i},
        {"OwI16i48o4i", tag::OwI16i48o4i},
        {"OwI16i64o2i", tag::OwI16i64o2i},
        {"OwI16i64o4i", tag::OwI16i64o4i},
        {"gOwI16i32o2i", tag::gOwI16i32o2i},
        {"gOwI16i32o4i", tag::gOwI16i32o4i},
        {"gOwI16i48o2i", tag::gOwI16i48o2i},
        {"gOwI16i48o4i", tag::gOwI16i48o4i},
        {"gOwI16i64o2i", tag::gOwI16i64o2i},
        {"gOwI16i64o4i", tag::gOwI16i64o4i},
        {"OhwI16i32o2i", tag::OhwI16i32o2i},
        {"OhwI16i32o4i", tag::OhwI16i32o4i},
        {"OhwI16i48o2i", tag::OhwI16i48o2i},
        {"OhwI16i48o4i", tag::OhwI16i48o4i},
        {"OhwI16i64o2i", tag::OhwI16i64o2i},
        {"OhwI16i64o4i", tag::OhwI16i64o4i},
        {"gOhwI16i32o2i", tag::gOhwI16i32o2i},
        {"gOhwI16i32o4i", tag::gOhwI16i32o4i},
        {"gOhwI16i48o2i", tag::gOhwI16i48o2i},
        {"gOhwI16i48o4i", tag::gOhwI16i48o4i},
        {"gOhwI16i64o2i", tag::gOhwI16i64o2i},
        {"gOhwI16i64o4i", tag::gOhwI16i64o4i},
        {"OdhwI16i32o2i", tag::OdhwI16i32o2i},
        {"OdhwI16i32o4i", tag::OdhwI16i32o4i},
        {"OdhwI16i48o2i", tag::OdhwI16i48o2i},
        {"OdhwI16i48o4i", tag::OdhwI16i48o4i},
        {"OdhwI16i64o2i", tag::OdhwI16i64o2i},
        {"OdhwI16i64o4i", tag::OdhwI16i64o4i},
        {"gOdhwI16i32o2i", tag::gOdhwI16i32o2i},
        {"gOdhwI16i32o4i", tag::gOdhwI16i32o4i},
        {"gOdhwI16i48o2i", tag::gOdhwI16i48o2i},
        {"gOdhwI16i48o4i", tag::gOdhwI16i48o4i},
        {"gOdhwI16i64o2i", tag::gOdhwI16i64o2i},
        {"gOdhwI16i64o4i", tag::gOdhwI16i64o4i},
        {"hwioG16g", tag::hwioG16g},
        {"NCdhw40n32c", tag::NCdhw40n32c},
        {"NChw40n32c", tag::NChw40n32c},
        {"NCw40n32c", tag::NCw40n32c},
        {"OIdhw4o8i8o2i", tag::OIdhw4o8i8o2i},
        {"OIhw4o8i8o2i", tag::OIhw4o8i8o2i},
        {"OIw4o8i8o2i", tag::OIw4o8i8o2i},
        {"gOIdhw4o8i8o2i", tag::gOIdhw4o8i8o2i},
        {"gOIhw4o8i8o2i", tag::gOIhw4o8i8o2i},
        {"gOIw4o8i8o2i", tag::gOIw4o8i8o2i},
        {"IOdhw4i8o8i2o", tag::IOdhw4i8o8i2o},
        {"IOhw4i8o8i2o", tag::IOhw4i8o8i2o},
        {"IOw4i8o8i2o", tag::IOw4i8o8i2o},
        {"gIOdhw4i8o8i2o", tag::gIOdhw4i8o8i2o},
        {"gIOhw4i8o8i2o", tag::gIOhw4i8o8i2o},
        {"gIOw4i8o8i2o", tag::gIOw4i8o8i2o},
        {"NCdhw40n16c", tag::NCdhw40n16c},
        {"NCw40n16c", tag::NCw40n16c},
        {"NChw40n16c", tag::NChw40n16c},
        {"NCw2c32n8c", tag::NCw2c32n8c},
        {"NChw2c32n8c", tag::NChw2c32n8c},
        {"NCdhw2c32n8c", tag::NCdhw2c32n8c},
        {"OIw2i8o16i4o", tag::OIw2i8o16i4o},
        {"OIhw2i8o16i4o", tag::OIhw2i8o16i4o},
        {"OIdhw2i8o16i4o", tag::OIdhw2i8o16i4o},
        {"OIw2o8i16o4i", tag::OIw2o8i16o4i},
        {"OIw2o8i16o2i", tag::OIw2o8i16o2i},
        {"IOw2i8o16i4o", tag::IOw2i8o16i4o},
        {"IOw2i8o16i2o", tag::IOw2i8o16i2o},
        {"OIhw2o8i16o4i", tag::OIhw2o8i16o4i},
        {"OIhw2o8i16o2i", tag::OIhw2o8i16o2i},
        {"IOhw2i8o16i4o", tag::IOhw2i8o16i4o},
        {"IOhw2i8o16i2o", tag::IOhw2i8o16i2o},
        {"OIdhw2o8i16o4i", tag::OIdhw2o8i16o4i},
        {"OIdhw2o8i16o2i", tag::OIdhw2o8i16o2i},
        {"IOdhw2i8o16i4o", tag::IOdhw2i8o16i4o},
        {"IOdhw2i8o16i2o", tag::IOdhw2i8o16i2o},
        {"gOIw2o8i16o2i", tag::gOIw2o8i16o2i},
        {"gIOw2i8o16i2o", tag::gIOw2i8o16i2o},
        {"gIOhw2i8o16i2o", tag::gIOhw2i8o16i2o},
        {"gIOdhw2i8o16i2o", tag::gIOdhw2i8o16i2o},
        {"gOIhw2o8i16o2i", tag::gOIhw2o8i16o2i},
        {"gOIdhw2o8i16o2i", tag::gOIdhw2o8i16o2i},
        {"gOIw2o8i16o4i", tag::gOIw2o8i16o4i},
        {"gOIhw2o8i16o4i", tag::gOIhw2o8i16o4i}
      };
    std::string key = "";
    for (const auto& c : layout) {
      if (std::isalpha(c, std::locale("C"))) {
        char lower_c = std::tolower(c);
        if (std::isupper(c) && (layout.find(lower_c) != std::string::npos)) {
          key.push_back(c);
        } else {
          key.push_back(lower_c);
        }
      } else if (std::isdigit(c)) {
        key.push_back(c);
      } else {
        LOG(FATAL) << "invalid char '" << c << "' in " << layout << std::endl;
      }
    }
    tag res_tag = str2tag.count(key) > 0 ? str2tag.at(key) : tag::any;
    return res_tag;
  };

  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("dnnl.conv2d_bias" == op_name) {
          Conv2d(nid, false, true);
        } else if ("dnnl.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("dnnl.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
        } else if ("dnnl.conv2d_bias_sum_relu" == op_name) {
          Conv2d(nid, true, true, true);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("dnnl.dense_bias_relu" == op_name) {
          Dense(nid, true, true);
        } else if ("dnnl.dense_bias" == op_name) {
          Dense(nid, true);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("add" == op_name) {
          Binary(nid, dnnl::algorithm::binary_add);
        } else if ("multiply" == op_name) {
          Binary(nid, dnnl::algorithm::binary_mul);
        } else if ("concatenate" == op_name) {
          Concat(nid);
        } else if ("nn.max_pool2d" == op_name) {
          Pooling(nid, dnnl::algorithm::pooling_max);
        } else if ("nn.avg_pool2d" == op_name) {
          Pooling(nid, dnnl::algorithm::pooling_avg);
        } else if ("nn.global_max_pool2d" == op_name) {
          Pooling(nid, dnnl::algorithm::pooling_max, true);
        } else if ("nn.global_avg_pool2d" == op_name) {
          Pooling(nid, dnnl::algorithm::pooling_avg, true);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Bind a JSON graph node entry to a DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
    }
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  void SetBuffer(uint32_t eid){
    dnnl::memory& mem = entry_out_mem_[eid].first;
    const size_t offset = entry_out_mem_[eid].second;
    const DLTensor* tensor = data_entry_[eid];
    if (tensor == nullptr || tensor->data == nullptr) {
      LOG(FATAL) << "empty data entry" << std::endl;
    }
    size_t mem_bytes = mem.get_desc().get_size();
    size_t tensor_bytes = GetDataSize(*tensor);
    void* buffer = tensor->data;
    if (mem_bytes == tensor_bytes) {
      if (offset == 0) {
        mem.set_data_handle(buffer);
      } else {
        LOG(FATAL) << "faild to fill a tensor with " << mem_bytes << " bytes using " << tensor_bytes
                   << " bytes buffer and offset=" << offset << std::endl;
      }
    } else if (mem_bytes > tensor_bytes) {
      size_t offset_in_bytes = offset * ((tensor->dtype.bits + 7) / 8);
      uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
      if (offset_in_bytes + tensor_bytes <= mem_bytes) {
        std::copy(reinterpret_cast<uint8_t*>(buffer),
                  reinterpret_cast<uint8_t*>(buffer) + tensor_bytes,
                  dst + offset_in_bytes);
      } else {
        LOG(FATAL) << "faild to fill a tensor with " << mem_bytes << " bytes using " << tensor_bytes
                   << " bytes buffer and offset=" << offset << std::endl;
      }
    }else{
      LOG(FATAL) << "faild to fill a tensor with " << mem_bytes << " bytes using " << tensor_bytes
                 << " bytes buffer" << std::endl;
    }
  }

  dt dtype_dl2dnnl(DLDataType dltype) {
    dt dnnl_type = dt::undef;
    if (dltype.code == DataType::TypeCode::kFloat) {
      if (dltype.bits == 16) {
        dnnl_type = dt::f16;
      } else if (dltype.bits == 32) {
        dnnl_type = dt::f32;
      }
    } else if (dltype.code == DataType::TypeCode::kBFloat && dltype.bits == 16) {
      dnnl_type = dt::bf16;
    } else if (dltype.code == DataType::TypeCode::kInt) {
      if (dltype.bits == 8) {
        dnnl_type = dt::s8;
      } else if (dltype.bits == 32) {
        dnnl_type = dt::s32;
      }
    } else if (dltype.code == DataType::TypeCode::kUInt && dltype.bits == 8) {
      dnnl_type = dt::u8;
    }
    if (dnnl_type == dt::undef) {
      LOG_ERROR << "unsupported datatype: code=" << dltype.code << ", bits=" << dltype.bits;
    }
    return dnnl_type;
  }

  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false, const bool has_sum = false) {
    auto node = nodes_[nid];
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    auto src_df = layout2tag(node.GetAttr<std::vector<std::string>>("data_layout")[0]);
    auto weight_df = (groups > 1) ? tag::goihw : layout2tag(node.GetAttr<std::vector<std::string>>("kernel_layout")[0]);
    std::vector<std::string> outC = node.GetAttr<std::vector<std::string>>("channels");

    //assume the default layout as NHWC
    dnnl::memory::dim N = input_shape[0],       // batch size
        IH = input_shape[1],                    // input height
        IW = input_shape[2],                    // input width
        IC = input_shape[3],                    // input channels
        OC = std::stoi(outC[0]),                // output channels
        KH = weight_shape[1],                   // weight height
        KW = weight_shape[2],                   // weight width
        PH_L = std::stoi(str_padding[0]),       // height padding: left
        PH_R = std::stoi(str_padding[2]),       // height padding: right
        PW_L = std::stoi(str_padding[1]),       // width padding: left
        PW_R = std::stoi(str_padding[3]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[1]);         // weight-wise stride
    
    if (node.GetAttr<std::vector<std::string>>("data_layout")[0].substr(0,4)=="NCHW") {
        IC = input_shape[1];
        IH = input_shape[2];
        IW = input_shape[3];
        if (input_shape.size()==5) {
          IC = IC * input_shape[input_shape.size()-1];
        }
    }
    if (node.GetAttr<std::vector<std::string>>("kernel_layout")[0].substr(0,4)=="OIHW") {
        KH = weight_shape[2];
        KW = weight_shape[3];
    }

    dnnl::memory::dim OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
                      OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};  

    // Memory descriptions.
    dt dtype = dtype_dl2dnnl(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_]);
    auto conv_src_md = dnnl::memory::desc(src_dims, dtype, src_df);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dtype, weight_df);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dtype, tag::x);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dtype, tag::any);

    // Covn2d description.
    auto conv_desc = has_bias? 
        dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r) : \ 

        dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);
    

    // Enable ReLU
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;
    if (has_sum) {
      ops.append_sum(1.f);
      attr.set_post_ops(ops);
    }
    if (has_relu) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 1.f);
      attr.set_post_ops(ops);
    }

    // check if the primitive is supported
    try {
        dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);
    } catch (dnnl::error &e) {
      LOG(FATAL) << e.what();
    }
    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    auto conv2d_src_memory = BindDNNLMemory(data_entry, conv_src_md);

    // Weight memory.
    auto conv2d_weights_memory = BindDNNLMemory(
        weight_entry, conv_weights_md);

    // Bias memory.
    auto conv2d_bias_memory = dnnl::memory(conv_bias_md, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, conv2d_bias_memory);
    }

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = dnnl::memory(conv2d_prim_desc.dst_desc(), engine_);
    if (has_sum) {
      auto dst_entry = node.GetInputs()[3];
      conv2d_dst_memory = BindDNNLMemory(dst_entry, conv2d_prim_desc.dst_desc());
    }
      BindDNNLMemory(out_entry, conv2d_dst_memory);

    // Bind memory buffers.
    if (has_bias) {
      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_BIAS, conv2d_bias_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
    } 
    else {
      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
    }
    
  }

  void Dense(const size_t& nid, const bool has_bias = false, const bool has_relu=false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim B = input_shape[0],  // batch size
        IC = input_shape[1],               // input channels
        OC = weight_shape[0];              // output channels

    // Memory shapes.
    dnnl::memory::dims data_dims = {B, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    dt dtype = dtype_dl2dnnl(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_]);
    auto data_md = dnnl::memory::desc({data_dims, dtype, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dtype, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dtype, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dtype, tag::nc});

    // Dense description.
    auto dense_desc = has_bias? 
        dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md) : \
        dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, dst_md);

    // Enable ReLU
    dnnl::post_ops ops;
    if (has_relu) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 1.f);
    }
    dnnl::primitive_attr attr;
    attr.set_post_ops(ops);
    
    // check if the primitive is supported
    try {
        dnnl::inner_product_forward::primitive_desc(dense_desc, attr, engine_);
    } catch (dnnl::error &e) {
      LOG(FATAL) << e.what();
    }

    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, attr, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    auto bias_memory = dnnl::memory(bias_md, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, bias_memory);
    } 
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());

    if (has_bias) {
      net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
    }
    else {
      net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_DST, dst_memory}});
    }
    
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    dnnl::memory::dims data_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim IC = data_shape[1];
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    // Memory description.
    dt dtype = dtype_dl2dnnl(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_]);
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dtype);

    // BN description.
    auto bn_desc = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, data_md, epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);        
    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(bn_desc, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_prim_desc);
    net_.push_back(bn);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);
    auto mean_memory = BindDNNLMemory(mean_entry, bn_prim_desc.mean_desc());
    auto variance_memory = BindDNNLMemory(variance_entry, bn_prim_desc.variance_desc());

    // In DNNL, weight is composed of gamma+beta, so we point them to the same DNNL memory but
    // assign an offset to beta data for runtime serialization.
    auto weight_memory = BindDNNLMemory(gamma_entry, bn_prim_desc.weights_desc(), 0);
    BindDNNLMemory(beta_entry, weight_memory, IC);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_DST, out_memory},
                         {DNNL_ARG_SCALE_SHIFT, weight_memory},
                         {DNNL_ARG_MEAN, mean_memory},
                         {DNNL_ARG_VARIANCE, variance_memory}});
  }

  void Relu(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dt dtype = dtype_dl2dnnl(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_]);
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(shape, dtype);

    auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::eltwise_relu, data_md, 0);
    auto relu_prim_desc = dnnl::eltwise_forward::primitive_desc(relu_desc, engine_);
    ICHECK(data_md == relu_prim_desc.dst_desc());

    auto relu = dnnl::eltwise_forward(relu_prim_desc);
    net_.push_back(relu);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Binary(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];

    // Memory and compute description.
    std::vector<dnnl::memory::dims> data_dims;
    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;

    ICHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dt dtype = dtype_dl2dnnl(nodes_[entry.id_].GetOpDataType()[entry.index_]);
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dtype);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));
    }
    ICHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    auto binary_desc = dnnl::binary::desc(algo, data_mds[0], data_mds[1], out_md);
    auto binary_prim_desc = dnnl::binary::primitive_desc(binary_desc, engine_);
    auto binary = dnnl::binary(binary_prim_desc);
    net_.push_back(binary);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

  void Concat(const size_t& nid) {
    auto node = nodes_[nid];

    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;
    
    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]); // axis
    
    if (axis < 0) {
        ICHECK_LE(0 - axis, shape.size());
        axis = shape.size() + axis;
    }
    else ICHECK_LT(axis, shape.size());
    
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dt dtype = dtype_dl2dnnl(nodes_[entry.id_].GetOpDataType()[entry.index_]);
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dtype);
        data_mds.push_back(data_md);
        data_memories.push_back(BindDNNLMemory(entry, data_md));
      }
    
    auto concat_prim_desc = dnnl::concat::primitive_desc(axis, data_mds, engine_);
    auto concat = dnnl::concat(concat_prim_desc);
    net_.push_back(concat);
    
    auto out_md = concat_prim_desc.dst_desc();
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    std::unordered_map<int, dnnl::memory> args;
    for (int i = 0; i < (int)data_memories.size(); i++) {
        args.insert({DNNL_ARG_MULTIPLE_SRC + i, data_memories[i]});
    }
    args.insert({DNNL_ARG_DST, out_memory});
    net_args_.push_back(args);
  }

  void Pooling(const size_t& nid, dnnl::algorithm algo, const bool isGlobal = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    using VecStr = std::vector<std::string>;
    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim pool_size0 = isGlobal ? 0 : std::stoi(node.GetAttr<VecStr>("pool_size")[0]);
    dnnl::memory::dim pool_size1 = isGlobal ? 0 : std::stoi(node.GetAttr<VecStr>("pool_size")[1]);
    VecStr str_strides = isGlobal ? VecStr(2, "1") : node.GetAttr<VecStr>("strides");
    VecStr str_padding = isGlobal ? VecStr(4, "0") : node.GetAttr<VecStr>("padding");
    VecStr ceil_mode = isGlobal ? VecStr(1, "false") : node.GetAttr<VecStr>("ceil_mode");
    dnnl::memory::dim dilation0 = isGlobal ? 0 : std::stoi(node.GetAttr<VecStr>("dilation")[0]);
    dnnl::memory::dim dilation1 = isGlobal ? 0 : std::stoi(node.GetAttr<VecStr>("dilation")[1]);
    auto src_df = layout2tag(node.GetAttr<VecStr>("layout")[0]);

    // Attributes related to AvgPool
    if (!isGlobal && algo == dnnl::algorithm::pooling_avg) {
      int int_countpad = std::stoi(node.GetAttr<VecStr>("count_include_pad")[0]);
      bool count_include_pad = int_countpad != 0 ? true : false;
      algo = count_include_pad ? dnnl::algorithm::pooling_avg_include_padding
                               : dnnl::algorithm::pooling_avg_exclude_padding;
    }

    //assume the default layout as NHWC
    dnnl::memory::dim N = input_shape[0],
      IH = input_shape[1],
      IW = input_shape[2],
      IC = input_shape[3],
      KH = pool_size0,
      KW = pool_size1,
      PW_L = std::stoi(str_padding[1]),
      PW_R = std::stoi(str_padding[3]),
      PH_L = std::stoi(str_padding[0]),
      PH_R = std::stoi(str_padding[2]),
      SH = std::stoi(str_strides[0]),
      SW = std::stoi(str_strides[1]),
      DH = dilation0,
      DW = dilation1;
      
    if (node.GetAttr<VecStr>("layout")[0].substr(0,4)=="NCHW") {
        IC = input_shape[1];
        IH = input_shape[2];
        IW = input_shape[3];
        if (input_shape.size()==5) {
          IC = IC * input_shape[input_shape.size()-1];
        }
    }

    if (isGlobal) {
      KH = IH;
      KW = IW;
    }

    dnnl::memory::dim OH = (IH - KH + PH_L + PH_R) / SH + 1,
                      OW = (IW - KW + PW_L + PW_R) / SW + 1;
    
    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims kernel_dims = {KH, KW}; // modified
    dnnl::memory::dims dst_dims = {N, IC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};
    dnnl::memory::dims dilation = {DH, DW};

    // Memory descriptions.
    dt dtype = dtype_dl2dnnl(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_]);
    auto pool_src_md = dnnl::memory::desc(src_dims, dtype, src_df);
    auto pool_dst_md = dnnl::memory::desc(dst_dims, dtype, tag::any);

    // Pooling description.
    auto pool_desc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, algo,
      pool_src_md, pool_dst_md, strides_dims, kernel_dims,
      padding_dims_l, padding_dims_r
    );
    auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(pool_desc, engine_, true);
    auto pool = dnnl::pooling_forward(pool_prim_desc);
    net_.push_back(pool);

    // Memories.
    auto pool2d_src_memory = BindDNNLMemory(data_entry, pool_src_md);

    JSONGraphNodeEntry out_entry(nid, 0);
    auto pool2d_dst_memory = BindDNNLMemory(out_entry, pool_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({
      {DNNL_ARG_SRC, pool2d_src_memory},
      {DNNL_ARG_DST, pool2d_dst_memory}
    });
  }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
  }

  // Generate DNNL memory description and infer the data layout by the given shape.
  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
    dnnl::memory::desc data_md;
    switch (shape.size()) {
      case 1:
        data_md = dnnl::memory::desc({shape, dtype, tag::a});
        break;
      case 2:
        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::pair<dnnl::memory, size_t>> entry_out_mem_;
};

runtime::Module DNNLJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
