'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
import mxnet as mx
import gluoncv

import warnings
warnings.filterwarnings("ignore")
import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay

import numpy as np
import os
from tvm.contrib import utils
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.download import download_testdata
from PIL import Image

network_dict = {"resnet18":"ResNet18_v1b",
                "resnet34":"ResNet34_v1b",
                "resnet50":"ResNet50_v1b",
                "resnet101":"ResNet101_v1b",
                "resnet152":"ResNet152_v1b",
                "vgg11":"VGG11",
                "vgg13":"VGG13",
                "vgg16":"VGG16",
                "vgg19":"VGG19",
                "VGG11_bn":"VGG11_bn",
                "VGG13_bn":"VGG13_bn",
                "VGG16_bn":"VGG16_bn",
                "VGG19_bn":"VGG19_bn",}

translate_dict = {"abcd":"NCHW",
                "Acdb8a": "OHWI8o",
                "Acdb16a": "OHWI16o",
                "ABcd8b8a": "OIHW8i8o",
                "ABcd16b16a": "OIHW16i16o",
                "aBcd8b": "NCHW8c",
                "aBcd16b": "NCHW16c",}

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """Simple test function to replace one argument to another."""

    def __init__(self):
        # self.multiplier = multiplier
        self.cnt = 0
        self.merge_dict = {}
        self.branch_dict = {}
        self.block_dict = {}
        self.op_lst = []

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        self.merge_consecutive_add(func.body)
        res = self.rewrite_graph()
        res = relay.Function([self.op_lst[-1]], res)
        return res

    def rewrite_graph(self):
        try:
            if(max(self.block_dict.keys())<max(self.merge_dict.keys())):
                start = len(self.op_lst)-1
            else:
                start = max(self.block_dict.keys())
        except:
            start = max(self.block_dict.keys())
        new_node = self.op_lst[start]
        cur_node = self.op_lst[start]
        for i in range(start-1, -1, -1):
            
            cur_node = self.op_lst[i]

            if i+1 in self.block_dict.keys():
                node_for_next_block = new_node
            
            if i in self.branch_dict.keys():
                branch_lst = self.branch_dict[i]
                tmp_new_node = node_for_next_block
                for j in range(len(branch_lst)-2, -1, -1):
                    tmp_cur_node = branch_lst[j]
                    tmp_new_node = self.get_op(tmp_cur_node, tmp_new_node, tmp_cur_node.args[1])
                new_node = self.get_op(cur_node, new_node, tmp_new_node)

            elif i in self.merge_dict.keys():
                new_node = self.get_op(cur_node, new_node, self.merge_dict[i])

            elif cur_node.op.name=="add":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1])
            elif cur_node.op.name=="nn.conv2d":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1], cur_node.attrs)
            elif cur_node.op.name=="nn.dense":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1])
            elif cur_node.op.name=="nn.batch_flatten":
                new_node = self.get_op(cur_node, new_node)
            else:
                new_node = self.get_op(cur_node, new_node)
        return new_node

    def merge_consecutive_add(self, node):
        while node:
            try:
                if self.check_block(node):
                    self.block_dict[self.cnt] = node
                
                elif self.check_consecutive_add(node):
                    a1 = node
                    a2 = a1.args[0]
                    data = relay.add(a1.args[1], a2.args[1])
                    self.merge_dict[self.cnt] = data
                    node = a2

                elif self.check_branch(node):
                    tmp_node = node.args[1]
                    self.branch_lst = [tmp_node]
                    while (not self.check_block(tmp_node)):
                        tmp_node = tmp_node.args[0]
                        self.branch_lst.append(tmp_node)
                    self.branch_dict[self.cnt] = self.branch_lst
                
                self.op_lst.append(node)
                node = node.args[0]
                self.cnt += 1
            except:
                self.cnt = 0
                break

    def check_consecutive_add(self, node):
        try:
            return node.op.name=='add' and len(node.type_args[1].shape)==3 and node.args[0].op.name=='add' and len(node.args[0].type_args[1].shape)==3
        except:
            return False
    
    def check_block(self, node):
        try:
            return (node.op.name=='nn.relu' and not self.check_constant(node.args[0].args[1])) or node.op.name=='nn.max_pool2d'
        except:
            return False

    def check_branch(self, node):
        try:
            return node.op.name=='add' and not self.check_constant(node.args[1])
        except:
            return False

    def check_constant(self, node):
        try:
            return 'Constant' in str(type(node))
        except:
            return False

    def get_op(self, node, *args):
        if node.op.name=='nn.conv2d':
            return relay.nn.conv2d(args[0], args[1], **node.attrs)
        elif node.op.name=='nn.relu':
            return relay.nn.relu(args[0])
        elif node.op.name=='add':
            return relay.add(args[0], args[1])
        elif node.op.name=='nn.max_pool2d':
            return relay.nn.max_pool2d(args[0], **node.attrs)
        elif node.op.name=='nn.global_avg_pool2d':
            return relay.nn.global_avg_pool2d(args[0], **node.attrs)
        elif node.op.name=='nn.batch_flatten':
            return relay.nn.batch_flatten(args[0])
        elif node.op.name=='nn.dense':
            return relay.nn.dense(args[0], args[1], **node.attrs)
        else:
            return False

@relay.op.register_alter_op_layout("nn.conv2d", level=114)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    
    def get_shape(tensor):
        if 'Var' in str(type(tensor)):
            return tensor.type_annotation.concrete_shape
        elif 'Constant' in str(type(tensor)):
            return tensor.data.shape
        elif 'TensorType' in str(type(tensor)):
            return tensor.concrete_shape
        else:
            return (-1, -1, -1, -1)
    
    N, IC, IH, IW = get_shape(data)
    OC, IC, KH, KW = get_shape(weight)
    N, _, OH, OW = get_shape(out_type)
    PH_L, PH_R, PW_L, PW_R = attrs.padding
    PH_L, PH_R, PW_L, PW_R = int(PH_L), int(PH_R), int(PW_L), int(PW_R)
    SH, SW = attrs.strides
    SH, SW = int(SH), int(SW)

    from tvm import relay
    res = relay.query_layout.AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW)

    new_attrs = dict(attrs)

    src_df, weight_df, dst_df = res.split(',')

    new_attrs['data_layout'] = translate_dict[src_df]
    new_attrs['kernel_layout'] = translate_dict[weight_df]
    new_attrs['out_layout'] = translate_dict[dst_df]
    return relay.nn.conv2d(data, weight, **new_attrs)

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def benchmark(network, batch_size, profiling=False, check_acc=False, warmup=100, batches=400, dtype="float32", target="llvm"):
    ctx = tvm.cpu()

    input_shape = (batch_size, 3, 224, 224)
    block = gluoncv.model_zoo.get_model(network_dict[network], pretrained=True)
    mod, params = relay.frontend.from_mxnet(
        block, shape={"data": input_shape}, dtype=dtype
    )

    seq = tvm.transform.Sequential(
        [   
            # tvm.transform.PrintIR(),
            relay.transform.CanonicalizeOps(),
            relay.transform.InferType(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            # tvm.transform.PrintIR(),
            CustomPipeline(),
            relay.transform.FoldConstant(),
            
            relay.transform.AlterOpLayout(),
            
            relay.transform.MergeComposite(pattern_table()),
            relay.transform.AnnotateTarget("dnnl"),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(),
            # tvm.transform.PrintIR(),
        ]
    )


    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    with tvm.transform.PassContext(opt_level=3):#, instruments=[PrintIR()]):# 
        json, lib, params = relay.build(seq(mod), target=target, params=params)

    if check_acc:
        img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
        img_name = "cat.png"
        img_path = download_testdata(img_url, img_name, module="data")
        image = Image.open(img_path).resize((224, 224))
        sample = transform_image(image)

        import tvm.contrib.graph_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
    
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        out = rt_mod.run()
        sample_for_mxnet = mx.ndarray.array(sample)
        mxnet_output = block(sample_for_mxnet)
        tvm_output = rt_mod.get_output(0)
        # print("mxnet_output:{}".format(mxnet_output))
        # print("tvm_output:{}".format(tvm_output))
        print("mse:{}".format(np.mean((tvm_output.asnumpy()-mxnet_output.asnumpy())**2)))
    elif profiling:
        from tvm.contrib.debugger import debug_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
        sample = np.random.rand(batch_size, 3, 224, 224)#np.ones((batch_size, 3, 224, 224))#
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        total_time_lst = []
        for i in range(batches+warmup):
            tmp = rt_mod.profile()
            gap = tmp.calls[1]["Duration (us)"].microseconds
            #percent = tmp.calls[0]["Percent"].percent
            reorder = tmp.calls[2]["Duration (us)"].microseconds
            #total_time = us * 100 / percent / 1000
            print("{}/{}: gap:{:.4f}, reorder:{:.4f}".format(i, batches+warmup, gap, reorder))
            total_time = gap+reorder
            total_time_lst.append(total_time)
        print("all ops' execution time:{}".format(np.mean(total_time_lst[warmup::])))
        print("all ops' execution time:{}".format(np.mean(total_time_lst[warmup::])/1000))
        print("profiling time:{}".format(datetime.datetime.now()-tic))
    
    else:
        import tvm.contrib.graph_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)
        sample = np.random.rand(batch_size, 3, 224, 224)#np.ones((batch_size, 3, 224, 224))#
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        for i in range(batches+warmup):
            if i == warmup:
                tic = time.time()
            out = rt_mod.run()
        with_fuse_fps = batches * batch_size / (time.time() - tic)
        print("{}: with_fuse_fps: {:.4f} fps".format(network, with_fuse_fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet50", "resnet18", "resnet34", "resnet101", "resnet152",
                "vgg11", "vgg13", "vgg16", "vgg19", 
                "VGG11_bn", "VGG13_bn", "VGG16_bn", "VGG19_bn"],
        default="VGG19_bn",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=platinum-8124m -mcpu=skylake-avx512",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batches", type=int, default=400)
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--check_acc", type=bool, default=False)
    args = parser.parse_args()

    target = tvm.target.Target(args.target)

    # Benchmark
    benchmark(args.network, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
     warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)
