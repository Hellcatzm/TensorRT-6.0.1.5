# TensorRT-6.0.1.5开发相关
利用TensorRT-6.0.1.5开发了一个自定义plugin，本意是适配TensorFlow2.1版本的相关节点，但是试验后发现graphsurgeon子模组在TF2版本不能顺利加载，当然了使用trt的Python或者C++接口重载整个网络肯定是行得通的，不过不是很必要，所以改成了官方推荐的TF2优化方案——TF_TRT包直接优化，这部分代码保存上来以便必要时重新上手更为快捷，NVIDIA相关资料也做了个汇总，如下：<br>
[TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#overview)<br>
[TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#overview)<br>
[Accelerating Inference In TF-TRT User Guide](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)<br>
[Using TensorRT in TensorFlow (TF-TRT)](https://github.com/tensorflow/tensorflow/tree/r2.1/tensorflow/python/compiler/tensorrt)<br>
[Documentation for TensorRT in TensorFlow (TF-TRT)](https://github.com/tensorflow/tensorrt)
