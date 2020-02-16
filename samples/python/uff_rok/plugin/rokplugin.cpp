#include<cmath>
#include <vector>
#include <string>
#include <cassert>
#include "NvInferPlugin.h"

using namespace nvinfer1;

namespace
{
const char* REGIONOFKEYPOINTS_PLUGIN_VERSION{"1"};
const char* REGIONOFKEYPOINTS_PLUGIN_NAME{"RegionOfKeypoints_TRT"};
}

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

class RegionOfKeypointsPlugin : public IPluginV2
{
public:
    // 构建方法
    RegionOfKeypointsPlugin(int region_shape=5)
        : region_shape(region_shape)
    {
    };

    // 构建方法，用于clone方法
    RegionOfKeypointsPlugin(int region_shape, int image_shape, int region_num)
        : region_shape(region_shape)
        , image_shape(image_shape)
        , region_num(region_num)  
    {
    };
    
    // 构建方法，用于deserializePlugin方法
    RegionOfKeypointsPlugin(const void* data, size_t length)
    {
        // Deserialize in the same order as serialization
        const char *d = static_cast<const char *>(data);
        const char *a = d;

        region_shape = readFromBuffer<int>(d);
        image_shape = readFromBuffer<int>(d);
        region_num = readFromBuffer<int>(d);

        assert(d == (a + length));
    };
    
    // 输出个数
    int getNbOutputs() const override
    {
        return 1;
    };

    // 输出size和格式
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    /* 
    Get the dimension of an output tensor. This function is called by the implementations 
    of INetworkDefinition and IBuilder. In particular, it is called prior to any call to 
    initialize(). 

    Parameters
        index	The index of the output tensor.
        inputs	The input tensors.
        nbInputDims	The number of input tensors.
    */
    {
        // two input needed
        assert(nbInputDims == 2);
        assert(inputs[0].nbDims == 3 && inputs[0].d[0]==1);
        assert(inputs[1].nbDims == 2);
        // only have one output
        assert(index == 0);
        // input[0]取[n, c, h, w]
        // input[1]取[n, 2, k]，由于不同的图片k不一致，故n只能取1
        // output为[n, k, 5, 5]
        return Dims3(inputs[1].d[1], region_shape, region_shape);
    };
 
    // 推理前初始化
    int initialize() override
    {
        // 状态码0表示成功
        return 0;
    };

    // 释放初始化时占用资源
    void terminate() override {};

    // 计算工作区大小
    size_t getWorkspaceSize(int maxBatchSize) const override
    {
        // The operation is done in place, it doesn't use GPU memory
        return 0;
    }

    // 执行推理
    int enqueue(int batchSize, const void* const* inputs, void** outputs, 
                void* workspace, cudaStream_t stream) override
    {
        const float* images = reinterpret_cast<const float*>(inputs[0]);  // [n, c, h, w]
        const int* coords = reinterpret_cast<const int*>(inputs[1]);      // [n, 2, k]
        float* regions = reinterpret_cast<float*>(outputs[0]);            // [n, k, 5, 5]
        // loop batch size
        for (int batch_idx = 0; batch_idx<batchSize; ++batch_idx)
        {
            // loop regions
            for (int k=0; k<region_num; ++k) { 
                int co_x_lt = coords[batch_idx*region_num*2 + k] - region_shape/2;
                int co_y_lt = coords[batch_idx*region_num*2 + region_num + k] - region_shape/2;
                // loop region position
                for (int reg_x=0; reg_x<region_shape; ++reg_x) {
                    for (int reg_y=0; reg_y<region_shape; ++reg_y) {
                        int co_x = co_x_lt + reg_x;
                        int co_y = co_y_lt + reg_y;
                        int idx_reg = pow(region_shape, 2) * (batch_idx*region_num + k) + \
                                      region_shape * reg_x + reg_y;
                        if (co_x<0 || co_y<0)
                            regions[idx_reg] = 0.;
                        else {
                            int idx_img = pow(image_shape, 2) * (batch_idx*1 + k) + \
                                          image_shape * co_x + co_y;
                            regions[idx_reg] = images[idx_img];
                        }
                    }
                }
            }
        }
        return 0;
    };

    // 计算序列化空间要求
    size_t getSerializationSize() const override
    {
        size_t size = sizeof(image_shape) + sizeof(region_num) + sizeof(region_shape);
        return size;
    };

    // 序列化层
    void serialize(void* buffer) const override
    {
        char *d = static_cast<char *>(buffer);
        const char *a = d;

        writeToBuffer(d, region_shape);
        writeToBuffer(d, image_shape);
        writeToBuffer(d, region_num);

        assert(d == a + getSerializationSize());
    };

    // 在initialize之前被builder调用，用于根据相关信息调整算法参数
    void configureWithFormat(const Dims* inputDims, int nbInputs, 
                             const Dims* outputDims, int nbOutputs, 
                             DataType type, PluginFormat format, int maxBatchSize) override
    /*
    Configure the layer. This function is called by the builder prior to initialize(). It provides 
    an opportunity for the layer to make algorithm choices on the basis of its weights, dimensions, 
    and maximum batch size.

    Parameters
        inputDims	The input tensor dimensions.
        nbInputs	The number of inputs.
        outputDims	The output tensor dimensions.
        nbOutputs	The number of outputs.
        type	The data type selected for the engine.
        format	The format selected for the engine.
        maxBatchSize	The maximum batch size.
    */
    {
        assert(nbInputs == 2);
        region_num = inputDims[1].d[0];
        image_shape = inputDims[1].d[1];
    };

    // 类型检查
    bool supportsFormat(DataType type, PluginFormat format) const override
    /*
    Check format support. This function is called by the implementations of 
    INetworkDefinition, IBuilder, and safe::ICudaEngine/ICudaEngine. In particular, 
    it is called when creating an engine and when deserializing an engine.

    Parameters
        type	DataType requested.
        format	PluginFormat requested.
    Returns
        true if the plugin supports the type-format combination.
    */
    {
        // 不知道怎么实现
        return true;
    };

    // Plugin名称
    const char* getPluginType() const override
    {
        return REGIONOFKEYPOINTS_PLUGIN_NAME;
    };

    // Plugin版本
    const char* getPluginVersion() const override
    {
        return REGIONOFKEYPOINTS_PLUGIN_VERSION;
    };

    // This gets called when the network containing plugin is destroyed
    void destroy() override
    {
        delete this;
    };

    nvinfer1::IPluginV2* clone() const override
    /*
    Clone the plugin object. This copies over internal plugin parameters and returns 
    a new plugin object with these parameters.
    */
    {
        return new RegionOfKeypointsPlugin(region_shape, image_shape, region_num);
    };

    void setPluginNamespace(const char* pluginNamespace) override
    {
        mNamespace = pluginNamespace;
    };

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    };

private:
    int region_shape;
    int image_shape;
    int region_num;
    std::string mNamespace;
};

class RegionOfKeypointsCreator : public IPluginCreator
{
public:
    RegionOfKeypointsCreator(){
        mPluginAttributes.emplace_back(PluginField("region_shape", nullptr, PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    };

    const char* getPluginName() const override
    {
        return REGIONOFKEYPOINTS_PLUGIN_NAME;
    };

    const char* getPluginVersion() const override
    {
        return REGIONOFKEYPOINTS_PLUGIN_VERSION;
    };

    const PluginFieldCollection* getFieldNames() override
    {
        return &mFC;
    };

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        int region_shape;
        const PluginField* fields = fc->fields;

        // Parse fields from PluginFieldCollection
        assert(fc->nbFields == 1);
        assert(fields[0].type == PluginFieldType::kINT32);
        region_shape = *(static_cast<const int*>(fields[0].data));
        return new RegionOfKeypointsPlugin(region_shape);
    };

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {
        return new RegionOfKeypointsPlugin(serialData, serialLength);  // 必须使用new？
    };

    void setPluginNamespace(const char* pluginNamespace) override
    {
        mNamespace = pluginNamespace;
    };

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    };

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

PluginFieldCollection RegionOfKeypointsCreator::mFC{};
std::vector<PluginField> RegionOfKeypointsCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(RegionOfKeypointsCreator);